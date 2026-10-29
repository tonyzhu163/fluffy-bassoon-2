"""Verify a SLURM GPU step or an explicitly selected GPU on a standalone host."""

import argparse
import csv
import errno
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import stat
import subprocess
import sys


def visible_tokens(environment):
    raw = environment.get("CUDA_VISIBLE_DEVICES", "")
    tokens = [value.strip() for value in raw.split(",")]
    if not raw or any(not value or value == "-1" for value in tokens) or len(set(tokens)) != len(tokens):
        raise ValueError("CUDA_VISIBLE_DEVICES must list distinct allocated GPU indices/UUIDs")
    if any(not (value.isdigit() or value.startswith("GPU-")) for value in tokens):
        raise ValueError("This experiment requires complete GPUs; unsupported visibility token")
    return tokens


def command(arguments):
    return subprocess.check_output(arguments, text=True, timeout=30).strip()


def nvidia_query(arguments):
    """Keep healthy rows when another card reports the known NVML handle fault."""
    result = subprocess.run(arguments, text=True, timeout=30, capture_output=True)
    lines = (result.stdout + '\n' + result.stderr).splitlines()
    fault = 'Unable to determine the device handle for GPU'
    faults = [line for line in lines if line.startswith(fault)]
    if (result.returncode and not faults) or any(
            line.strip() and not line.startswith(fault) for line in result.stderr.splitlines()):
        raise subprocess.CalledProcessError(result.returncode or 1, arguments, result.stdout, result.stderr)
    return '\n'.join(line for line in lines if line.strip() and not line.startswith(fault))


def gpu_inventory():
    raw = nvidia_query(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,pci.bus_id",
                   "--format=csv,noheader,nounits"])
    rows = []
    for fields in csv.reader(raw.splitlines(), skipinitialspace=True):
        if len(fields) != 7:
            raise ValueError("Unrecognized nvidia-smi GPU inventory")
        index, uuid, name, total, used, utilization, pci = (field.strip() for field in fields)
        rows.append({"index": int(index), "uuid": uuid, "name": name,
                     "memory_total_mib": float(total), "memory_used_mib": float(used),
                     "utilization_percent": float(utilization), "pci_bus_id": pci})
    if not rows:
        raise ValueError('nvidia-smi reported no usable GPU')
    return rows


def compute_processes():
    raw = nvidia_query(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                   "--format=csv,noheader,nounits"])
    rows = []
    for fields in csv.reader(raw.splitlines(), skipinitialspace=True):
        if not fields or not fields[0].strip():
            continue
        if len(fields) != 4:
            raise ValueError("Cannot verify GPU exclusivity from nvidia-smi process inventory")
        uuid, pid, name, memory = (field.strip() for field in fields)
        rows.append({"gpu_uuid": uuid, "pid": int(pid), "process_name": name, "memory_used_mib": memory})
    return rows


def canonical_full_uuid(value):
    """Compare PyTorch's bare UUID with nvidia-smi's GPU-prefixed spelling."""
    value = str(value).lower()
    if value.startswith("gpu-"):
        value = value[4:]
    if re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", value) is None:
        raise ValueError(f"Expected a complete GPU UUID, got {value!r}")
    return value


def match_cuda_device_uuid(cuda_uuid, inventory):
    """Require one exact physical UUID match, retaining nvidia-smi's record."""
    expected = canonical_full_uuid(cuda_uuid)
    matches = [gpu for gpu in inventory if canonical_full_uuid(gpu["uuid"]) == expected]
    if len(matches) != 1:
        raise ValueError(f"CUDA UUID {cuda_uuid!r} is not uniquely identifiable in nvidia-smi")
    return matches[0]


def canonical_pci_bus_id(value):
    match = re.fullmatch(r"([0-9a-fA-F]+):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])", value.strip())
    if match is None:
        raise ValueError(f"Cannot identify PCI bus address {value!r}")
    domain, bus, device, function = match.groups()
    return f"{int(domain, 16):04x}:{bus.lower()}:{device.lower()}.{function}"


def expand_gres_gpu_files(text, hostname):
    """Read the explicit, increasing NVIDIA File order used by these clusters.

    SLURM GPU bitmap IDs index the configured File list, not device suffixes
    or the possibly cgroup-renumbered NVML inventory. Unsupported automatic/MIG layouts fail
    closed instead of inventing a physical mapping.
    """
    files = []
    for raw_line in text.splitlines():
        words = shlex.split(raw_line, comments=True)
        if not words:
            continue
        fields = dict(word.split("=", 1) for word in words)
        node = fields.get("NodeName")
        if node not in (None, "ALL", hostname, hostname.split(".")[0]):
            if "[" in node or "," in node:
                raise ValueError("GRES NodeName hostlists need an explicitly verified local mapping")
            continue
        if fields.get("AutoDetect", "off").lower() != "off":
            raise ValueError("GPU identity verification requires explicit GRES File entries, not AutoDetect")
        if fields.get("Name") != "gpu":
            continue
        if "MultipleFiles" in fields or "CountOnly" in fields.get("Flags", ""):
            raise ValueError("GPU identity verification does not support MIG or CountOnly GRES")
        expression = fields.get("File", "")
        match = re.fullmatch(r"/dev/nvidia(\d+|\[([\d,-]+)\])", expression)
        if match is None:
            raise ValueError(f"Unsupported explicit GPU device-file expression: {expression!r}")
        numbers = []
        for item in (match.group(2) or match.group(1)).split(","):
            if re.fullmatch(r"\d+-\d+", item):
                first, last = map(int, item.split("-"))
                if last < first:
                    raise ValueError("Reversed GRES device-file range")
                numbers.extend(range(first, last + 1))
            elif item.isdigit():
                numbers.append(int(item))
            else:
                raise ValueError("Invalid GRES device-file range")
        if "Count" in fields and (not fields["Count"].isdigit() or int(fields["Count"]) != len(numbers)):
            raise ValueError("GRES GPU Count disagrees with its explicit device files")
        files.extend(f"/dev/nvidia{number}" for number in numbers)
    numbers = [int(path.removeprefix("/dev/nvidia")) for path in files]
    if not files or numbers != sorted(set(numbers)):
        raise ValueError("Require unique GRES GPU File entries in increasing numeric order")
    return files


def parse_proc_gpu_information(text):
    fields = {}
    for line in text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
    return {"uuid": "GPU-" + canonical_full_uuid(fields["GPU UUID"]),
            "minor_number": int(fields["Device Minor"]),
            "pci_bus_id": canonical_pci_bus_id(fields["Bus Location"])}


def build_global_gpu_inventory(files, proc_records, device_minors):
    by_minor = {}
    for record in proc_records:
        if record["minor_number"] in by_minor:
            raise ValueError("Physical NVIDIA metadata repeats a device minor")
        by_minor[record["minor_number"]] = record
    result = []
    # Slurm GPU bitmap indexes follow configured-file order, including across filename gaps.
    for global_id, path in enumerate(files):
        minor = device_minors[path]
        if minor not in by_minor:
            raise ValueError(f"GRES device {path} has no matching NVIDIA physical UUID metadata")
        result.append({**by_minor[minor], "index": global_id, "slurm_global_index": global_id,
                       "device_file": path})
    if len({record["uuid"] for record in result}) != len(result):
        raise ValueError("GRES device files alias one physical GPU more than once")
    return result


def validate_kernel_allocation(actual_uuids, accessible_uuids, environment, cgroup):
    """Use enforced device access, not guessed NVML/global ordinals."""
    job, step = environment.get("SLURM_JOB_ID"), environment.get("SLURM_STEP_ID")
    if not job or not step or f"/job_{job}/step_{step}/" not in cgroup:
        raise ValueError("Missing matching Slurm job/step kernel cgroup")
    counts = []
    for key in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS"):
        ids = set()
        for token in environment.get(key, "").split(","):
            if re.fullmatch(r"\d+-\d+", token):
                lo, hi = map(int, token.split("-")); ids.update(range(lo, hi + 1))
            elif token.isdigit():
                ids.add(int(token))
            else:
                raise ValueError("Unsupported configless allocation GPU identifiers")
        counts.append(len(ids))
    if counts != [len(actual_uuids)] * 2 or len(set(actual_uuids)) != len(actual_uuids):
        raise ValueError("Configless GPU counts disagree with job/step allocation")
    if set(actual_uuids) != set(accessible_uuids):
        raise ValueError("CUDA GPUs differ from kernel-authorized GPU device files")


def delta_configless_inventory(config):
    """Delta's root-owned NVML config has no explicit GRES File ordinals.

    Attest each physical device using /proc and character-device minors, then
    require the CUDA UUID set to equal devices accessible in the step cgroup.
    No fallback from an explicit GRES layout or ordinal inference is allowed.
    """
    if not re.search(r"^ClusterName\s*=\s*delta-gh\s*$", config, re.MULTILINE):
        raise ValueError("Unsupported configless Slurm cluster")
    gres = Path("/run/slurm/conf/gres.conf")
    for path in (gres.resolve(), gres.resolve().parent, Path("/run/slurm/conf/slurm.conf").resolve()):
        st = path.stat()
        if st.st_uid != 0 or st.st_mode & 0o022:
            raise ValueError("Slurm config cache must be root-owned and not writable by users")
    text = gres.read_text()
    lines = [shlex.split(line, comments=True) for line in text.splitlines()]
    lines = [line for line in lines if line]
    if len(lines) != 1:
        raise ValueError("Require the verified single-rule Delta NVML GRES layout")
    rule = dict(word.split("=", 1) for word in lines[0])
    if set(rule) != {"NodeName", "AutoDetect"} or rule["AutoDetect"].lower() != "nvml":
        raise ValueError("Unsupported Delta automatic GRES rule")
    node = os.environ.get("SLURMD_NODENAME", "")
    if node not in command(["scontrol", "show", "hostnames", rule["NodeName"]]).splitlines():
        raise ValueError("Allocated node is outside the automatic GRES rule")
    node_info = command(["scontrol", "show", "node", node])
    if not re.search(r"Gres=gpu:nvidia_gh200_120gb:4(?:\(|\s)", node_info):
        raise ValueError("Require the verified four whole GH200 scheduler layout")
    records = []
    for path in sorted(Path("/proc/driver/nvidia/gpus").glob("*/information")):
        records.append({**parse_proc_gpu_information(path.read_text()), "proc_information_file": str(path)})
    if len(records) != 4:
        raise ValueError("Physical inventory disagrees with four-GH200 scheduler layout")
    records.sort(key=lambda row: row["minor_number"])
    accessible = []
    for row in records:
        path = f"/dev/nvidia{row['minor_number']}"
        st = os.stat(path)
        if not stat.S_ISCHR(st.st_mode) or os.minor(st.st_rdev) != row["minor_number"]:
            raise ValueError("NVIDIA character-device identity mismatch")
        row.update(device_file=path, index=None, slurm_global_index=None)
        try:
            fd = os.open(path, os.O_RDWR | os.O_CLOEXEC)
        except OSError as exc:
            if exc.errno != errno.EPERM:
                raise
        else:
            os.close(fd); accessible.append(row["uuid"])
    return records, {"slurm_config": "/run/slurm/conf/slurm.conf", "gres_config": str(gres),
                     "gres_sha256": hashlib.sha256(text.encode()).hexdigest(),
                     "method": "Delta NVML config + physical /proc device minors + enforced Slurm cgroup device access",
                     "kernel_accessible_uuids": accessible,
                     "cgroup": Path("/proc/self/cgroup").read_text(), "node_info": node_info}


def validate_whole_node_allocation(actual_uuids, physical_uuids, environment, node_info, job_info):
    """All physical GPUs belong to this job; global GPU ordering is irrelevant."""
    node = environment.get("SLURMD_NODENAME", "")
    job = environment.get("SLURM_JOB_ID", "")
    if not node or not job or environment.get("SLURM_STEP_ID") in (None, "batch", "extern"):
        raise ValueError("Whole-node verification requires a Slurm job step")
    fields = dict(re.findall(r"(?:^|\s)(\w+)=([^\s]+)", job_info))
    if (fields.get("JobId") != job or fields.get("JobState") != "RUNNING"
            or fields.get("NumNodes") != "1" or fields.get("NodeList") != node):
        raise ValueError("Scheduler did not confirm this running single-node allocation")
    if not re.search(r"(?:^|\s)Gres=gpu:nvidia_gh200_120gb:4(?:\(|\s|$)", node_info):
        raise ValueError("Scheduler node must have exactly four whole GH200 GPUs")
    tres = dict(item.split("=", 1) for item in fields.get("AllocTRES", "").split(",") if "=" in item)
    if tres.get("gres/gpu") != "4":
        raise ValueError("The job does not own all four node GPUs")
    for key in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS"):
        if environment.get(key) not in ("0,1,2,3", "0-3"):
            raise ValueError("The step must select the entire four-GPU allocation")
    if (len(actual_uuids) != 4 or len(set(actual_uuids)) != 4
            or len(physical_uuids) != 4 or set(actual_uuids) != set(physical_uuids)):
        raise ValueError("CUDA must expose exactly the four physical node UUIDs")


def delta_whole_node_inventory():
    """Confirm full-node ownership directly with Slurm, without GRES files/cgroups."""
    records = [parse_proc_gpu_information(path.read_text())
               for path in sorted(Path("/proc/driver/nvidia/gpus").glob("*/information"))]
    node_info = command(["scontrol", "show", "node", os.environ["SLURMD_NODENAME"]])
    job_info = command(["scontrol", "show", "job", os.environ["SLURM_JOB_ID"], "-o"])
    uuids = [row["uuid"] for row in records]
    validate_whole_node_allocation(uuids, uuids, os.environ, node_info, job_info)
    for row in records:
        row.update(index=None, slurm_global_index=None)
    return records, {"method": "Slurm confirms all four node GPUs assigned to job; CUDA UUID set equals physical inventory",
                     "whole_node_uuids": uuids, "node_info": node_info, "job_info": job_info}


def slurm_gpu_inventory():
    """Attest global IDs through active GRES files and actual character devices."""
    config = command(["scontrol", "show", "config"])
    if (re.search(r"^ClusterName\s*=\s*delta-gh\s*$", config, re.MULTILINE)
            and os.environ.get("SLURM_JOB_GPUS") in ("0,1,2,3", "0-3")
            and os.environ.get("SLURM_STEP_GPUS") in ("0,1,2,3", "0-3")):
        return delta_whole_node_inventory()
    match = re.search(r"^SLURM_CONF\s*=\s*(\S+)\s*$", config, re.MULTILINE)
    if match is None:
        raise ValueError("Cannot locate the active SLURM configuration for GPU identity verification")
    gres_path = Path(match.group(1)).with_name("gres.conf")
    if not gres_path.exists() and Path("/run/slurm/conf/gres.conf").exists():
        return delta_configless_inventory(config)
    gres_text = gres_path.read_text()
    files = expand_gres_gpu_files(gres_text, socket.gethostname())
    records = []
    for path in sorted(Path("/proc/driver/nvidia/gpus").glob("*/information")):
        records.append({**parse_proc_gpu_information(path.read_text()), "proc_information_file": str(path)})
    device_minors = {}
    for path in files:
        metadata = os.stat(path)
        if not stat.S_ISCHR(metadata.st_mode):
            raise ValueError(f"Configured GRES GPU file is not a character device: {path}")
        device_minors[path] = os.minor(metadata.st_rdev)
    inventory = build_global_gpu_inventory(files, records, device_minors)
    return inventory, {"slurm_config": match.group(1), "gres_config": str(gres_path),
                       "gres_sha256": hashlib.sha256(gres_text.encode()).hexdigest(),
                       "method": "GRES configured-file order -> stat(device).minor -> /proc NVIDIA UUID/PCI identity"}


def crosscheck_physical_inventory(nvml_inventory, global_inventory):
    for record in nvml_inventory:
        physical = match_cuda_device_uuid(record["uuid"], global_inventory)
        if canonical_pci_bus_id(record["pci_bus_id"]) != physical["pci_bus_id"]:
            raise ValueError("NVML PCI identity disagrees with NVIDIA /proc physical GPU metadata")


def resolve_slurm_devices(raw, inventory):
    """Resolve global IDs against an attested GRES inventory, never NVML ordinals."""
    resolved = set()
    for token in raw.split(","):
        token = token.strip()
        if re.fullmatch(r"\d+-\d+", token):
            first, last = map(int, token.split("-"))
            candidates = [str(index) for index in range(first, last + 1)]
        else:
            candidates = [token]
        for candidate in candidates:
            matches = [gpu for gpu in inventory if
                       (candidate.isdigit() and gpu["index"] == int(candidate)) or
                       (candidate.startswith("GPU-") and gpu["uuid"].startswith(candidate))]
            if len(matches) != 1:
                raise ValueError(f"Cannot resolve SLURM GPU identifier {candidate!r}")
            resolved.add(matches[0]["uuid"])
    return resolved


def validate_allocation(actual_uuids, inventory, environment, *, allow_step_subset=False):
    actual = set(actual_uuids)
    if len(actual) != len(actual_uuids):
        raise ValueError("CUDA exposed one physical GPU more than once")
    if not any(environment.get(field) for field in ("SLURM_STEP_GPUS", "SLURM_JOB_GPUS")):
        raise ValueError("Missing SLURM GPU assignment; cannot independently verify CUDA visibility")
    for field in ("SLURM_STEP_GPUS", "SLURM_JOB_GPUS"):
        if environment.get(field):
            allowed = resolve_slurm_devices(environment[field], inventory)
            if not actual <= allowed or (field == "SLURM_STEP_GPUS" and not allow_step_subset and actual != allowed):
                raise ValueError(f"CUDA-visible GPU UUIDs do not match {field}")
    count = environment.get("SLURM_GPUS_ON_NODE", "")
    if count.isdigit() and len(actual) > int(count):
        raise ValueError("CUDA sees more GPUs than SLURM_GPUS_ON_NODE")


def check_inventory_for_purpose(inventory, global_inventory, *, purpose, node_names):
    """Node02 permits training on inventoried GPUs, but no timing comparisons."""
    if purpose not in ("measurement", "training"):
        raise ValueError("Unknown GPU probe purpose")
    node02 = any(re.search(r"(?:^|[-_.])node0?2(?:$|[-_.])", name.lower()) for name in node_names)
    if node02 and purpose != "training":
        raise RuntimeError("node02 is excluded from measurements until GPU allocation isolation is verified")
    known = {canonical_full_uuid(record["uuid"]) for record in global_inventory}
    unmanaged = [record for record in inventory if canonical_full_uuid(record["uuid"]) not in known]
    checked = [record for record in inventory if record not in unmanaged] if node02 else inventory
    crosscheck_physical_inventory(checked, global_inventory)
    # Unmanaged devices are recorded, never added to CUDA visibility or the
    # allowed allocation. Every CUDA-visible device still must match GRES below.
    return unmanaged


def validate_direct_request(environment, tokens, expected_hostname, actual_hostname, *, slurm_installed):
    """Standalone use is explicit and cannot silently replace a SLURM allocation."""
    if not expected_hostname or actual_hostname != expected_hostname:
        raise ValueError("Standalone GPU host differs from the explicitly selected hostname")
    if slurm_installed or any(environment.get(key) for key in ("SLURM_JOB_ID", "SLURM_STEP_ID")):
        raise ValueError("Use the existing allocated-step probe on a SLURM host")
    for token in tokens:
        if not token.startswith("GPU-"):
            raise ValueError("Standalone GPU selection requires complete physical UUIDs, not indices")
        canonical_full_uuid(token)


def run_probe(expected_count, max_idle_memory_mib=256, *, purpose="measurement", direct_hostname=None,
              allow_step_subset=False):
    if sys.platform != "linux":
        raise RuntimeError("GPU probe requires Linux")
    if allow_step_subset and (purpose != "training" or direct_hostname is not None):
        raise ValueError("Step subsets require a SLURM training/evaluation probe, not a timing measurement")
    environment = {key: os.environ.get(key) for key in (
        "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "SLURM_JOB_ID", "SLURM_STEP_ID",
        "SLURM_STEP_GPUS", "SLURM_JOB_GPUS", "SLURM_GPUS_ON_NODE", "SLURMD_NODENAME",
        "SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS",
    )}
    names = (socket.gethostname(), environment.get("SLURMD_NODENAME") or "")
    if direct_hostname is None and (not environment["SLURM_JOB_ID"] or environment["SLURM_STEP_ID"] in (None, "batch", "extern")):
        raise RuntimeError("Run inside an allocated srun step, not an SSH or sbatch shell")
    tokens = visible_tokens(os.environ)
    if len(tokens) != expected_count:
        raise ValueError("Visibility count differs from requested probe count")
    inventory = gpu_inventory()
    if direct_hostname is not None:
        validate_direct_request(environment, tokens, direct_hostname, socket.gethostname(),
                                slurm_installed=shutil.which("srun") is not None)
        if purpose not in ("measurement", "training"):
            raise ValueError("Unknown GPU probe purpose")
        global_inventory = inventory
        unmanaged = []
        mapping_evidence = {"method": "explicit standalone host and complete GPU UUIDs",
                            "expected_hostname": direct_hostname, "requested_uuids": tokens,
                            "scheduler_isolation": False}
    else:
        global_inventory, mapping_evidence = slurm_gpu_inventory()
        unmanaged = check_inventory_for_purpose(inventory, global_inventory, purpose=purpose, node_names=names)
    before = compute_processes()
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != len(tokens):
        raise RuntimeError("CUDA initialization/device count disagrees with the allocation")
    devices = []
    for ordinal, token in enumerate(tokens):
        props = torch.cuda.get_device_properties(ordinal)
        uuid = str(props.uuid)
        physical = match_cuda_device_uuid(uuid, inventory)
        if props.name != physical["name"]:
            raise ValueError("CUDA and nvidia-smi GPU names disagree")
        devices.append({"allocation_ordinal": ordinal, "visibility_token": token,
                        "uuid": physical["uuid"], "cuda_name": props.name,
                        "cuda_total_bytes": props.total_memory, "inventory": physical,
                        "global_gres_device": (None if direct_hostname is not None else match_cuda_device_uuid(uuid, global_inventory))})
    actual_uuids = [gpu["uuid"] for gpu in devices]
    if direct_hostname is not None:
        if list(map(canonical_full_uuid, actual_uuids)) != list(map(canonical_full_uuid, tokens)):
            raise ValueError("CUDA-visible devices differ from the explicitly selected physical GPUs")
    elif "whole_node_uuids" in mapping_evidence:
        validate_whole_node_allocation(actual_uuids, mapping_evidence["whole_node_uuids"],
                                       os.environ, mapping_evidence["node_info"], mapping_evidence["job_info"])
    elif "kernel_accessible_uuids" in mapping_evidence:
        if allow_step_subset:
            raise ValueError("Configless kernel attestation requires the entire allocated step")
        validate_kernel_allocation(actual_uuids, mapping_evidence["kernel_accessible_uuids"],
                                   os.environ, mapping_evidence["cgroup"])
    else:
        validate_allocation(actual_uuids, global_inventory, os.environ, allow_step_subset=allow_step_subset)
    co_tenants = [row for row in before if row["gpu_uuid"] in actual_uuids and row["pid"] != os.getpid()]
    if co_tenants:
        raise RuntimeError(f"Allocated GPUs already have compute processes: {co_tenants}")
    for device in devices:
        physical = device["inventory"]
        if physical["memory_used_mib"] > max_idle_memory_mib or physical["utilization_percent"] > 5:
            raise RuntimeError(f"Allocated GPU is not idle or has a leaked context: {physical}")
        with torch.cuda.device(device["allocation_ordinal"]):
            value = torch.ones((1024, 1024), device="cuda", dtype=torch.float32)
            observed = (value * 2).sum().item()
            torch.cuda.synchronize()
            if observed != 2097152:
                raise RuntimeError("Real CUDA allocation/computation failed")
            device["allocation_check"] = {"elements": value.numel(), "result": observed}
            del value
            torch.cuda.empty_cache()
    after = compute_processes()
    if any(row["gpu_uuid"] in actual_uuids and row["pid"] != os.getpid() for row in after):
        raise RuntimeError("Another compute process appeared on an allocated GPU during the probe")
    return {"status": "passed", "purpose": purpose, "unmanaged_nvml_devices": unmanaged,
            "allow_step_subset": allow_step_subset,
            "timing_comparison_eligible": purpose == "measurement",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(), "probe_pid": os.getpid(), "environment": environment,
            "allocation_type": "standalone_explicit_uuid" if direct_hostname is not None else "slurm_step",
            "global_gres_inventory": None if direct_hostname is not None else global_inventory,
            "gpu_identity_mapping": mapping_evidence,
            "devices": devices, "compute_processes_before": before,
            "compute_processes_after": after, "topology": nvidia_query(["nvidia-smi", "topo", "-m"]),
            "note": "Probe process exits before model processes start; CUDA contexts then release."}


def run_baige_probe(expected_count, max_idle_memory_mib, purpose):
    from fluffy_bassoon.baige_gpu import validate_container, cuda_pci, map_devices
    if sys.platform != 'linux' or purpose not in ('training', 'evaluation'):
        raise ValueError('Baige container probe requires allocated Linux training/evaluation')
    assigned = validate_container(os.environ, socket.gethostname())
    tokens = visible_tokens(os.environ)
    if len(tokens) != expected_count or len(assigned) != expected_count:
        raise ValueError('Baige visibility differs from the requested GPU count')
    inventory = gpu_inventory()
    before = compute_processes()
    if before:
        raise RuntimeError(f'Baige allocation already has compute processes: {before}')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != expected_count:
        raise RuntimeError('Baige CUDA initialization/device count disagrees with allocation')
    cuda = []
    for ordinal, token in enumerate(tokens):
        props = torch.cuda.get_device_properties(ordinal)
        cuda.append(dict(allocation_ordinal=ordinal, visibility_token=token,
            cuda_uuid=str(props.uuid), cuda_name=props.name, cuda_total_bytes=props.total_memory,
            pci_bus_id=cuda_pci(ordinal)))
    devices = map_devices(cuda, inventory, assigned)
    for row in devices:
        gpu = row['inventory']
        # Baige's scheduler allocation is exclusive. Evaluation admission uses
        # real CUDA/PID/PCI evidence below rather than an idle utilization sample,
        # which can report100% with no memory or compute process on this runtime.
        if gpu['memory_used_mib'] > max_idle_memory_mib or (purpose == 'training' and gpu['utilization_percent'] > 5):
            raise RuntimeError(f'Baige GPU is not idle: {gpu}')
        with torch.cuda.device(row['allocation_ordinal']):
            value = torch.ones((1024, 1024), device='cuda')
            observed = (value * 2).sum().item()
            torch.cuda.synchronize()
            if observed != 2097152:
                raise RuntimeError('Real Baige CUDA computation failed')
            processes = compute_processes()
            if any(x['pid'] != os.getpid() for x in processes):
                raise RuntimeError('Foreign compute process appeared during Baige probe')
            own = {x['gpu_uuid'] for x in processes if x['pid'] == os.getpid()}
            expected = {x['uuid'] for x in devices[:row['allocation_ordinal'] + 1]}
            if own != expected:
                raise RuntimeError('CUDA allocation PID accounting disagrees with the PCI mapping')
            row['allocation_check'] = dict(elements=value.numel(), result=observed, own_pid_gpu_uuids=sorted(own))
            del value
            torch.cuda.empty_cache()
    return dict(status='passed', purpose=purpose, timing_comparison_eligible=False,
        allocation_type='baige_container', hostname=socket.gethostname(), probe_pid=os.getpid(),
        created_at_utc=datetime.now(timezone.utc).isoformat(), devices=devices,
        idle_utilization_required=purpose != 'evaluation',
        gpu_identity_mapping=dict(method='scheduler NVIDIA UUID set + CUDA/NVML PCI bijection + own-PID allocation',
            assigned_nvml_uuids=assigned), compute_processes_before=before,
        compute_processes_after=compute_processes(), topology=nvidia_query(['nvidia-smi', 'topo', '-m']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--max-idle-memory-mib", type=float, default=256)
    parser.add_argument("--purpose", choices=("measurement", "training", "evaluation"), default="measurement")
    parser.add_argument("--direct-hostname", help="Explicit standalone host only; requires full GPU UUID visibility and no SLURM")
    parser.add_argument('--baige-container', action='store_true', help='Attest scheduled Baige container by PCI identity')
    parser.add_argument("--allow-step-subset", action="store_true",
                        help="Probe only the selected GPUs within the recorded SLURM step (training/evaluation only)")
    args = parser.parse_args()
    if args.purpose == 'evaluation' and not args.baige_container:
        parser.error('Evaluation admission is specific to an exclusive Baige scheduler container')
    if args.baige_container and (args.direct_hostname or args.allow_step_subset):
        parser.error("Baige container attestation cannot replace direct/SLURM attestation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error("Use a new probe output path")
    try:
        result = (run_baige_probe(args.expected_count, args.max_idle_memory_mib, args.purpose)
                  if args.baige_container else run_probe(args.expected_count, args.max_idle_memory_mib,
                  purpose=args.purpose, direct_hostname=args.direct_hostname, allow_step_subset=args.allow_step_subset))
    except Exception as exc:
        result = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc),
                  "hostname": socket.gethostname(), "probe_pid": os.getpid(),
                  "environment": {key: os.environ.get(key) for key in
                                  ("CUDA_VISIBLE_DEVICES", "SLURM_JOB_ID", "SLURM_STEP_ID",
                                   "SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "SLURM_GPUS_ON_NODE")}}
        try:
            result["nvml_inventory"] = gpu_inventory()
        except Exception:
            pass
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        raise
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
