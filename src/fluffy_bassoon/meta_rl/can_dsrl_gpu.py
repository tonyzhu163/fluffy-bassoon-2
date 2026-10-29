"""Check a one-GPU worker against a fresh, complete Baige allocation probe.

The scheduler's NVIDIA visibility stays unchanged. A worker selects one CUDA
ordinal from a parent probe that ran before any model process was started.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys


def _source_commit(expected):
    if re.fullmatch(r"[0-9a-f]{40}", expected) is None:
        raise ValueError("Require an exact source commit")
    root = Path(__file__).resolve().parents[3]
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    if actual != expected or dirty:
        raise ValueError("GPU attestation requires the clean pinned worker source")
    return root


def _parent_report(path, source_commit, assigned, hostname):
    payload = path.read_bytes()
    report = json.loads(payload)
    if (report.get("status") != "passed" or report.get("purpose") != "training"
            or report.get("allocation_type") != "baige_container"
            or report.get("hostname") != hostname):
        raise ValueError("Require a passed parent training probe from this Baige allocation")
    devices = report.get("devices", [])
    if len(devices) not in (1, 8) or len(devices) != len(assigned):
        raise ValueError("Can workers require a complete one- or eight-GPU parent probe")
    if ({row["uuid"] for row in devices} != set(assigned)
            or set(report["gpu_identity_mapping"]["assigned_nvml_uuids"]) != set(assigned)
            or [row["allocation_ordinal"] for row in devices] != list(range(len(devices)))
            or [row["visibility_token"] for row in devices] != list(map(str, range(len(devices))))):
        raise ValueError("Parent GPU probe differs from the complete scheduler UUID allocation")
    recorded_source = report.get("source_commit")
    # The initial one-GPU qualification exercises this API on the generic
    # probe output. The eight-GPU launcher always binds its probe to Git first.
    if recorded_source != source_commit and not (len(devices) == 1 and recorded_source is None):
        raise ValueError("Parent GPU probe source differs from the worker source")
    created = datetime.fromisoformat(report["created_at_utc"])
    if created.tzinfo is None:
        raise ValueError("Parent probe timestamp must include its UTC offset")
    age = (datetime.now(timezone.utc) - created).total_seconds()
    if not -30 <= age <= 3600:
        raise ValueError("Parent GPU probe is stale or from the future")
    if report.get("compute_processes_before") != []:
        raise ValueError("Parent probe did not establish an empty allocation before launch")
    if any(not row.get("allocation_check") for row in devices):
        raise ValueError("Parent probe lacks real CUDA/PID allocation evidence")
    return report, hashlib.sha256(payload).hexdigest()


def _own_processes(selected_uuid):
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return dict(available=False, reason="nvidia-smi is unavailable in this container")
    result = subprocess.run(
        [executable, "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True, timeout=30,
    )
    rows = []
    for fields in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if not fields or not fields[0].strip():
            continue
        if len(fields) != 4:
            raise ValueError("Cannot parse worker GPU process accounting")
        uuid, pid, name, memory = (field.strip() for field in fields)
        rows.append(dict(gpu_uuid=uuid, pid=int(pid), process_name=name, memory_used_mib=memory))
    own = {row["gpu_uuid"] for row in rows if row["pid"] == os.getpid()}
    if own != {selected_uuid}:
        raise RuntimeError("Worker PID is not confined to its attested physical GPU")
    if any(row["gpu_uuid"] == selected_uuid and row["pid"] != os.getpid() for row in rows):
        raise RuntimeError("Another compute process occupies this worker's selected GPU")
    return dict(available=True, own_pid=os.getpid(), own_gpu_uuids=sorted(own),
                selected_gpu_processes=[row for row in rows if row["gpu_uuid"] == selected_uuid])


def validate_prequalified_worker(path, source_commit, output):
    """Write a distinct worker probe after checking a full parent allocation.

    Call inside the actual worker, before motor construction. ``output`` must
    be fresh and distinct from ``path``. The initial qualification may pass a
    generic one-GPU probe; eight-GPU probes must record ``source_commit``.
    """
    if sys.platform != "linux":
        raise RuntimeError("Worker CUDA qualification requires allocated Linux")
    from fluffy_bassoon.baige_gpu import mapping, selected_device, validate_container, verify_selected_cuda

    path, output = Path(path).resolve(), Path(output).resolve()
    if path == output or output.exists():
        raise ValueError("Worker GPU evidence must use a fresh distinct output path")
    _source_commit(source_commit)
    hostname = socket.gethostname()
    assigned = validate_container(os.environ, hostname)
    report, parent_sha256 = _parent_report(path, source_commit, assigned, hostname)
    selected = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if re.fullmatch(r"[0-7]", selected) is None or int(selected) >= len(assigned):
        raise ValueError("Select exactly one parent CUDA ordinal for this worker")
    if os.environ.get("LYCO_BAIGE_ISOLATED_UUID"):
        raise ValueError("Can workers select their parent ordinal directly")
    os.environ["LYCO_BAIGE_GPU_MAP"] = str(path)
    if len(mapping()) != len(assigned):
        raise RuntimeError("Worker lost the complete parent GPU mapping")
    row = selected_device()
    import torch

    selected_uuid = verify_selected_cuda(torch)
    value = torch.ones((64, 64), device="cuda")
    observed = float((value * 2).sum().item())
    torch.cuda.synchronize()
    if observed != 8192.:
        raise RuntimeError("Worker's real CUDA tensor check failed")
    processes = _own_processes(selected_uuid)
    verify_selected_cuda(torch)
    evidence = dict(status="passed", purpose="training", allocation_type="baige_prequalified_worker",
                    hostname=hostname, source_commit=source_commit, parent_probe=str(path),
                    parent_probe_sha256=parent_sha256, parent_gpu_count=len(assigned),
                    source_binding="parent_and_worker" if report.get("source_commit") else "fresh_clean_one_gpu_worker",
                    scheduler_nvidia_visible_devices=assigned, cuda_visible_devices=selected,
                    selected_device=row, elements=value.numel(), tensor_result=observed,
                    pid_allocation=processes, created_at_utc=datetime.now(timezone.utc).isoformat(),
                    timing_comparison_eligible=False)
    with output.open("x") as stream:
        stream.write(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    del value
    torch.cuda.empty_cache()
    return evidence
