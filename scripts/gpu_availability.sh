#!/usr/bin/env bash
# Report scheduler capacity and read-only GPU telemetry on both school clusters.
# Usage: scripts/gpu_availability.sh [ssh-alias ...]   (default: thu1-master thu2-node01)
# No allocations or CUDA work: telemetry cannot certify allocation isolation.
exec python3 - "$@" <<'PY'
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import csv
from datetime import datetime, timezone
from pathlib import Path
import re
import shlex
import subprocess
import sys
from tempfile import TemporaryDirectory


def query(host, command):
    try:
        result = subprocess.run(
            ["ssh", "-F", str(ssh_config), "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, command],
            capture_output=True, text=True, timeout=25,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return "", str(exc)
    error = (result.stderr or result.stdout or f"SSH exit {result.returncode}").strip()
    return result.stdout, error if result.returncode else ""


def number(fields, key, default=None):
    value = fields.get(key, default)
    return int(value) if value is not None and str(value).isdigit() else None


def gpu_count(value):
    if value in (None, "N/A"):
        return None
    if value in ("", "(null)"):
        return 0
    # Strip topology/IDX lists before splitting mixed typed GRES entries.
    entries = re.sub(r"\([^)]*\)", "", value).split(",")
    counts = [entry.rsplit(":", 1)[-1] for entry in entries if entry.startswith("gpu:")]
    return sum(map(int, counts)) if counts and all(count.isdigit() for count in counts) else None


def remaining(total, allocated):
    return None if total is None or allocated is None else total - allocated


def show(value, gib=False):
    return "?" if value is None or value < 0 else f"{value / 1024:.1f}" if gib else str(value)


def telemetry(host, node):
    name = node["NodeName"]
    # Reuse local secondary aliases; other clusters route through their entry host.
    secondary = host == "thu2-node01"
    target = "thu2-" + name if secondary else host
    command = shlex.join(["sh", "-c",
        "nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,temperature.gpu,ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits 2>&1"
        " || printf 'GPU query failed (exit %s)\\n' \"$?\"; "
        "printf '\\n__GPU_PROCESSES__\\n'; "
        "nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>&1"
        " || printf 'Process query failed (exit %s)\\n' \"$?\"",
    ])
    if not secondary:
        command = shlex.join(
            ["timeout", "20", "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", name, command]
        )
    raw, error = query(target, command)
    raw, separator, process_raw = raw.partition("\n__GPU_PROCESSES__\n")
    errors = [error] if error else []
    if not separator and not error:
        errors.append("incomplete telemetry response")
    gpus = []
    for row in csv.reader(raw.splitlines(), skipinitialspace=True):
        if len(row) == 8 and row[0].isdigit() and row[1].startswith("GPU-"):
            gpus.append([field.strip() for field in row])
        elif row:
            errors.append(", ".join(row))
    processes = {}
    processes_known = bool(separator) and not error
    for row in csv.reader(process_raw.splitlines(), skipinitialspace=True):
        if len(row) == 2 and row[0].startswith("GPU-") and row[1].strip().isdigit():
            processes.setdefault(row[0].strip(), []).append(row[1].strip())
        elif row:
            processes_known = False
            errors.append(", ".join(row))
    configured = gpu_count(node.get("Gres"))
    detected = len(gpus) if gpus else None
    mismatch = configured is not None and detected is not None and detected != configured
    status = "partial/error" if errors else "readable" if gpus else "unknown"
    lines = [f"  {host}/{name}: telemetry={status}, detected={show(detected)}/configured={show(configured)}"
             + (" INVENTORY MISMATCH" if mismatch else "")
             + f"; GRES={node.get('Gres', '?')}; isolation=unverified"]
    for index, uuid, model, total, used, utilization, temperature, ecc in gpus:
        pids = ",".join(processes.get(uuid, [])) or ("none" if processes_known else "unknown")
        lines.append(f"    GPU {index} {uuid} {model}: VRAM={used}/{total} MiB, util={utilization}%, "
                     f"temp={temperature} C, uncorrected_ECC={ecc}, compute_PIDs={pids}"
                     + (" CHECK ECC" if ecc.isdigit() and int(ecc) > 0 else ""))
    lines.extend("    query issue: " + issue.replace("\n", " | ") for issue in dict.fromkeys(errors))
    return "\n".join(lines)


hosts = sys.argv[1:] or ["thu1-master", "thu2-node01"]
checked_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
print("Unallocated resources are scheduler counts, not guaranteed job fit; check state, partition limits and topology.")
print("RAM unalloc = configured - reserved - allocated; OS-free is informational. GPU telemetry is not a CUDA health test.")
print("Isolation remains unverified: use scripts/[train]gpu_probe.py inside the actual job step before GPU work.", flush=True)
with TemporaryDirectory(prefix="gpu-ssh-", dir="/tmp") as control_dir, ThreadPoolExecutor(max_workers=8) as pool:
    # A short private socket path also covers ProxyJump's child ssh via -F.
    ssh_config = Path(control_dir) / "config"
    ssh_config.write_text(
        f"Host *\n  ControlMaster auto\n  ControlPersist 30\n  ControlPath {control_dir}/%C\n"
        "Include ~/.ssh/config\nHost *\nInclude /etc/ssh/ssh_config\n"
    )
    pending = {pool.submit(query, host, "scontrol -d -o show nodes"): (host, False) for host in hosts}
    while pending:
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            host, is_telemetry = pending.pop(future)
            if is_telemetry:
                print(future.result(), flush=True)
                continue
            raw, error = future.result()
            print(f"\n== {host} (scheduler snapshot started {checked_at})", flush=True)
            if error:
                print("  scheduler unknown/unreachable: " + error, flush=True)
                continue
            nodes = []
            for line in raw.splitlines():
                fields = dict(re.findall(r"\b(\w+)=(\S*)", line))
                if fields.get("NodeName") and re.search(r"(?:^|,)gpu:", fields.get("Gres", "")):
                    nodes.append(fields)
            if not nodes:
                print("  scheduler unknown: no GPU node records returned", flush=True)
                continue
            print(f"{'node':<10} {'partition':<18} {'state':<18} {'GPU total/alloc/unalloc':<24} {'CPU unalloc/total':<19} {'RAM unalloc GiB':>16} {'OS-free GiB':>12}")
            for node in nodes:
                total, allocated = gpu_count(node.get("Gres")), gpu_count(node.get("GresUsed"))
                cpu = number(node, "CPUEfctv", node.get("CPUTot"))
                memory = remaining(number(node, "RealMemory"), number(node, "MemSpecLimit", "0"))
                gpu_text = "/".join(show(n) for n in (total, allocated, remaining(total, allocated)))
                cpu_text = f"{show(remaining(cpu, number(node, 'CPUAlloc')))}/{show(cpu)}"
                print(f"{node['NodeName']:<10} {node.get('Partitions', '?'):<18} {node.get('State', '?'):<18} "
                      f"{gpu_text:<24} {cpu_text:<19} {show(remaining(memory, number(node, 'AllocMem')), True):>16} "
                      f"{show(number(node, 'FreeMem'), True):>12}", flush=True)
                pending[pool.submit(telemetry, host, node)] = (host, True)
PY
