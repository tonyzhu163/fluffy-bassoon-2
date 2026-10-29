"""Read-only worker inspection; execute only in this run's scheduler container."""
from pathlib import Path
import json
import socket
import subprocess
import time

assert socket.gethostname() == "job-orgyjb0z9h6d-master-0"
root = Path("/mowenjie/fluffy-bassoon-2/outputs/metaworld-pi05-noise-baige-20261029-r2/preparation")
report = {"hostname": socket.gethostname(), "checked_unix": time.time(), "processes": []}
for path in Path("/proc").iterdir():
    if not path.name.isdigit():
        continue
    try:
        args = (path / "cmdline").read_bytes().split(b"\0")
        if b"scripts/diagnostics/metaworld_pi05_noise_cache.py" not in args:
            continue
        report["processes"].append({
            "pid": int(path.name), "executable": args[0].decode(),
            "status": (path / "status").read_text(),
            "wchan": (path / "wchan").read_text(), "io": (path / "io").read_text(),
        })
    except OSError:
        pass
report["gpu"] = subprocess.run([
    "nvidia-smi", "--query-gpu=uuid,pci.bus_id,memory.used,utilization.gpu,utilization.memory",
    "--format=csv",
], capture_output=True, text=True).stdout
(root / "worker-diagnostic.json").write_text(json.dumps(report, indent=2) + "\n")
print("WORKER_DIAGNOSTIC_SAVED", [(p["pid"], p["wchan"]) for p in report["processes"]], flush=True)
