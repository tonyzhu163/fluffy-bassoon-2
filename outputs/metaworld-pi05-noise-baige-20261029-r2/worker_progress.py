"""Read-only progress snapshot; retain only loop counters from py-spy locals."""
from pathlib import Path
import datetime
import json
import socket
import subprocess

assert socket.gethostname() == "job-orgyjb0z9h6d-master-0"
assert Path("/proc/1415/cwd").resolve() == Path("/mowenjie/fluffy-bassoon-2/worktrees/metaworld-pi05-noise-baige-20261029-r2")
assert b"scripts/diagnostics/metaworld_pi05_noise_cache.py" in Path("/proc/1415/cmdline").read_bytes().split(b"\0")
data = json.loads(subprocess.check_output([
    "/mowenjie/bin/uvx", "py-spy", "dump", "--pid", "1415", "--locals", "--json",
], text=True, timeout=20))
allowed = {"g", "scene", "arm", "repeat", "step", "rows", "row", "num_steps", "timed_out"}
frames = []


def visit(value):
    if isinstance(value, dict):
        name = value.get("name", value.get("function", ""))
        if name in {"main", "rollout", "sample_actions"}:
            local = value.get("locals", {})
            counters = {}
            if isinstance(local, dict):
                counters = {k: str(v)[:500] for k, v in local.items() if k in allowed}
            elif isinstance(local, list):
                counters = {v.get("name"): str(v)[:500] for v in local
                            if isinstance(v, dict) and v.get("name") in allowed}
            frames.append({"function": name, "keys": list(value), "counters": counters})
        for child in value.values():
            visit(child)
    elif isinstance(value, list):
        for child in value:
            visit(child)


visit(data)
report = {"checked_at": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
          "frames": frames}
root = Path("/mowenjie/fluffy-bassoon-2/outputs/metaworld-pi05-noise-baige-20261029-r2/preparation")
name = "worker-progress-" + report["checked_at"][11:19].replace(":", "") + ".json"
(root / name).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report), flush=True)
