"""Read-only CPU-slot check, invoked before analysis on the shared development host."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

slots = {46, 47}


def counters():
    with open("/proc/stat") as stream:
        return {s[0]: list(map(int, s[1:9])) for line in stream if (s := line.split())[0] in {"cpu46", "cpu47"}}


before = counters()
time.sleep(1)
after = counters()
bound = []
for tid in Path("/proc").glob("[0-9]*/task/[0-9]*"):
    try:
        mask = os.sched_getaffinity(int(tid.name))
        if mask & slots and len(mask) <= 4:
            bound.append([int(tid.name), (tid / "comm").read_text().strip(), sorted(mask)])
    except (PermissionError, FileNotFoundError, ProcessLookupError):
        continue
delta = {k: [b-a for a,b in zip(before[k],after[k])] for k in sorted(before)}
print(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), host=os.uname().nodename,
                      narrow_affinities=bound, cpu_stat_deltas=delta)))
