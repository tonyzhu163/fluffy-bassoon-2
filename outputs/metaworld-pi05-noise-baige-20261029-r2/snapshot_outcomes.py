"""Copy complete readable outcome lines to a closed monitoring artifact."""
from pathlib import Path
import datetime
import json
import os
import socket

assert socket.gethostname() == "job-orgyjb0z9h6d-master-0"
root = Path("/mowenjie/fluffy-bassoon-2/outputs/metaworld-pi05-noise-baige-20261029-r2")
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
report = {"checked_at": now.isoformat(), "conditions": {}, "open_output_positions": {}}
for path in root.glob("s0/*/outcomes.jsonl"):
    records = []
    for line in path.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    report["conditions"][path.parent.name] = {"complete_rows": len(records), "bytes": path.stat().st_size}
    if records:
        snapshot = root / "preparation" / (path.parent.name + "-snapshot-" + now.strftime("%H%M%S") + ".jsonl")
        snapshot.write_text("".join(json.dumps(row) + "\n" for row in records))
        report["conditions"][path.parent.name]["snapshot"] = str(snapshot)
for fd in Path("/proc/1415/fd").iterdir():
    try:
        name = os.readlink(fd)
        if name.endswith("outcomes.jsonl"):
            report["open_output_positions"][name] = (Path("/proc/1415/fdinfo") / fd.name).read_text()
    except OSError:
        pass
(root / "preparation" / "outcome-snapshot-receipt.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report), flush=True)
