"""Run on the Baige CPU reader, from the run root. Read-only, JSON to stdout."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import socket
import subprocess
import time

root = Path.cwd()
store = root / "reference-native"
assert store.is_dir()
cutoff = time.time()


def utc(t):
    return datetime.fromtimestamp(t, timezone.utc).isoformat()


def rel(path):
    return str(path.relative_to(root))


def evidence(path):
    data = path.read_bytes()
    return data, dict(path=rel(path), read_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def complete_lines(data):
    return [line.decode().rstrip("\r\n") for line in data.splitlines(keepends=True) if line.endswith(b"\n")]


out = dict(root=str(root), reader_host=socket.gethostname(), cutoff_epoch=cutoff,
           receipt_cutoff_utc=utc(cutoff), attempts={}, streams={}, logs={})
out["lock_filesystem"] = subprocess.check_output(
    ["findmnt", "-n", "-o", "FSTYPE,OPTIONS", "-T", str(store/"locks")], text=True).strip()
keys = defaultdict(list)
for directory in sorted((store / "attempts").iterdir()):
    if not directory.is_dir():
        continue
    reserve = directory / "reservation.json"
    if not reserve.is_file():
        out["attempts"][directory.name] = dict(reservation=None)
        continue
    data, source = evidence(reserve)
    reservation = json.loads(data)
    info = dict(reservation=reservation, source=source)
    out["attempts"][directory.name] = info
    if reservation["phase"] not in ("all", "collect", "babble"):
        continue
    gpu = directory / "gpu.json"
    if gpu.is_file():
        blob, fp = evidence(gpu)
        probe = json.loads(blob)
        info["probe"] = dict(**fp, hostname=probe["hostname"], status=probe["status"])
    if (directory / "exit.json").is_file():
        info["exit"] = json.loads((directory / "exit.json").read_bytes())
    for path in sorted(directory.glob("*.jsonl")):
        if not path.stem.startswith(("babble-", "collect-")):
            continue
        blob, fp = evidence(path)
        items = []
        for i, line in enumerate(complete_lines(blob), 1):
            row = json.loads(line)
            if row["utc"] <= cutoff:
                items.append(dict(line=i, **row))
                keys[row["unit"]].append(dict(attempt=directory.name, lane=path.stem,
                                              host=reservation["host"], line=i, **row))
        out["streams"][rel(path)] = dict(**fp, rows=items)
    for path in sorted(directory.glob("*.log")):
        blob, fp = evidence(path)
        lines = []
        for i, line in enumerate(complete_lines(blob), 1):
            if (line.startswith(("[batch-checkpoint]", "[replay]", "[collect]", "{'event':")) or
                re.search(r"Traceback|\.tmp\b|FileNotFoundError|(?:Error|Exception):|failed writing|unexpected pos", line)):
                lines.append([i, line])
        out["logs"][rel(path)] = dict(**fp, lines=lines)

duplicates = {}
for key, rows in sorted(keys.items()):
    if len({r["host"] for r in rows}) < 2:
        continue
    pub = [r for r in rows if r["event"] == "published"]
    duplicates[key] = dict(rows=rows, comparable=len({r["host"] for r in pub}) >= 2,
                           counts=sorted({(r["transitions"], r["episodes"]) for r in pub}))
out["cross_host_keys"] = duplicates
summary = {}
for phase in ("collect", "babble"):
    phase_keys = {k: v for k, v in keys.items() if k.split("/")[3] == phase}
    repeated = [v for k,v in duplicates.items() if k in phase_keys]
    published = [v for v in repeated if v["comparable"]]
    summary[phase] = dict(events=dict(Counter(r["event"] for rows in phase_keys.values() for r in rows)),
                          unique_keys_started=len({k for k,rs in phase_keys.items() if any(r["event"]=="start" for r in rs)}),
                          unique_keys_published=len({k for k,rs in phase_keys.items() if any(r["event"]=="published" for r in rs)}),
                          cross_host_started_or_published=len(repeated), cross_host_published=len(published),
                          counts_differ=sum(len(v["counts"]) > 1 for v in published))
out["summary"] = summary
out["read_finished_utc"] = utc(time.time())
print(json.dumps(out, indent=1, allow_nan=False))
