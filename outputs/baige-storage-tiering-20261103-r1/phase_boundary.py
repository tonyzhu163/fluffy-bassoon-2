"""Stop only this audit's first driver after its six disjoint cold moves."""
import json
import os
from pathlib import Path
import signal
import time

BASE = Path('/mowenjie/fluffy-bassoon-2/outputs/baige-storage-tiering-20261103-r1')
PID = 1943091
expected = json.loads((BASE / 'PLAN.json').read_text())['sources'][:6]
while True:
    try:
        data = json.loads((BASE / 'SUMMARY.json').read_text())
    except json.JSONDecodeError:
        time.sleep(2)
        continue
    completed = {x['source'] for x in data['completed']}
    if all('/mowenjie/' + rel in completed for rel in expected):
        cmdline = Path(f'/proc/{PID}/cmdline').read_bytes().split(b'\0')
        assert str(BASE / 'archive.py').encode() in cmdline
        os.kill(PID, signal.SIGSTOP)
        # Stop before any later first-phase source can be reclaimed. A later
        # source may already have been cold-moved by the disjoint r2 owner.
        later = json.loads((BASE / 'PLAN.json').read_text())['sources'][6:]
        assert not any((BASE/(x.replace('/', '__')+'.json')).exists() for x in later)
        assert not any(Path('/mowenjie', x+'.tiering-baige-storage-tiering-20261103-r1').exists() for x in later)
        parents = {}
        for p in Path('/proc').iterdir():
            try:
                if p.name.isdigit():
                    parents[int(p.name)] = int((p/'stat').read_text().split()[3])
            except (OSError, ValueError):
                pass
        children = {PID}
        while True:
            updated = children | {child for child,parent in parents.items() if parent in children}
            if updated == children:
                break
            children = updated
        children.remove(PID)
        for child in children:
            try:
                os.kill(child, signal.SIGTERM)
            except ProcessLookupError:
                pass
        os.kill(PID, signal.SIGTERM)
        os.kill(PID, signal.SIGCONT)
        data['phase_boundary'] = 'First six sources complete; parent stopped before reclaim of any later source. Remaining disjoint sources owned by r2.'
        data['finished_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        (BASE/'PHASE-BOUNDARY.json').write_text(json.dumps(data, indent=2)+'\n')
        print(data['phase_boundary'], flush=True)
        break
    time.sleep(2)
