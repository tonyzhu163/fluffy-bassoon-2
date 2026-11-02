"""Run-specific completion-lifecycle qualification on allocated Linux CPUs."""
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import threading
import time

assert sys.platform == 'linux'
os.environ['TASK_READOUT_METHODS'] = '["dpt_full"]'
driver = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'opendrawer-ad-dpt-k1-id-ood-20261102-r1/READOUT.py'))
wait = driver['wait_for_nodes']
checks = []
with tempfile.TemporaryDirectory() as name:
    root = Path(name)
    def receipt(node, codes):
        p = root / f'node-{node:02d}' / 'workers-terminal.json'
        p.parent.mkdir(exist_ok=True)
        temporary = p.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(exit_codes=codes)))
        temporary.replace(p)
    receipt(0, [0, 0])
    for node in (1, 2):
        receipt(node, [0, 0])
    try:
        wait(root, time.time() - 1)
    except RuntimeError:
        checks.append('master cannot exit while one node receipt is missing')
    else:
        raise AssertionError('Missing worker was admitted')
    thread = threading.Thread(target=lambda: (time.sleep(.1), receipt(3, [0, 0])))
    thread.start()
    wait(root, time.time() + 10)
    thread.join()
    checks.append('master waits for the slower final node and admits four successes')
    receipt(2, [0, 1])
    try:
        wait(root, time.time() + 10)
    except RuntimeError:
        checks.append('failed node cannot become scheduler-success evidence')
    else:
        raise AssertionError('Failed worker was admitted')
print(json.dumps(dict(passed=True, checks=checks, gpu_computation=False), indent=2))
