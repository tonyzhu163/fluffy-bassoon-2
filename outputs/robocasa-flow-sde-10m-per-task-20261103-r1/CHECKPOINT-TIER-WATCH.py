"""Run the unchanged qualified checkpoint tierer at new complete boundaries.

The parent sleeps; at most one two-thread tier subprocess runs. Together with
the10-thread numerical watch and1-thread serial closer this fits the13CPU quota.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT = Path('/mowenjie/fluffy-bassoon-2/outputs/robocasa-flow-sde-10m-per-task-20261103-r1')
BOS = '/mnt/bos/adgfyl/strong-plant-archives/robocasa-flow-sde-10m-20261103-r1'
TOOL = ROOT / 'tools/475e1c5d-TIER-CHECKPOINTS.py'
TOOL_SHA = '88a12eaf79e18bfd527f22870085414505eb473737773230b364c3b790bc3b97'
TASKS = {'OpenDrawer': 'job-9j5eqjo0dfzj', 'CloseToasterOvenDoor': 'job-jpfh2j0xelgo',
         'PickPlaceDrawerToCounter': 'job-zqn304j2b9lw', 'CoffeeSetupMug': 'job-wv3zgiuj05j8',
         'TurnOffStove': 'job-z60s3hh8cn7w'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, state):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', type=Path, required=True)
    args = parser.parse_args()
    assert args.record.parent == ROOT and args.record.name == 'CHECKPOINT-TIER-WATCH-R1.json'
    assert not args.record.exists() and sha(TOOL) == TOOL_SHA
    lock = (ROOT / '.checkpoint-tier-watch.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(completed=False, pid=os.getpid(), started_unix=time.time(),
                 script_sha256=sha(Path(__file__)), tier_tool=str(TOOL),
                 tier_tool_sha256=TOOL_SHA, tier_records=[], new_controls=0, SGD=0, GPU_jobs=0)
    try:
        while True:
            eligible, closed = [], []
            for task, job in TASKS.items():
                production = ROOT / task / 'production-r1'
                manifest = read(production / 'manifest.json')
                assert manifest['hostname'] == job + '-master-0' and manifest['allocated_gpus'] == 8
                assert manifest['source'] == '0988b5f9e9bf31ac3db673cc2860959611b9ae12'
                clock = production / 'clock.json'
                if clock.exists():
                    assert read(clock)['exit_code'] == 0 and read(production / 'progress.json')['completed_iterations'] == 130
                    assert read(production / 'after-evaluation.json')['completed']
                    closed.append(task)
                steps = []
                for marker in (ROOT / task).glob(f'*/{task}/checkpoints/global_step_*/pilot-boundary.json'):
                    if not (marker.parent / 'COLD-STORAGE.json').exists():
                        boundary = read(marker)
                        if boundary['completed']:
                            steps.append(boundary['global_step'])
                if steps and min(steps) < max(steps):
                    eligible.append(task)
            state.update(checked_unix=time.time(), native_closed=closed, eligible=eligible)
            write(args.record, state)
            if eligible:
                output = ROOT / f'TIER-CHECKPOINT-WATCH-R1-{len(state["tier_records"])+1:03d}.json'
                assert not output.exists()
                subprocess.run(['python3', str(TOOL), '--root', str(ROOT), '--bos', BOS,
                                '--record', str(output)], check=True)
                result = read(output)
                assert result['completed'] and result['latest_full_boundaries_preserved']
                state['tier_records'].append(dict(path=str(output), sha256=sha(output),
                                                   boundaries=len(result['receipts'])))
                write(args.record, state)
                continue
            if len(closed) == 5:
                state.update(completed=True, finished_unix=time.time())
                write(args.record, state)
                return
            time.sleep(30)
    except BaseException:
        write(args.record.with_name(args.record.stem + '-ERROR.json'),
              dict(completed=False, failed_unix=time.time(), traceback=traceback.format_exc(), state=state))
        raise


if __name__ == '__main__':
    main()
