"""Bounded read-only observations of the four pooled learners every 20 minutes."""
import argparse
import datetime
import json
from pathlib import Path
import time


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def observe(root, job, attempt):
    allocation = root / 'joint' / attempt
    probe = read_json(allocation / 'GPU-PROBE.json')
    released = read_json(allocation / 'RELEASED.json')
    methods = {}
    for name in ('ad', 'dpt', 'dit', 'ic_iql'):
        directory = root / 'joint/fits' / name
        path = directory / 'metrics.jsonl'
        metric = None
        if path.exists():
            with path.open('rb') as stream:
                stream.seek(max(0, path.stat().st_size - 20000))
                lines = stream.read().decode(errors='replace').splitlines()
            for line in reversed(lines):
                try:
                    metric = json.loads(line)
                    break
                except json.JSONDecodeError:
                    pass
        checkpoint = directory / 'latest.pt'
        methods[name] = dict(status=read_json(directory / 'training-status.json'),
                             last_metrics=metric, resumable_checkpoint_present=checkpoint.exists(),
                             checkpoint_mtime_unix=checkpoint.stat().st_mtime if checkpoint.exists() else None)
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
    result = dict(checked_shanghai=now.isoformat(), job_id=job, attempt=attempt,
        probe_passed=bool(probe and probe.get('status') == 'passed'
                          and probe.get('hostname') == job + '-master-0'
                          and len(probe.get('devices', [])) == 4),
        methods=methods, worker_release=released, scheduler_release_requires_console=True)
    output = root / 'fit-monitor'
    output.mkdir(exist_ok=True)
    payload = json.dumps(result, indent=2) + '\n'
    (output / ('CHECK-' + now.strftime('%Y%m%d-%H%M%S') + '.json')).write_text(payload)
    temporary = output / 'LIVE-CHECK.json.tmp'
    temporary.write_text(payload)
    temporary.replace(output / 'LIVE-CHECK.json')
    print(json.dumps(result), flush=True)
    return released is not None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/mowenjie/fluffy-bassoon-2/outputs/robocasa-three-task-meta-baseline-20261104-r2'))
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--attempt', default='fit-4gpu-01')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    for index in range(26):
        if observe(args.root, args.job_id, args.attempt) or args.once:
            return
        if index != 25:
            time.sleep(1200)


if __name__ == '__main__':
    main()
