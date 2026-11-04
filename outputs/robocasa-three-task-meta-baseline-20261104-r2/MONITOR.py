#!/usr/bin/env python3
"""Read-only 20-minute preparation observations; never submits or stops jobs."""
import argparse
import datetime
import json
from pathlib import Path
import time

JOBS = (
    ('OpenDrawer', 'prepare-4gpu-03', 'job-2y8auhjtt9ak'),
    ('CloseToasterOvenDoor', 'prepare-4gpu-01', 'job-75dnd0uputmn'),
    ('CoffeeSetupMug', 'prepare-4gpu-01', 'job-38ix8q2xm8wt'),
)
SOURCE = 'd71cb532f6cf4933a4a5e236e1b085bce74ce969'


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def snapshot(root):
    rows = []
    for task, attempt, job in JOBS:
        directory = root / task
        invocation = directory / attempt
        probe = read_json(invocation / 'GPU-PROBE.json')
        released = read_json(invocation / 'RELEASED.json')
        logs = []
        for path in sorted(invocation.glob('prepare-*.log')):
            with path.open('rb') as stream:
                stream.seek(max(0, path.stat().st_size - 12000))
                tail = stream.read().decode(errors='replace')
            logs.append(dict(name=path.name, bytes=path.stat().st_size,
                             traceback='Traceback' in tail,
                             loaded_models=tail.count('openpi[eval]: loaded')))
        shards = []
        for path in sorted((directory / 'preparation').glob('shard-*')):
            receipt = read_json(path / 'SHARD.json')
            shards.append(dict(shard=path.name, progress=read_json(path / 'PROGRESS.json'),
                               completed=bool(receipt and receipt.get('completed'))))
        rows.append(dict(task=task, attempt=attempt, job_id=job,
                         probe_passed=bool(probe and probe.get('status') == 'passed'
                                           and probe.get('hostname') == job + '-master-0'
                                           and len(probe.get('devices', [])) == 4),
                         released=released, logs=logs, shards=shards,
                         assembly=read_json(directory / 'preparation' / 'ASSEMBLY.json')))
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
    result = dict(checked_shanghai=now.isoformat(), source_commit=SOURCE,
                  scheduler_status_requires_console=True,
                  scope='Worker evidence only; no scheduler submissions, stops, or automatic retries',
                  jobs=rows)
    output = root / 'monitor'
    output.mkdir(exist_ok=True)
    payload = json.dumps(result, indent=2) + '\n'
    (output / ('CHECK-' + now.strftime('%Y%m%d-%H%M%S') + '.json')).write_text(payload)
    tmp = output / 'LIVE-CHECK.json.tmp'
    tmp.write_text(payload)
    tmp.replace(output / 'LIVE-CHECK.json')
    print(json.dumps(dict(checked_shanghai=result['checked_shanghai'],
                          tasks=[dict(task=r['task'], probe=r['probe_passed'],
                                      progress=[s['progress'] for s in r['shards']],
                                      released=bool(r['released'])) for r in rows])), flush=True)
    return all(row['released'] for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/mowenjie/fluffy-bassoon-2/outputs/robocasa-three-task-meta-baseline-20261104-r2'))
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    for index in range(14):
        if snapshot(args.root) or args.once:
            return
        if index != 13:
            time.sleep(1200)


if __name__ == '__main__':
    main()
