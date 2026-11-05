"""Bounded read-only observations of pooled learners every 20 minutes."""
import argparse
import datetime
import json
import math
from pathlib import Path
import time


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def observe(root, job, attempt, allocation, release_path, gpu_count, output, ledger_path):
    probe = read_json(allocation / 'GPU-PROBE.json')
    released = read_json(release_path)
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
                             checkpoint_mtime_unix=checkpoint.stat().st_mtime if checkpoint.exists() else None,
                             last_metrics_finite=metric is not None and all(
                                 math.isfinite(v) for v in metric.values() if isinstance(v, float)))
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
    ledger = read_json(ledger_path) if ledger_path else None
    budget = None
    if ledger:
        charged = reserved = ledger['preparation_GPU_hours']
        for row in ledger['allocations']:
            start = datetime.datetime.fromisoformat(row['created_shanghai'])
            end = datetime.datetime.fromisoformat(row['released_shanghai']) if row.get('released_shanghai') else now
            cost = row['GPUs'] * (end - start).total_seconds() / 3600
            charged += cost
            reserved += cost if row.get('released_shanghai') else max(cost, row['max_reservation_GPU_hours'])
        budget = dict(charged_GPU_hours=charged, remaining_GPU_hours=ledger['approved_GPU_hours']-charged,
                      remaining_after_reservations_GPU_hours=ledger['approved_GPU_hours']-reserved)
    result = dict(checked_shanghai=now.isoformat(), job_id=job, attempt=attempt,
        probe_passed=bool(probe and probe.get('status') == 'passed'
                          and probe.get('hostname') == job + '-master-0'
                          and len(probe.get('devices', [])) == gpu_count),
        methods=methods, worker_release=released, source_budget=budget,
        scheduler_release_requires_console=True)
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
    parser.add_argument('--allocation-dir', type=Path)
    parser.add_argument('--release-json', type=Path)
    parser.add_argument('--gpu-count', type=int, default=4)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--ledger', type=Path)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    allocation = args.allocation_dir or args.root / 'joint' / args.attempt
    release = args.release_json or allocation / 'RELEASED.json'
    output = args.output_dir or args.root / 'fit-monitor'
    for index in range(26):
        if observe(args.root, args.job_id, args.attempt, allocation, release,
                   args.gpu_count, output, args.ledger) or args.once:
            return
        if index != 25:
            time.sleep(1200)


if __name__ == '__main__':
    main()
