"""Static per-worker time reservations for immutable shared-PPO continuations."""
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import time


def _write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def begin(root, contract, *, slice_seconds, worker_seconds, resume=False, now=None):
    """An unclosed attempt stays charged at its full reserved slice."""
    if (type(slice_seconds) is not int or not 600 <= slice_seconds <= 86400
            or type(worker_seconds) is not int or not slice_seconds + 60 <= worker_seconds <= 86400):
        raise ValueError('Require a bounded slice and at most24 cumulative worker-hours')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'worker-time.json'
    started = time.time() if now is None else float(now)
    with (root / 'worker-time.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            data = json.loads(path.read_text())
            if not resume or data['contract'] != contract or data['worker_seconds'] != worker_seconds:
                raise ValueError('Only the exact pinned worker contract and ceiling may resume')
        else:
            if resume:
                raise ValueError('Cannot resume a worker without its original time ledger')
            data = dict(format='source40_ppo_worker_time_v1', contract=contract,
                worker_seconds=worker_seconds, attempts=[])
        charge = sum(row.get('charged_seconds', row['reserved_seconds']) for row in data['attempts'])
        reservation = slice_seconds + 60  # Includes a scheduler-startup allowance.
        if charge + reservation > worker_seconds:
            raise RuntimeError('This slice exceeds the original cumulative worker-time reservation')
        attempt = root / 'attempts' / f'{int(started * 1000000)}-{len(data["attempts"]):04d}'
        attempt.mkdir(parents=True, exist_ok=False)
        row = dict(path=str(attempt), started_unix=started, slice_seconds=slice_seconds,
            reserved_seconds=reservation, status='started', scheduler_allowance_seconds=60)
        data['attempts'].append(row)
        _write(path, data)
        _write(attempt / 'attempt.json', row)
    return attempt


def finish(root, attempt, exit_code, *, now=None):
    root, attempt = Path(root), Path(attempt)
    ended = time.time() if now is None else float(now)
    with (root / 'worker-time.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / 'worker-time.json'
        data = json.loads(path.read_text())
        rows = [row for row in data['attempts'] if row['path'] == str(attempt)]
        if len(rows) != 1 or rows[0]['status'] != 'started':
            raise ValueError('Attempt is absent or already closed')
        row = rows[0]
        row.update(ended_unix=ended, status='finished', exit_code=int(exit_code),
            charged_seconds=math.ceil(max(0., ended - row['started_unix'])) + row['scheduler_allowance_seconds'])
        # Never truncate an actual overtime charge to make the plan look satisfied.
        row['slice_overrun'] = row['charged_seconds'] > row['reserved_seconds']
        _write(path, data)
        _write(attempt / 'attempt.json', row)
    return row


PPO_WORKER_SECONDS = 21600
PPO_DRIVER_SECONDS = 21600


def worker_contract(args):
    """Scheduler slice length may vary; scientific training inputs stay fixed."""
    contract = dict(source_commit=args.source, run_id=args.run, maximum_total_actions=args.action_cap,
        updates=args.updates, memory_mode=args.memory_mode, maximum_driver_seconds=PPO_DRIVER_SECONDS,
        campaign_sha256=hashlib.sha256(args.campaign.read_bytes()).hexdigest(),
        schedule_sha256=hashlib.sha256(args.schedule.read_bytes()).hexdigest(),
        architecture=dict(width=args.width, layers=args.layers, heads=args.heads, context_length=args.context_length),
        optimizer=dict(learning_rate=1e-4, epochs=2), seed=2,
        status='Approved six-hour cumulative source-PPO ceiling; scheduler allocation is separately identified')
    if getattr(args, 'camera_reset_max_lsb', 0):
        from .vla_reset_identity import RESET_IDENTITY
        contract['reset_identity'] = RESET_IDENTITY
    return contract


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='operation', required=True)
    start = sub.add_parser('begin')
    for name in ('root', 'campaign', 'schedule'):
        start.add_argument('--' + name, type=Path, required=True)
    for name in ('source', 'run', 'memory-mode'):
        start.add_argument('--' + name, required=True)
    for name in ('action-cap', 'updates', 'slice-seconds', 'worker-seconds'):
        start.add_argument('--' + name, type=int, required=True)
    for name, default in (('width', 128), ('layers', 2), ('heads', 4), ('context-length', 300)):
        start.add_argument('--' + name, type=int, default=default)
    start.add_argument('--resume', action='store_true')
    start.add_argument('--camera-reset-max-lsb', type=int, choices=(0, 1), default=0)
    end = sub.add_parser('finish')
    end.add_argument('--root', type=Path, required=True)
    end.add_argument('--attempt', type=Path, required=True)
    end.add_argument('--exit-code', type=int, required=True)
    args = ap.parse_args()
    if args.operation == 'begin':
        if (len(args.source) != 40 or any(c not in '0123456789abcdef' for c in args.source)
                or not args.run or any(not(c.isalnum() or c == '-') for c in args.run)
                or args.memory_mode not in ('full', 'episode') or args.action_cap < 1650
                or args.updates < 1 or any(value < 1 for value in
                    (args.width, args.layers, args.heads, args.context_length))
                or args.width % args.heads or args.width % 2):
            raise ValueError('Invalid pinned shared-PPO worker identity or action budget')
        if args.worker_seconds != PPO_WORKER_SECONDS:
            raise ValueError('Approved source PPO has an exact21600-second cumulative worker ceiling')
        contract = worker_contract(args)
        print(begin(args.root, contract, slice_seconds=args.slice_seconds,
                    worker_seconds=args.worker_seconds, resume=args.resume))
    else:
        print(json.dumps(finish(args.root, args.attempt, args.exit_code), allow_nan=False))


if __name__ == '__main__':
    main()
