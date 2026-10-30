"""Cumulative two-hour clock for the fixed six-case target baseline.

Only metadata is handled here. The existing begin/finish accounting charges
finished failures by actual time and unfinished attempts by their full slice.
"""
import argparse
import json
import math
from pathlib import Path
import re
import time

from .vla_source40_ppo_worker import begin, finish


WORKER_SECONDS = 7200
FIRST_SLICE_SECONDS = 7140
SCHEDULER_ALLOWANCE_SECONDS = 60


def worker_contract(source, run):
    if (not re.fullmatch(r'[0-9a-f]{40}', source)
            or not re.fullmatch(r'[A-Za-z0-9-]+', run)):
        raise ValueError('Require an exact pinned source and immutable target run identity')
    return dict(format='target_online_dsrl_six_worker_clock_v1', source_commit=source, run_id=run,
        maximum_worker_seconds=WORKER_SECONDS, maximum_driver_seconds=WORKER_SECONDS,
        target_task_ids=[0, 10, 20, 30, 40, 50], nominal_state=0, scene_seed=123,
        training_episode_milestones=[0, 1, 4, 8, 13, 16], maximum_actions_per_case=9500,
        maximum_total_actions=64000, learner_seed=0,
        recipe='Canonical authors PixelSAC >500chunks/batch256/UTD20; source pin binds all scientific settings',
        status='Two cumulative GPU-hours including setup, failures, scheduler allowance and retries; no automatic extension')


def prepare_attempt(root, source, run, *, resume=False, started_unix=None):
    root = Path(root)
    started = time.time() if started_unix is None else float(started_unix)
    if not math.isfinite(started) or started < 0:
        raise ValueError('Require a finite nonnegative invocation start time')
    contract = worker_contract(source, run)
    ledger_path = root / 'worker-time.json'
    if ledger_path.exists():
        data = json.loads(ledger_path.read_text())
        if not resume or data.get('contract') != contract or data.get('worker_seconds') != WORKER_SECONDS:
            raise ValueError('Target clock resumes require the original source/run/two-hour contract')
        charge = sum(row.get('charged_seconds', row['reserved_seconds']) for row in data['attempts'])
    else:
        if resume:
            raise ValueError('Cannot resume target baseline without its cumulative worker clock')
        charge = 0
    slice_seconds = min(FIRST_SLICE_SECONDS, WORKER_SECONDS - charge - SCHEDULER_ALLOWANCE_SECONDS)
    if slice_seconds < 600:
        raise RuntimeError('Less than a minimum ten-minute target slice remains in the original two-hour budget')
    # The launcher already holds its run lock. begin independently locks and
    # checks the time ledger again before adding the reservation.
    attempt = begin(root, contract, slice_seconds=slice_seconds, worker_seconds=WORKER_SECONDS,
        resume=resume, now=started)
    stop_unix = math.floor(started + slice_seconds)
    value = dict(attempt=str(attempt), slice_seconds=slice_seconds, stop_unix=stop_unix,
        prior_charged_seconds=charge, scheduler_allowance_seconds=SCHEDULER_ALLOWANCE_SECONDS,
        maximum_worker_seconds=WORKER_SECONDS)
    (attempt / 'invocation-clock.json').write_text(json.dumps(value, indent=2) + '\n')
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    start = sub.add_parser('begin')
    start.add_argument('--root', type=Path, required=True)
    start.add_argument('--source', required=True)
    start.add_argument('--run', required=True)
    start.add_argument('--started-unix', type=float, required=True)
    start.add_argument('--resume', action='store_true')
    end = sub.add_parser('finish')
    end.add_argument('--root', type=Path, required=True)
    end.add_argument('--attempt', type=Path, required=True)
    end.add_argument('--exit-code', type=int, required=True)
    args = parser.parse_args(argv)
    if args.operation == 'begin':
        value = prepare_attempt(args.root, args.source, args.run, resume=args.resume,
            started_unix=args.started_unix)
        print(f"{value['attempt']}\t{value['stop_unix']}\t{value['slice_seconds']}")
    else:
        print(json.dumps(finish(args.root, args.attempt, args.exit_code), allow_nan=False))


if __name__ == '__main__':
    main()
