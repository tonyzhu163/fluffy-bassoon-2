"""Stage A native/tied-noise headroom screen on twelve closed LIBERO90 tasks.

Frozen official OpenPI pi0.5, no learner and no SGD. Records are development
exposure with role `target-headroom-screen` and never enter source40
pretraining. Plan: outputs/libero90-pi05-headroom-screen-20261101-r1/PLAN.md.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import re
import socket
import subprocess
import sys
import time

import numpy as np

from . import vla_target_online_dsrl as onset
from .vla_libero import run_episode
from .vla_noise_headroom import wilson
from .vla_source40_corpus import PI05_RECEIPT
from .vla_source40_ppo_worker import begin, finish
from .vla_source40_schedule import content_hash
from .vla_target_online_dsrl_task57 import output_bytes


FORMAT = 'openpi_pi05_libero90_headroom_screen_v1'
PLAN_COMMIT = '3b71a31'
TASKS = (0, 29, 33, 38, 44, 53, 59, 64, 70, 77, 79, 82)
STATES = tuple(range(1, 9))
FORBIDDEN_STATES = frozenset((0, *range(40, 50)))
ARMS = ('native', 'tied')
REPEATS = 2
EPISODES_PER_ARM = len(STATES) * REPEATS
NATIVE_ELIGIBLE = (2, 9)
TIED_MINIMUM = 2
TARGET_RATE = .2
MAX_ACTIONS, MAX_SECONDS, MAX_BYTES = 170000, 10800, 20 * 1024 ** 3
SCENE_SEED, HORIZON = 123, 400
DATA_ROLE = 'target-headroom-screen'
MOTOR_PARAMETER_SHA256 = 'c565dd3f0c4005be728cf7597698cba937c4011cd5bd813700a89b789f9d69e9'
STORAGE_RESERVE = 1024 ** 3
SELECTION = dict(native_successes_inclusive=list(NATIVE_ELIGIBLE), tied_minimum_successes=TIED_MINIMUM,
    per_arm_episodes=EPISODES_PER_ARM, target_native_rate=TARGET_RATE,
    order='closest native rate to .2, then more tied successes, then lower task ID',
    no_eligible='stop and report the full table; no widening')


def admit(task, state):
    if task not in TASKS:
        raise ValueError(f'Task {task} is not in the closed twelve-task screen list')
    if state in FORBIDDEN_STATES or state not in STATES:
        raise ValueError(f'State {state} is reserved (fixed100 state0, DSRL query 40..49) or outside 1..8')


def episode_plan():
    """Task-major (one simulator per task); arms interleaved within each state."""
    plan = []
    for task in TASKS:
        for state in STATES:
            for repeat in range(REPEATS):
                for arm in ARMS:
                    admit(task, state)
                    identity = f'libero90-{task:02d}-state{state}-{arm}-r{repeat}'
                    seed = int(content_hash(dict(format=FORMAT, task=task, state=state, arm=arm, repeat=repeat))[:8], 16)
                    plan.append(dict(identity=identity, suite='libero_90', task_id=task, initial_state=state,
                        scene_seed=SCENE_SEED, arm=arm, repeat=repeat, episode_seed=seed))
    return plan


def scientific_contract(source, grant_sha256):
    if not re.fullmatch('[0-9a-f]{40}', source) or not re.fullmatch('[0-9a-f]{64}', grant_sha256):
        raise ValueError('Require exact source and funding SHA256')
    plan = episode_plan()
    return dict(format=FORMAT, source_commit=source, plan_commit=PLAN_COMMIT, grant_sha256=grant_sha256,
        data_role=DATA_ROLE, target_only=True, source_assembly_admission=False,
        tasks=list(TASKS), states=list(STATES), forbidden_states=sorted(FORBIDDEN_STATES),
        arms=dict(native='independent H10x32 Gaussian per chunk through the explicit noise path',
                  tied='one fresh unbounded 32D Gaussian per chunk tiled over H10 (DSRL warmup distribution)'),
        repeats=REPEATS, episodes=len(plan), plan_sha256=content_hash(plan),
        scene_seed=SCENE_SEED, horizon=HORIZON, settling_steps=10, native_horizon=10, execute_steps=10,
        motor_receipt_sha256=PI05_RECEIPT, model_config='pi05_libero',
        motor_parameter_sha256=MOTOR_PARAMETER_SHA256, entire_motor_frozen=True,
        learner=None, optimizer_updates=0,
        selection=SELECTION,
        maximum_total_actions=MAX_ACTIONS, maximum_gpu_seconds=MAX_SECONDS, maximum_output_bytes=MAX_BYTES,
        interpretation='Development task selection only; not population success-rate estimates')


def load_grant(path, expected_sha256, source, run):
    if onset.sha256(path) != expected_sha256:
        raise ValueError('Headroom screen funding file SHA256 differs')
    grant = json.loads(Path(path).read_text())
    required = dict(format='pi05_libero90_headroom_screen_grant_v1', approved=True, source_commit=source,
        run_id=run, maximum_gpu_seconds=MAX_SECONDS, maximum_total_actions=MAX_ACTIONS,
        maximum_output_bytes=MAX_BYTES, tasks=list(TASKS), episodes=len(episode_plan()),
        motor_receipt_sha256=PI05_RECEIPT, motor_parameter_sha256=MOTOR_PARAMETER_SHA256, data_role=DATA_ROLE)
    if any(grant.get(key) != value for key, value in required.items()):
        raise ValueError('Headroom screen funding record must bind the unchanged ceilings/identity')
    return grant


def prepare_attempt(root, source, run, grant, grant_sha256, *, slice_seconds=MAX_SECONDS - 60,
                    resume=False, started_unix=None):
    if not re.fullmatch('[A-Za-z0-9-]+', run):
        raise ValueError('Require immutable headroom screen run identity')
    load_grant(grant, grant_sha256, source, run)
    root = Path(root)
    contract = dict(scientific_contract(source, grant_sha256), run_id=run)
    ledger = root / 'worker-time.json'
    prior = 0
    if ledger.exists():
        data = json.loads(ledger.read_text())
        if not resume or data['contract'] != contract or data['worker_seconds'] != MAX_SECONDS:
            raise ValueError('Resume only the exact headroom screen worker contract')
        prior = sum(row.get('charged_seconds', row['reserved_seconds']) for row in data['attempts'])
    elif resume:
        raise ValueError('Cannot resume without the original cumulative screen clock')
    if type(slice_seconds) is not int:
        raise ValueError('Require an integer slice')
    actual_slice = min(slice_seconds, MAX_SECONDS - prior - 60)
    if not 600 <= actual_slice:
        raise ValueError('Insufficient cumulative time for a bounded screen slice')
    started = time.time() if started_unix is None else float(started_unix)
    attempt = begin(root, contract, slice_seconds=actual_slice, worker_seconds=MAX_SECONDS,
        resume=resume, now=started)
    stop = math.floor(started + actual_slice)
    value = dict(attempt=str(attempt), slice_seconds=actual_slice, stop_unix=stop,
        driver_stop_unix=stop - 180, outer_timeout_unix=stop - 60, prior_charged_seconds=prior)
    onset.write_json(attempt / 'invocation-clock.json', value)
    return value


def read_records(root):
    path = Path(root) / 'episodes.jsonl'
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len({row['identity'] for row in rows}) != len(rows):
        raise ValueError('Duplicate completed screen episode identity')
    return rows


def append_record(root, row):
    with (Path(root) / 'episodes.jsonl').open('a') as stream:
        stream.write(json.dumps(row, sort_keys=True) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def screen_episode(sim, motor, entry, *, seed_motor):
    """One complete frozen episode; no learner, replay or optimizer exists."""
    admit(entry['task_id'], entry['initial_state'])
    seed_motor(motor, entry['episode_seed'])
    rng = np.random.default_rng(entry['episode_seed'])
    native = entry['arm'] == 'native'

    def choose(observation):
        # Native draws its full H10x32 tensor inside run_episode from the same rng.
        noise = np.zeros(32, np.float32) if native else rng.standard_normal(32).astype(np.float32)
        sim.context = dict(identity=entry['identity'], arm=entry['arm'], tied_noise=None if native else noise)
        return noise

    rows, summary = run_episode(sim, motor, choose, initial_state=entry['initial_state'],
        scene_seed=entry['scene_seed'], chunk=10, native=native, rng=rng, bounded=False, record_features=False)
    if not rows or not rows[-1]['done'] or summary['primitive_steps'] > HORIZON:
        raise RuntimeError('Require a complete native-horizon screen episode')
    for index, row in enumerate(rows):
        row.update(data_role=DATA_ROLE, target_only=True, source_assembly_admission=False,
            episode_id=entry['identity'], chunk_index=index, arm=entry['arm'])
    return rows, summary


def run_screen(sim, motor, task, *, root, contract, deadline=float('inf'), seed_motor=None, plan=None):
    """Run this task's remaining planned episodes; resume skips completed identities."""
    root = Path(root)
    seed_motor = seed_motor or (lambda m, seed: setattr(m.policy, '_rng', onset._key(seed)))
    plan = [entry for entry in (plan or episode_plan()) if entry['task_id'] == task]
    done = {row['identity']: row for row in read_records(root)}
    output = root / f'case-{task:02d}'
    journal = onset.JournaledSimulator(sim, output, root, case_cap=MAX_ACTIONS, total_cap=MAX_ACTIONS)
    nominal = {row['initial_state']: row['initial_state_vector_sha256'] for row in done.values()
               if row['task_id'] == task and row.get('initial_state_vector_sha256')}
    for entry in plan:
        if entry['identity'] in done:
            continue
        if time.monotonic() + 90 >= deadline:
            return 'time_reservation'
        if not journal.reserve_episode():
            return 'action_reservation'
        if output_bytes(root) + STORAGE_RESERVE > MAX_BYTES:
            return 'storage_reservation'
        journal.role, journal.episode_id = 'query', entry['identity']
        rows, summary = screen_episode(journal, motor, entry, seed_motor=seed_motor)
        state_hash = summary.get('initial_state_vector_sha256')
        if not state_hash or nominal.setdefault(entry['initial_state'], state_hash) != state_hash:
            raise RuntimeError('Screen reset changed the declared nominal physical case')
        receipt = onset.save_trace(output, entry['identity'], rows, summary, contract)
        record = dict(entry, success=receipt['success'], primitive_steps=int(receipt['primitive_steps']),
            settling_steps=10, chunks=receipt['chunks'], initial_state_vector_sha256=state_hash,
            initial_observation_sha256=receipt['initial_observation_sha256'], trace=f"case-{task:02d}/{receipt['trace']}",
            trace_sha256=receipt['trace_sha256'], data_role=DATA_ROLE, optimizer_updates=0,
            cumulative_physical_actions=journal.total_charge)
        append_record(root, record)
        print(json.dumps(dict(identity=entry['identity'], success=record['success'],
            steps=record['primitive_steps'], actions=journal.total_charge)), flush=True)
    return 'complete'


def rates(records):
    table = []
    for task in TASKS:
        row = dict(task_id=task)
        for arm in ARMS:
            episodes = [r for r in records if r['task_id'] == task and r['arm'] == arm]
            successes = sum(bool(r['success']) for r in episodes)
            row.update({f'{arm}_episodes': len(episodes), f'{arm}_successes': successes,
                f'{arm}_wilson95': wilson(successes, len(episodes)) if episodes else None})
        table.append(row)
    return table


def select(records):
    """The predeclared rule; requires every planned episode exactly once."""
    planned = {entry['identity'] for entry in episode_plan()}
    if {row['identity'] for row in records} != planned or len(records) != len(planned):
        raise ValueError('Selection requires all planned screen episodes exactly once')
    table = rates(records)
    low, high = NATIVE_ELIGIBLE
    eligible = [row for row in table if low <= row['native_successes'] <= high
                and row['tied_successes'] >= TIED_MINIMUM]
    eligible.sort(key=lambda row: (abs(row['native_successes'] / EPISODES_PER_ARM - TARGET_RATE),
                                   -row['tied_successes'], row['task_id']))
    result = dict(format=FORMAT, rates=table, eligible_task_ids=[row['task_id'] for row in eligible],
        rule=SELECTION)
    if not eligible:
        return dict(result, status='stop_no_eligible_task', selected_task_id=None, runner_up_task_id=None)
    return dict(result, status='selected', selected_task_id=eligible[0]['task_id'],
        runner_up_task_id=eligible[1]['task_id'] if len(eligible) > 1 else None)


def write_report(root):
    root = Path(root)
    records = read_records(root)
    table = rates(records)
    with (root / 'rates.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['task_id', 'native', 'native_wilson95', 'tied', 'tied_wilson95'])
        for row in table:
            writer.writerow([row['task_id'], f"{row['native_successes']}/{row['native_episodes']}",
                row['native_wilson95'], f"{row['tied_successes']}/{row['tied_episodes']}", row['tied_wilson95']])
    try:
        selection = select(records)
    except ValueError as error:
        selection = dict(format=FORMAT, status='incomplete', completed_episodes=len(records),
            planned_episodes=len(episode_plan()), reason=str(error), rates=table)
    onset.write_json(root / 'SELECTION.json', selection)
    return selection


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    clock = sub.add_parser('begin')
    for name in ('root', 'grant'):
        clock.add_argument('--' + name, type=Path, required=True)
    for name in ('source', 'run', 'grant-sha256'):
        clock.add_argument('--' + name, required=True)
    clock.add_argument('--slice-seconds', type=int, default=MAX_SECONDS - 60)
    clock.add_argument('--started-unix', type=float, required=True)
    clock.add_argument('--resume', action='store_true')
    close = sub.add_parser('finish')
    close.add_argument('--root', type=Path, required=True)
    close.add_argument('--attempt', type=Path, required=True)
    close.add_argument('--exit-code', type=int, required=True)
    report = sub.add_parser('report')
    report.add_argument('--output', type=Path, required=True)
    run = sub.add_parser('run')
    for name in ('output', 'checkpoint', 'openpi-root', 'libero-root', 'simulator-python', 'grant'):
        run.add_argument('--' + name, type=Path, required=True)
    for name in ('source-commit', 'expected-hostname', 'simulator-pythonpath', 'grant-sha256', 'run-id'):
        run.add_argument('--' + name, required=True)
    run.add_argument('--stop-unix', type=float, required=True)
    run.add_argument('--port', type=int, default=8791)
    run.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if args.operation == 'begin':
        value = prepare_attempt(args.root, args.source, args.run, args.grant, args.grant_sha256,
            slice_seconds=args.slice_seconds, resume=args.resume, started_unix=args.started_unix)
        print(f'{value["attempt"]}\t{value["stop_unix"]}\t{value["slice_seconds"]}')
        return
    if args.operation == 'finish':
        print(json.dumps(finish(args.root, args.attempt, args.exit_code)))
        return
    if args.operation == 'report':
        print(json.dumps({k: v for k, v in write_report(args.output).items() if k != 'rates'}))
        return
    remaining = args.stop_unix - time.time()
    if not math.isfinite(remaining) or not 0 < remaining <= MAX_SECONDS:
        raise ValueError('Require a live invocation deadline within the screen ceiling')
    deadline = time.monotonic() + remaining
    if sys.platform != 'linux' or socket.gethostname() != args.expected_hostname:
        raise ValueError('Screen models/simulator only on the explicitly identified allocated Linux host')
    from fluffy_bassoon.gpu_execution import validate_execution
    validate_execution(os.environ.get('FLUFFY_GPU_EXECUTION', 'baige'), os.environ)
    repo = Path(__file__).resolve().parents[3]
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() != args.source_commit
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True).strip()):
        raise ValueError('Require exact clean reviewed screen code pin')
    load_grant(args.grant, args.grant_sha256, args.source_commit, args.run_id)
    if onset.sha256(args.checkpoint / 'READY.json') != PI05_RECEIPT:
        raise ValueError('Require official OpenPI40 pi0.5 receipt')
    contract = scientific_contract(args.source_commit, args.grant_sha256)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    import fcntl
    lock = (args.output / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (args.output / 'contract.json').exists():
        if not args.resume or json.loads((args.output / 'contract.json').read_text()) != contract:
            raise ValueError('Screen resume contract differs or fresh run reuses an identity')
    else:
        onset.write_json(args.output / 'contract.json', contract)
        onset.write_json(args.output / 'plan.json', dict(plan_sha256=contract['plan_sha256'], episodes=episode_plan()))
    attempt = args.output / 'runtime-attempts' / str(time.time_ns())
    attempt.mkdir(parents=True)
    args.model_config, args.feature_interface = 'pi05_libero', 'sac'
    original_output, args.output = args.output, attempt
    motor = onset.FrozenOpenPI(args)
    args.output = original_output
    from .vla_motor_replay import parameter_signature
    before = parameter_signature(motor.policy._model)
    onset.write_json(attempt / 'motor-parameters-before.json', before)
    status, qualified = 'not_started', False
    try:
        if before['sha256'] != MOTOR_PARAMETER_SHA256:
            raise ValueError('Screen entire frozen motor parameter SHA256 differs')
        for task in TASKS:
            completed = {row['identity'] for row in read_records(args.output)}
            if all(e['identity'] in completed for e in episode_plan() if e['task_id'] == task):
                continue
            server = onset.TargetServer(task_id=task, python=args.simulator_python,
                pythonpath=args.simulator_pythonpath, output=attempt / f'server-{task:02d}',
                port=args.port, libero_root=args.libero_root)
            try:
                if (server.metadata.get('suite') != 'libero_90' or server.metadata.get('task_id') != task
                        or server.metadata.get('fresh_environment_on_reset') is not True):
                    raise ValueError('Screen simulator metadata differs from the declared task')
                if not qualified:
                    journal = onset.JournaledSimulator(server, args.output / f'case-{task:02d}', args.output,
                        case_cap=MAX_ACTIONS, total_cap=MAX_ACTIONS)
                    initial = journal.reset(STATES[0], SCENE_SEED)
                    zero = np.zeros((10, 32), np.float32)
                    first = motor.infer(initial['observation'], server.metadata['instruction'], zero)
                    second = motor.infer(initial['observation'], server.metadata['instruction'], zero.copy())
                    if motor.horizon != 10 or not np.allclose(first, second, rtol=0, atol=1e-5):
                        raise RuntimeError('Screen frozen H10 same-noise qualification failed')
                    onset.write_json(attempt / 'qualification.json', dict(task_id=task,
                        max_same_noise_delta=float(np.abs(first - second).max()), settling_actions=10,
                        data_role=DATA_ROLE, motor=motor.metadata))
                    qualified = True
                np.random.seed(0)
                random.seed(0)
                status = run_screen(server, motor, task, root=args.output, contract=contract, deadline=deadline)
            finally:
                server.close()
            if status != 'complete':
                break
        else:
            status = 'complete'
    finally:
        after = parameter_signature(motor.policy._model)
        onset.write_json(attempt / 'motor-parameters-after.json', dict(before=before, after=after,
            equal=before == after, expected_parameter_sha256=MOTOR_PARAMETER_SHA256))
        selection = write_report(args.output)
        onset.write_json(args.output / 'index.json', dict(format=FORMAT, run_id=args.run_id,
            source_commit=args.source_commit, status=status, selection_status=selection['status'],
            completed_episodes=len(read_records(args.output)), planned_episodes=len(episode_plan()),
            total_action_charge=onset.joint_charge(args.output), maximum_total_actions=MAX_ACTIONS,
            motor_parameters_unchanged=before == after, data_role=DATA_ROLE))
        if before != after:
            raise RuntimeError('Entire pi0.5 motor parameters changed during the screen')


if __name__ == '__main__':
    main()
