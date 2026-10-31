"""Authors' task57, ordinary target-online DSRL on entirely frozen OpenPI pi0.5.

Reuses the qualified PixelSAC/physical journal. Execution slices never reset the
500k-update, one-million-action, ten-hour scientific contract. Target records
are excluded from source40 controller pretraining.
"""
import argparse
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
from types import SimpleNamespace

import numpy as np

from . import vla_target_online_dsrl as onset
from .vla_source40_corpus import OnlineSourceSimulator, PI05_RECEIPT
from .vla_source40_ppo_worker import begin, finish
from .vla_source40_schedule import content_hash


FORMAT = 'openpi_pi05_task57_dsrl_v1'
REVIEWED_MAIN = 'f7dde2c94baee3ee59c4450a0ed30543d1cba67f'
MAX_UPDATES, PANEL_INTERVAL = 500000, 10000
MAX_ACTIONS, MAX_SECONDS, MAX_BYTES = 1000000, 36000, 64 * 1024 ** 3
TRAIN_STATES, QUERY_STATES = tuple(range(40)), tuple(range(40, 50))
DATA_ROLE = 'target-online-baseline'
MOTOR_PARAMETER_SHA256 = 'c565dd3f0c4005be728cf7597698cba937c4011cd5bd813700a89b789f9d69e9'


def target(state):
    return dict(benchmark='libero90', suite='libero_90', task_id=57,
        initial_state=int(state), scene_seed=123,
        case_id=f'libero90:57/state-{state}/seed-123')


def scientific_contract(source, grant_sha256):
    if not re.fullmatch('[0-9a-f]{40}', source) or not re.fullmatch('[0-9a-f]{64}', grant_sha256):
        raise ValueError('Require exact source and funding SHA256')
    return dict(format=FORMAT, source_commit=source, reviewed_main_commit=REVIEWED_MAIN,
        grant_sha256=grant_sha256, data_role=DATA_ROLE, target_only=True,
        source_assembly_admission=False, source_checkpoint_reused=False,
        task_id=57, suite='libero_90', train_states=list(TRAIN_STATES), query_states=list(QUERY_STATES),
        scene_seed=123, learner_seed=0, horizon=400, settling_steps=10,
        motor_receipt_sha256=PI05_RECEIPT, model_config='pi05_libero',
        motor_parameter_sha256=MOTOR_PARAMETER_SHA256,
        entire_motor_frozen=True, native_horizon=10, execute_steps=10,
        nominal_optimizer_stop=MAX_UPDATES, maximum_optimizer_updates=MAX_UPDATES + 780,
        complete_burst_overshoot_maximum=780,
        learned_query_interval_updates=PANEL_INTERVAL, query_episodes_per_panel=10,
        maximum_total_actions=MAX_ACTIONS, maximum_gpu_seconds=MAX_SECONDS,
        maximum_output_bytes=MAX_BYTES, prefix_features_recorded=False,
        training_state_schedule='completed training episode modulo40; no outcome selection',
        query_schedule='initial native; first complete burst crossing each10k; final native',
        native_sampling='independent H10x32 Gaussian through explicit custom-noise path',
        query_seed_bank='onset._seed(target(state),query,0), fixed across panels',
        recipe=dict(onset.RECIPE, departures='Frozen pi0.5 H10/execute10 and actual-duration discount; explicit train0..39/query40..49 state banks; first-postburst evaluation thresholds'),
        recovery='two atomic full slots; immutable milestone hardlinks; SHA-bound atomic query cursor',
        interpretation='Single-seed pi0.5-adapted DSRL curve on authors predeclared task57; not literal pi0 reproduction or convergence guarantee')


def load_grant(path, expected_sha256, source, run):
    if onset.sha256(path) != expected_sha256:
        raise ValueError('Task57 funding file SHA256 differs')
    grant = json.loads(Path(path).read_text())
    required = dict(format='pi05_task57_dsrl_grant_v1', approved=True, source_commit=source,
        run_id=run, maximum_gpu_seconds=MAX_SECONDS, maximum_total_actions=MAX_ACTIONS,
        maximum_output_bytes=MAX_BYTES, task_id=57, learner_seed=0,
        motor_receipt_sha256=PI05_RECEIPT, reviewed_main_commit=REVIEWED_MAIN, data_role=DATA_ROLE)
    required['motor_parameter_sha256'] = MOTOR_PARAMETER_SHA256
    if any(grant.get(key) != value for key, value in required.items()):
        raise ValueError('Task57 funding record must explicitly bind the unchanged phase ceilings/identity')
    return grant


def prepare_attempt(root, source, run, grant, grant_sha256, *, slice_seconds=7140,
                    resume=False, started_unix=None):
    if not re.fullmatch('[A-Za-z0-9-]+', run):
        raise ValueError('Require immutable task57 run identity')
    load_grant(grant, grant_sha256, source, run)
    root = Path(root)
    contract = dict(scientific_contract(source, grant_sha256), run_id=run)
    ledger = root / 'worker-time.json'
    if ledger.exists():
        data = json.loads(ledger.read_text())
        if not resume or data['contract'] != contract or data['worker_seconds'] != MAX_SECONDS:
            raise ValueError('Resume only the exact task57 worker contract')
        prior = sum(row.get('charged_seconds', row['reserved_seconds']) for row in data['attempts'])
    else:
        if resume:
            raise ValueError('Cannot resume without the original cumulative task57 clock')
        prior = 0
    actual_slice = min(int(slice_seconds), MAX_SECONDS - prior - 60)
    if type(slice_seconds) is not int or not 600 <= actual_slice <= MAX_SECONDS - 60:
        raise ValueError('Insufficient cumulative time for a bounded task57 slice')
    started = time.time() if started_unix is None else float(started_unix)
    if not math.isfinite(started) or started < 0:
        raise ValueError('Require finite invocation start')
    attempt = begin(root, contract, slice_seconds=actual_slice, worker_seconds=MAX_SECONDS,
        resume=resume, now=started)
    stop = math.floor(started + actual_slice)
    value = dict(attempt=str(attempt), slice_seconds=actual_slice, stop_unix=stop,
        driver_stop_unix=stop - 180, outer_timeout_unix=stop - 60,
        prior_charged_seconds=prior, maximum_worker_seconds=MAX_SECONDS)
    onset.write_json(attempt / 'invocation-clock.json', value)
    return value


def invocation_deadline(stop_unix, *, now_unix=None, now_monotonic=None):
    now = time.time() if now_unix is None else now_unix
    remaining = stop_unix - now
    if not math.isfinite(remaining) or not 0 < remaining <= MAX_SECONDS:
        raise ValueError('Require a live invocation deadline within the cumulative phase ceiling')
    return (time.monotonic() if now_monotonic is None else now_monotonic) + remaining


def output_bytes(root):
    """Count hardlinked milestone checkpoints once, including temporary files."""
    seen, total = set(), 0
    for directory, _, files in os.walk(root):
        for name in files:
            stat = (Path(directory) / name).stat()
            identity = stat.st_dev, stat.st_ino
            if identity not in seen:
                seen.add(identity)
                total += stat.st_size
    return total


def storage_status(root, output):
    slots = [path.stat().st_size for path in Path(output).glob('recovery-*.pkl')]
    traces = [path.stat().st_size for path in Path(output).glob('raw/*.pkl.gz')]
    full = max(slots, default=0)
    trace = max(traces, default=0)
    # Alternating-slot replacement can temporarily retain the old inode while
    # writing the new state; milestone hardlinks retain their own prior inode.
    # Existing replay growth + one complete raw episode fit this conservative
    # floor; measured checkpoint/trace sizes can increase it rather than shrink it.
    reservation = max(4 * 1024 ** 3, 3 * full + 4 * trace)
    return dict(actual_output_bytes=output_bytes(root), full_slot_logical_bytes=sum(slots),
        maximum_full_checkpoint_bytes=full, maximum_complete_trace_bytes=trace,
        next_episode_reservation_bytes=reservation, hardlinks_counted_once=True)


def native_query(sim, motor, learner, case):
    # Select native sampling without mutating the real actor update counter.
    view = SimpleNamespace(agent=learner.agent, replay=learner.replay, updates=0,
        frozen_signature=lambda: onset._signature(learner))
    rows, summary = onset.frozen_query(sim, motor, view, case, record_features=False)
    summary.update(target_optimizer_updates=learner.updates,
        underlying_actor_optimizer_updates=learner.updates,
        native_sampling_view_only=True, policy_signature=onset._signature(learner))
    return rows, summary


def _annotate(rows, *, episode_id, state, updates):
    elapsed = 0
    for chunk, row in enumerate(rows):
        count = int(row['primitive_steps'])
        row.update(data_role=DATA_ROLE, target_only=True, source_assembly_admission=False,
            episode_id=episode_id, chunk_index=chunk, episode_primitive_start=elapsed,
            episode_primitive_end=elapsed + count, initial_state=state,
            valid_primitive_mask=[True] * count + [False] * (10 - count),
            behavior_optimizer_updates=updates)
        elapsed += count


def run_profile(sim, motor, learner, sampler, *, output, root, contract, resume=False,
                deadline=float('inf'), stop_updates=MAX_UPDATES, panel_interval=PANEL_INTERVAL,
                query_states=QUERY_STATES, max_actions=MAX_ACTIONS, stop_after_updates=None,
                storage_limit_bytes=MAX_BYTES, storage_root=None):
    """New state-population driver; legacy fixed-case run_case remains unchanged.

    Smaller internal stop/panel/query values support CPU fixtures only. Public
    run CLI binds production constants. No checkpoint is taken mid-episode or
    mid-optimizer-burst; interrupted physical work remains charged in journals.
    """
    output, root = Path(output), Path(root)
    storage_root = root if storage_root is None else Path(storage_root)
    output.mkdir(parents=True, exist_ok=True)
    state_path, contract_path = output / 'state.json', output / 'contract.json'
    if (stop_updates < 1 or panel_interval < 1 or not query_states
            or stop_updates % panel_interval or any(state not in QUERY_STATES for state in query_states)):
        raise ValueError('Require complete predeclared task57 panel schedule')
    science_resume = resume and state_path.exists()
    if science_resume:
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError('Task57 immutable scientific contract changed')
        progress = json.loads(state_path.read_text())
        model = progress['model']
        checkpoint = output / model['path']
        if onset.sha256(checkpoint) != model['sha256']:
            raise ValueError('Task57 full recovery checksum differs')
        saved = learner.restore(checkpoint, sampler, contract)
        onset._restore_motor(motor, saved['motor_rng'])
        if (saved['training'] != progress['training'] or learner.updates != model['updates']
                or len(learner.replay) != model['replay_rows']
                or onset._signature(learner) != model['signature']):
            raise ValueError('Atomic task57 cursor/model/training recovery differs')
    else:
        if state_path.exists() or (contract_path.exists() and not resume):
            raise ValueError('Fresh task57 actor requires a new immutable output identity')
        if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
            raise ValueError('Task57 setup-only resume changed the scientific contract')
        if resume and any(json.loads(path.read_text()).get('role') in ('training', 'query')
                for path in (output / 'physical-journal').glob('*.json')):
            raise ValueError('Cannot restart scientific task57 data without its full recovery cursor')
        onset.write_json(contract_path, contract)
        progress = dict(format=FORMAT, phase='initial_native', next_threshold=panel_interval,
            active_panel=None, panels=[], training=dict(episodes=0, chunks=0,
                primitive_steps=0, settling_steps=0, receipts=[], state_hashes={}),
            query_state_hashes={}, model=None)
    journal = onset.JournaledSimulator(sim, output, root, case_cap=max_actions, total_cap=max_actions)
    wrapped = OnlineSourceSimulator(journal, learner, sampler, onset.sac_observation,
        warmup=500, batch=256, utd=20)

    def save(*, full=False, stop_reason=None):
        if full:
            previous = progress['model']
            slot = 1 if previous and previous['path'] == 'recovery-0.pkl' else 0
            path = output / f'recovery-{slot}.pkl'
            learner.save(path, sampler, contract, training=progress['training'], motor_rng=onset._save_motor(motor))
            progress['model'] = dict(path=path.name, sha256=onset.sha256(path), updates=learner.updates,
                replay_rows=len(learner.replay), signature=onset._signature(learner))
        # Full replacement precedes the cursor. The other slot stays valid if
        # this process stops between writes; milestone hardlinks stay immutable.
        onset.write_json(state_path, progress)
        storage = storage_status(storage_root, output)
        value = dict(progress, completed=progress['phase'] == 'complete', data_role=DATA_ROLE,
            target_only=True, source_assembly_admission=False, source_checkpoint_reused=False,
            contract_sha256=content_hash(contract), state_sha256=onset.sha256(state_path),
            target_optimizer_updates=learner.updates, replay_rows=len(learner.replay),
            costs=onset.costs(output), total_action_charge=onset.charged(output),
            maximum_total_actions=max_actions, output_bytes=storage['actual_output_bytes'], storage=storage,
            maximum_output_bytes=storage_limit_bytes, stop_reason=stop_reason,
            latest_recovery_sha256=progress['model']['sha256'])
        onset.write_json(output / 'index.json', value)
        return value

    def admitted():
        if time.monotonic() + 90 >= deadline:
            return 'complete_episode_time_reservation'
        if not journal.reserve_episode():
            return 'complete_episode_action_reservation'
        storage = storage_status(storage_root, output)
        if storage['actual_output_bytes'] + storage['next_episode_reservation_bytes'] > storage_limit_bytes:
            return 'complete_episode_storage_reservation'
        return None

    def start_panel(kind, threshold):
        name = f'{kind}-{threshold:06d}'
        archive_dir = output / 'milestones'
        archive_dir.mkdir(exist_ok=True)
        archive = archive_dir / f'{name}-recovery.pkl'
        if archive.exists():
            # A crash after hardlink creation but before cursor publication is
            # safe to recover only when the already immutable content matches.
            if onset.sha256(archive) != progress['model']['sha256']:
                raise ValueError('Immutable task57 milestone already exists with different content')
        else:
            os.link(output / progress['model']['path'], archive)
        progress['active_panel'] = dict(name=name, kind=kind, threshold_updates=threshold,
            actual_optimizer_updates=learner.updates, policy_signature=onset._signature(learner),
            recovery_path=str(archive.relative_to(output)), recovery_sha256=onset.sha256(archive),
            query_cursor=0, receipts=[])
        save()

    if not science_resume:
        save(full=True)
    while progress['phase'] != 'complete':
        phase = progress['phase']
        if phase in ('initial_native', 'learned_panel', 'final_native'):
            if progress['active_panel'] is None:
                threshold = 0 if phase == 'initial_native' else (
                    progress['next_threshold'] if phase == 'learned_panel' else stop_updates)
                start_panel('learned' if phase == 'learned_panel' else phase, threshold)
            panel = progress['active_panel']
            if learner.updates != panel['actual_optimizer_updates'] or onset._signature(learner) != panel['policy_signature']:
                raise RuntimeError('Task57 query panel actor lineage changed')
            while panel['query_cursor'] < len(query_states):
                reason = admitted()
                if reason:
                    return save(stop_reason=reason)
                state = query_states[panel['query_cursor']]
                case = target(state)
                episode_id = f'{panel["name"]}-state-{state:02d}'
                journal.role, journal.episode_id = 'query', episode_id
                rows, summary = (onset.frozen_query(journal, motor, learner, case, record_features=False)
                    if panel['kind'] == 'learned' else native_query(journal, motor, learner, case))
                state_hash = summary.get('initial_state_vector_sha256')
                previous_hash = progress['query_state_hashes'].get(str(state))
                if not state_hash or (previous_hash is not None and state_hash != previous_hash):
                    raise RuntimeError('Task57 matched query physical state changed across panels')
                progress['query_state_hashes'][str(state)] = state_hash
                _annotate(rows, episode_id=episode_id, state=state, updates=learner.updates)
                summary.update(data_role=DATA_ROLE, panel_name=panel['name'],
                    threshold_updates=panel['threshold_updates'], actual_optimizer_updates=learner.updates,
                    policy_signature=panel['policy_signature'], query_data_added_to_replay=False,
                    underlying_actor_optimizer_updates=learner.updates)
                panel['receipts'].append(onset.save_trace(output, episode_id, rows, summary, contract))
                panel['query_cursor'] += 1
                save()  # Query restores all training RNG; no duplicate replay checkpoint.
                print(json.dumps(dict(event='query', panel=panel['name'], completed_queries=panel['query_cursor'],
                    updates=learner.updates, actions=journal.total_charge)), flush=True)
            progress['panels'].append(panel)
            progress['active_panel'] = None
            if phase == 'initial_native':
                progress['phase'] = 'training'
            elif phase == 'final_native':
                progress['phase'] = 'complete'
            else:
                progress['next_threshold'] += panel_interval
                if progress['next_threshold'] <= stop_updates and learner.updates >= progress['next_threshold']:
                    progress['phase'] = 'learned_panel'
                else:
                    progress['phase'] = 'final_native' if learner.updates >= stop_updates else 'training'
            save()
            continue
        if learner.updates >= progress['next_threshold']:
            progress['phase'] = 'learned_panel'
            save()
            continue
        if stop_after_updates is not None and learner.updates >= stop_after_updates:
            return save(stop_reason='requested_invocation_update_pause')
        reason = admitted()
        if reason:
            return save(stop_reason=reason)
        training = progress['training']
        episode = training['episodes']
        state, episode_id = TRAIN_STATES[episode % len(TRAIN_STATES)], f'training-{episode:06d}'
        case = target(state)
        journal.role, journal.episode_id = 'training', episode_id
        before, before_updates = onset._signature(learner), learner.updates
        motor.policy._rng = onset._key(onset._seed(case, 'training_motor', episode))

        def choose(observation):
            noise = wrapped.choose(observation)
            journal.context = dict(wrapped.pending, native_initial_noise=onset.expand_noise(noise, 10),
                policy_signature=before, episode_id=episode_id, data_role=DATA_ROLE)
            return noise

        rows, summary = onset.run_episode(wrapped, motor, choose, initial_state=state,
            scene_seed=case['scene_seed'], chunk=10, bounded=False, record_features=False)
        if (onset._signature(learner) != before or learner.updates != before_updates
                or not rows or not rows[-1]['done'] or len(rows) != len(wrapped.provenance)
                or summary['primitive_steps'] > 400):
            raise RuntimeError('Task57 requires complete episodes with fixed behavior weights')
        state_hash = summary.get('initial_state_vector_sha256')
        old_hash = training['state_hashes'].get(str(state))
        if not state_hash or (old_hash is not None and state_hash != old_hash):
            raise RuntimeError('Task57 declared physical initial-state identity changed')
        metrics = wrapped.finish_episode()
        expected = before_updates + (20 * len(rows) if len(learner.replay) > 500 else 0)
        if learner.updates != expected or any(not np.isfinite(value) for value in metrics.values()):
            raise RuntimeError('Task57 published SAC update count/finite metric invariant failed')
        for row, provenance in zip(rows, wrapped.provenance, strict=True):
            row.update({key.replace('source_', 'target_', 1): value for key, value in provenance.items()})
            row['target_task_key'] = row.pop('source_task_key')
        _annotate(rows, episode_id=episode_id, state=state, updates=before_updates)
        summary.update(data_role=DATA_ROLE, completed=True, training_episode=episode,
            target_optimizer_updates_before=before_updates, target_optimizer_updates=learner.updates,
            replay_rows=len(learner.replay), update_metrics=metrics, actor_fixed_within_episode=True,
            behavior='warmup_gaussian' if before_updates == 0 else 'target_sac', policy_signature=before)
        training['receipts'].append(onset.save_trace(output, episode_id, rows, summary, contract))
        training['episodes'] += 1
        training['chunks'] += len(rows)
        training['primitive_steps'] += int(summary['primitive_steps'])
        training['settling_steps'] += 10
        training['state_hashes'][str(state)] = state_hash
        save(full=True)
        print(json.dumps(dict(event='training', episodes=training['episodes'], state=state,
            chunks=training['chunks'], updates=learner.updates, actions=journal.total_charge,
            metrics=metrics)), flush=True)
    return save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    clock = sub.add_parser('begin')
    for name in ('root', 'grant'):
        clock.add_argument('--' + name, type=Path, required=True)
    for name in ('source', 'run', 'grant-sha256'):
        clock.add_argument('--' + name, required=True)
    clock.add_argument('--slice-seconds', type=int, default=7140)
    clock.add_argument('--started-unix', type=float, required=True)
    clock.add_argument('--resume', action='store_true')
    close = sub.add_parser('finish')
    close.add_argument('--root', type=Path, required=True)
    close.add_argument('--attempt', type=Path, required=True)
    close.add_argument('--exit-code', type=int, required=True)
    run = sub.add_parser('run')
    for name in ('output', 'checkpoint', 'openpi-root', 'dsrl-root', 'libero-root', 'simulator-python', 'grant'):
        run.add_argument('--' + name, type=Path, required=True)
    for name in ('source-commit', 'expected-hostname', 'simulator-pythonpath', 'grant-sha256', 'run-id'):
        run.add_argument('--' + name, required=True)
    run.add_argument('--stop-unix', type=float, required=True)
    run.add_argument('--pause-after-updates', type=int)
    run.add_argument('--port', type=int, default=8787)
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
    started = time.monotonic()
    deadline = invocation_deadline(args.stop_unix)
    if sys.platform != 'linux' or socket.gethostname() != args.expected_hostname:
        raise ValueError('Task57 models/simulator only on the explicitly identified allocated Linux host')
    if args.pause_after_updates is not None and not 1 <= args.pause_after_updates <= MAX_UPDATES:
        raise ValueError('Invocation pause must stay inside the unchanged500k-update phase')
    from fluffy_bassoon.gpu_execution import validate_execution
    validate_execution(os.environ.get('FLUFFY_GPU_EXECUTION', 'baige'), os.environ)
    repo = Path(__file__).resolve().parents[3]
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() != args.source_commit
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True).strip()):
        raise ValueError('Require exact clean reviewed task57 code pin')
    load_grant(args.grant, args.grant_sha256, args.source_commit, args.run_id)
    if onset.sha256(args.checkpoint / 'READY.json') != PI05_RECEIPT:
        raise ValueError('Require official OpenPI40 pi0.5 receipt')
    contract = scientific_contract(args.source_commit, args.grant_sha256)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    import fcntl
    lock = (args.output / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    attempt = args.output / 'runtime-attempts' / str(time.time_ns())
    attempt.mkdir(parents=True)
    args.model_config, args.feature_interface = 'pi05_libero', 'sac'
    original_output, args.output = args.output, attempt
    motor = onset.FrozenOpenPI(args)
    args.output = original_output
    from .vla_motor_replay import parameter_signature
    before = parameter_signature(motor.policy._model)
    onset.write_json(attempt / 'motor-parameters-before.json', before)
    server, result = None, None
    try:
        if before['sha256'] != MOTOR_PARAMETER_SHA256:
            raise ValueError('Task57 entire frozen motor parameter SHA256 differs')
        server = onset.TargetServer(task_id=57, python=args.simulator_python,
            pythonpath=args.simulator_pythonpath, output=attempt / 'server-57',
            port=args.port, libero_root=args.libero_root)
        if (server.metadata.get('suite') != 'libero_90' or server.metadata.get('task_id') != 57
                or server.metadata.get('fresh_environment_on_reset') is not True):
            raise ValueError('Task57 simulator metadata differs from the declared experiment')
        output = args.output / 'case-57'
        journal = onset.JournaledSimulator(server, output, args.output, case_cap=MAX_ACTIONS, total_cap=MAX_ACTIONS)
        if not journal.reserve_episode():
            raise RuntimeError('No whole-episode budget remains for task57 runtime qualification')
        initial = journal.reset(0, 123)
        zero = np.zeros((10, 32), np.float32)
        first = motor.infer(initial['observation'], server.metadata['instruction'], zero)
        second = motor.infer(initial['observation'], server.metadata['instruction'], zero.copy())
        if motor.horizon != 10 or not np.allclose(first, second, rtol=0, atol=1e-5):
            raise RuntimeError('Task57 frozen H10 same-noise qualification failed')
        onset.write_json(attempt / 'qualification.json', dict(max_same_noise_delta=float(np.abs(first-second).max()),
            settling_actions=10, data_role=DATA_ROLE, motor=motor.metadata))
        np.random.seed(0)
        random.seed(0)
        learner, sampler = onset.OfficialSAC(args.dsrl_root, 0), np.random.default_rng(0)
        result = run_profile(server, motor, learner, sampler, output=output, root=args.output,
            contract=contract, resume=args.resume, deadline=deadline, stop_after_updates=args.pause_after_updates,
            storage_root=args.output.parent)
    finally:
        try:
            if server is not None:
                server.close()
        finally:
            after = parameter_signature(motor.policy._model)
            onset.write_json(attempt / 'motor-parameters-after.json', dict(before=before, after=after,
                equal=before == after, expected_parameter_sha256=MOTOR_PARAMETER_SHA256))
            if result is not None:
                onset.write_json(args.output / 'index.json', dict(result, run_id=args.run_id,
                    source_commit=args.source_commit, motor=motor.metadata,
                    elapsed_seconds_this_driver=time.monotonic() - started,
                    motor_parameters_unchanged=before == after,
                    motor_parameters_before_sha256=before['sha256'], motor_parameters_after_sha256=after['sha256']))
            if before != after:
                raise RuntimeError('Entire pi0.5 motor parameters changed during ordinary DSRL')


if __name__ == '__main__':
    main()
