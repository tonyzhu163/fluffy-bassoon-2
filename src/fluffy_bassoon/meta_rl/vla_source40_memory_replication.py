"""Predeclared paired memory replication with one physically shared support.

This is a counterfactual paired probe: support is executed once, then two real
fresh-reset queries retain or erase its context under the same frozen policy.
It is not two independently collected deployment trials or a reward lesion.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

from .vla_history import FrozenHistoryController, file_sha256
from .vla_libero import FrozenOpenPI, write_json
from .vla_source40_corpus import PI05_RECEIPT, save_motor_rng
from .vla_source40_evaluate import (CostLedger, TargetCase, _motor_signature, _policy_signature,
    _restore, _save_recovery, _snapshot, load_candidates, rollout_episode)
from .vla_source40_schedule import content_hash
from .vla_source40_subset_evaluate import TargetServer, load_scales


FORMAT = 'openpi_pi05_paired_memory_replication_v1'
MAX_ACTIONS = 8000
POLICY_SHA256 = 'ce0b36abf18c33268335816eea34b3d4d329e40a34fa60f1c97da5cf8437e654'
SCALES_SHA256 = 'a52a2317f5b353d8e6afa0bc81131896426987197b49256d80b8faef354139dc'
PHASES = ('shared_support', 'full_query', 'reset_query')


def replication_plan():
    return [TargetCase('libero90', f'libero90:00/state-{state}/seed-123/paired-memory',
        'libero_90', 0, state, 123) for state in range(1, 7)]


def charged(ledger):
    return sum(value[name] for value in ledger.costs().values() for name in
        ('primitive_steps', 'uncertain_primitive_steps_ceiling', 'settling_steps', 'uncertain_settling_steps_ceiling'))


def public_signature(observation):
    digest = hashlib.sha256()
    for key, value in sorted(observation.items()):
        array = np.asarray(value)
        digest.update(key.encode())
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def same_public(left, right):
    return left.keys() == right.keys() and all(
        np.asarray(left[key]).shape == np.asarray(right[key]).shape
        and np.asarray(left[key]).dtype == np.asarray(right[key]).dtype
        and np.array_equal(left[key], right[key]) for key in left)


def global_rng():
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_global_rng(saved):
    import torch
    random.setstate(saved['python'])
    np.random.set_state(saved['numpy'])
    torch.set_rng_state(saved['torch'])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(saved['cuda'])


def trace_rows(output, receipt):
    relative = Path(receipt['trace'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Replication trace must stay in its immutable run directory')
    path = Path(output) / relative
    if file_sha256(path) != receipt['trace_sha256']:
        raise ValueError('Replication trace checksum differs')
    with gzip.open(path, 'rb') as stream:
        return pickle.load(stream)


def raw_record(path, value):
    with gzip.open(path, 'xb', compresslevel=1) as stream:
        pickle.dump(value, stream, protocol=5)


class CaptureMotor:
    """Persist actual public input/noise/action before its physical RPC."""
    def __init__(self, motor, output):
        self.motor, self.output = motor, Path(output)
        self.horizon, self.metadata = motor.horizon, motor.metadata
        self.pending_feature, self.pending_path = None, None

    def feature_encoder(self, observation, instruction):
        self.pending_feature = self.motor.feature_encoder(observation, instruction)
        return self.pending_feature

    def infer(self, observation, instruction, noise):
        actions = self.motor.infer(observation, instruction, noise)
        self.pending_path = self.output / f'{time.time_ns()}-inference.pkl.gz'
        raw_record(self.pending_path, dict(observation=observation, instruction=instruction,
            public_features=self.pending_feature, native_initial_noise=np.asarray(noise).copy(),
            generated_actions=np.asarray(actions).copy()))
        return actions


class CaptureSimulator:
    def __init__(self, sim, motor, output):
        self.sim, self.motor, self.output = sim, motor, Path(output)
        self.metadata = sim.metadata

    def reset(self, state, seed):
        result = self.sim.reset(state, seed)
        raw_record(self.output / f'{time.time_ns()}-reset.pkl.gz', dict(
            initial_state=state, scene_seed=seed, reset_result=result))
        return result

    def step(self, actions):
        result = self.sim.step(actions)
        raw_record(self.output / f'{time.time_ns()}-outcome.pkl.gz', dict(
            inference_path=str(self.motor.pending_path), requested_actions=np.asarray(actions).copy(), outcome=result))
        return result


def paired_replication(sim, motor, policy, *, cases, output, source_provenance,
        policy_sha256=POLICY_SHA256, max_total_actions=MAX_ACTIONS, resume=False,
        controller_factory=None, reset_motor_rng=lambda seed: None,
        motor_rng_state=lambda: None, qualify=None, deadline=float('inf')):
    """One support plus full/reset branches per case, with boundary recovery."""
    if not cases or not 0 < max_total_actions <= MAX_ACTIONS or motor.horizon != 10:
        raise ValueError('Require explicit cases, native H10 and an at-most8000 joint action cap')
    if len({case.case_id for case in cases}) != len(cases) or len({case.initial_state for case in cases}) != len(cases):
        raise ValueError('Replication cases require unique predeclared IDs and nominal-state IDs')
    for case in cases:
        case.validate(sim.metadata)
    controller_factory = controller_factory or (lambda: FrozenHistoryController(policy, mode='full'))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'traces').mkdir(exist_ok=True)
    (output / 'contexts').mkdir(exist_ok=True)
    (output / 'raw-rpc').mkdir(exist_ok=True)
    contract = dict(format=FORMAT, cases=[asdict(case) for case in cases], policy_sha256=policy_sha256,
        motor_receipt_sha256=PI05_RECEIPT, source_provenance=source_provenance,
        simulator={key: value for key, value in sim.metadata.items() if key not in ('pid', 'hostname')},
        max_total_actions=max_total_actions, phases=list(PHASES), deployment_optimizer_updates=0,
        design='Counterfactual paired probe: one actual shared support, then full-context and erased-context fresh-reset queries',
        costs='Shared support is charged once; both query branches execute physically. All retries/uncertain RPCs/settling count.',
        query_randomness='Same case/role=query/ordinal=0 seed for both branches, independent of support action draws',
        interpretation='One goal, six predeclared new nominal states; no broad task-generalization or reward-causality claim')
    recovery_path = output / 'paired-recovery.pkl'
    if resume:
        if json.loads((output / 'contract.json').read_text()) != contract:
            raise ValueError('Paired replication resume contract differs')
        with recovery_path.open('rb') as stream:
            recovery = pickle.load(stream)
        if recovery['contract_sha256'] != content_hash(contract):
            raise ValueError('Paired replication recovery belongs to another contract')
        restore_global_rng(recovery['global_rng'])
    else:
        if (output / 'contract.json').exists():
            raise ValueError('Fresh replication requires a new immutable output identity')
        write_json(output / 'contract.json', contract)
        recovery = dict(contract_sha256=content_hash(contract), case_index=0, phase=0,
            support_context=None, support_motor_rng=None, records=[], global_rng=global_rng())
        _save_recovery(recovery_path, recovery)
    records = recovery['records']
    expected = [(case.case_id, phase) for case in cases for phase in PHASES]
    if (not 0 <= recovery['case_index'] <= len(cases) or not 0 <= recovery['phase'] < 3
            or len(records) != recovery['case_index'] * 3 + recovery['phase']
            or [(row['case_id'], row['phase']) for row in records] != expected[:len(records)]):
        raise ValueError('Paired replication cursor differs from complete chronological episodes')
    for receipt in records:
        trace_rows(output, receipt)
        if file_sha256(output / receipt['context']) != receipt['context_sha256']:
            raise ValueError('Saved support/query context checksum differs')
    ledger = CostLedger(output / 'physical-journal', max_primitive_steps=max_total_actions,
        max_settling_steps=max_total_actions, max_total_actions=max_total_actions)
    if charged(ledger) > max_total_actions:
        raise RuntimeError('Existing replication journal exceeds its immutable action ceiling')
    policy_signature, motor_signature = _policy_signature(policy), _motor_signature(motor)
    stop_reason = None
    while recovery['case_index'] < len(cases):
        case_index, phase = recovery['case_index'], recovery['phase']
        case = cases[case_index]
        if time.monotonic() >= deadline:
            stop_reason = 'invocation_time_reservation'
            break
        # Start/restart only if every still-required complete episode can fit.
        needed = (3 - phase) * (sim.metadata['horizon'] + sim.metadata['settle_steps'])
        if charged(ledger) + needed > max_total_actions:
            stop_reason = 'remaining_complete_episode_reservation'
            break
        controller = controller_factory()
        if phase == 1:
            if recovery['support_context'] is None:
                raise ValueError('Full query requires the actual saved shared-support context')
            _restore(controller, deepcopy(recovery['support_context']))
        # Phase2 deliberately starts with a fresh wrapper of exactly the same
        # trained policy; within-query recurrence remains active in both arms.
        initial_context = _snapshot(controller)
        role = 'support' if phase == 0 else 'query'
        episode_id = f'case-{case_index:03d}/{PHASES[phase]}'
        hook = qualify if not records and not (output / 'qualification.json').exists() else None
        capture_motor = CaptureMotor(motor, output / 'raw-rpc')
        capture_sim = CaptureSimulator(sim, capture_motor, output / 'raw-rpc')
        rows, receipt, sampler = rollout_episode(capture_sim, capture_motor, controller, case=case,
            method='ad500_shared_support' if phase == 0 else 'ad500_full' if phase == 1 else 'ad500_reset',
            role=role, ordinal=0, episode_id=episode_id, ledger=ledger,
            reset_motor_rng=reset_motor_rng, qualify=hook, deadline=deadline)
        trace = output / 'traces' / f'case-{case_index:03d}-{PHASES[phase]}-{time.time_ns()}.pkl.gz'
        with gzip.open(trace, 'xb', compresslevel=1) as stream:
            pickle.dump(rows, stream, protocol=5)
        receipt.update(case_id=case.case_id, case_index=case_index, phase=PHASES[phase],
            trace=str(trace.relative_to(output)), trace_sha256=file_sha256(trace),
            initial_public_arrays_sha256=public_signature(rows[0]['motor_observation']),
            policy_sha256=policy_sha256)
        boundary_context = _snapshot(controller)
        context = output / 'contexts' / f'case-{case_index:03d}-{PHASES[phase]}-{time.time_ns()}.pkl'
        _save_recovery(context, dict(case=asdict(case), phase=PHASES[phase],
            controller_before_episode=initial_context, controller_after_episode=boundary_context,
            actual_episode_random_seed=receipt['episode_random_seed'], motor_rng=motor_rng_state(),
            sampler=sampler, global_rng=global_rng(), policy_sha256=policy_sha256,
            contract_sha256=content_hash(contract)))
        receipt.update(context=str(context.relative_to(output)), context_sha256=file_sha256(context))
        case_records = [row for row in records if row['case_index'] == case_index]
        if phase:
            support = case_records[0]
            support_rows = trace_rows(output, support)
            if (receipt['initial_state_vector_sha256'] != support['initial_state_vector_sha256']
                    or receipt['initial_observation_sha256'] != support['initial_observation_sha256']
                    or not same_public(rows[0]['motor_observation'], support_rows[0]['motor_observation'])):
                raise RuntimeError('Paired support/query resets changed actual public arrays or nominal state')
            if phase == 2 and receipt['episode_random_seed'] != case_records[1]['episode_random_seed']:
                raise RuntimeError('Full/reset queries did not use exactly paired stochastic seeds')
        elif receipt['initial_state_vector_sha256'] in {row['initial_state_vector_sha256'] for row in records if row['phase'] == 'shared_support'}:
            raise RuntimeError('Different declared nominal IDs map to the same actual nominal state vector')
        if _policy_signature(policy) != policy_signature or _motor_signature(motor) != motor_signature:
            raise RuntimeError('Frozen motor/controller parameters changed during paired replication')
        records.append(receipt)
        if phase == 0:
            recovery['support_context'], recovery['support_motor_rng'] = boundary_context, motor_rng_state()
        recovery.update(case_index=case_index + int(phase == 2), phase=0 if phase == 2 else phase + 1,
            records=records, latest_controller=boundary_context, latest_query_initial_context=initial_context,
            motor_rng=motor_rng_state(), sampler=sampler, global_rng=global_rng(),
            boundary='Complete physical episode; later partial attempts remain charged by the shared RPC journal')
        if phase == 2:
            recovery['support_context'], recovery['support_motor_rng'] = None, None
        _save_recovery(recovery_path, recovery)
        write_json(output / 'progress.json', dict(completed_episodes=len(records), next_case=recovery['case_index'],
            next_phase=recovery['phase'], charged_total_actions=charged(ledger), maximum_total_actions=max_total_actions,
            latest_recovery_sha256=file_sha256(recovery_path)))
        print(json.dumps(dict(case_index=case_index, phase=PHASES[phase], success=receipt['success'],
            primitive_steps=receipt['primitive_steps'], charged_total_actions=charged(ledger))), flush=True)
    comparisons = []
    for case_index, case in enumerate(cases):
        selected = [row for row in records if row['case_index'] == case_index]
        if len(selected) != 3:
            continue
        support, full, reset = selected
        comparisons.append(dict(case=asdict(case), support_success=support['success'],
            full_query_success=full['success'], reset_query_success=reset['success'],
            shared_support_episode_id=support['episode_id'], full_query_episode_id=full['episode_id'],
            reset_query_episode_id=reset['episode_id'], query_rng_exactly_paired=full['episode_random_seed'] == reset['episode_random_seed'],
            nominal_state_sha256=support['initial_state_vector_sha256'],
            initial_public_arrays_exactly_equal=True,
            unique_actual_actions=sum(row['primitive_steps'] + row['settling_steps'] for row in selected)))
    committed = sum(row['primitive_steps'] + row['settling_steps'] for row in records)
    result = dict(format=FORMAT, completed=recovery['case_index'] == len(cases), contract=contract,
        comparisons=comparisons, episodes=records, stop_reason=stop_reason, physical_costs=ledger.costs(),
        charged_total_actions=charged(ledger), committed_unique_actions=committed,
        interrupted_or_uncertain_action_charge=charged(ledger) - committed,
        budget_matched_primary_eligible=charged(ledger) == committed,
        weights_unchanged=True, deployment_optimizer_updates=0,
        support_counted_once=True, counterfactual_paired_probe=True,
        full_query_successes=sum(row['full_query_success'] for row in comparisons),
        reset_query_successes=sum(row['reset_query_success'] for row in comparisons),
        full_only_successes=sum(row['full_query_success'] and not row['reset_query_success'] for row in comparisons),
        reset_only_successes=sum(row['reset_query_success'] and not row['full_query_success'] for row in comparisons),
        latest_recovery_sha256=file_sha256(recovery_path))
    write_json(output / 'index.json', result)
    return result


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'checkpoint', 'openpi-root', 'libero-root', 'simulator-python', 'scales'):
        ap.add_argument('--' + name, type=Path, required=True)
    for name in ('simulator-pythonpath', 'scales-sha256', 'source-commit', 'expected-hostname'):
        ap.add_argument('--' + name, required=True)
    ap.add_argument('--max-total-actions', type=int, default=MAX_ACTIONS)
    ap.add_argument('--max-elapsed-seconds', type=float, default=1200.)
    ap.add_argument('--history-device', default='cuda:0')
    ap.add_argument('--port', type=int, default=8769)
    ap.add_argument('--resume', action='store_true')
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    if (sys.platform != 'linux' or socket.gethostname() != args.expected_hostname
            or args.max_total_actions != MAX_ACTIONS or not 0 < args.max_elapsed_seconds <= 1200):
        raise ValueError('Require allocated Linux and the fixed8000-action/1200-second paired replication ceiling')
    if args.scales_sha256 != SCALES_SHA256:
        raise ValueError('Replication must retain the predeclared source-scale manifest')
    from fluffy_bassoon.gpu_execution import validate_execution
    validate_execution(os.environ.get('FLUFFY_GPU_EXECUTION', 'baige'), os.environ)
    root = Path(__file__).resolve().parents[3]
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() != args.source_commit
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip()):
        raise ValueError('Require the exact clean reviewed paired-replication commit')
    deadline = time.monotonic() + args.max_elapsed_seconds
    scales = load_scales(args.scales, args.scales_sha256)
    selected = next(row for row in scales['scales'] if row['updates'] == 500)
    matrix = json.loads(Path(selected['index_path']).read_text())
    row = next(row for row in matrix['controllers'] if row['objective'] == 'ad' and row['mode'] == 'full')
    if row['final_sha256'] != POLICY_SHA256:
        raise ValueError('Replication policy differs from the fixed AD500 full controller')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.output / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    attempt = args.output / 'attempts' / str(time.time_ns())
    attempt.mkdir(parents=True)
    args.model_config, args.feature_interface = 'pi05_libero', 'vla-prefix'
    motor = FrozenOpenPI(SimpleNamespace(**dict(vars(args), output=attempt)))
    candidates, provenance = load_candidates(selected['index_path'], selected['index_sha256'],
        scales['assembly_path'], motor, seed=scales['seed'], device=args.history_device, allow_small_pilot=True)
    server = TargetServer(task_id=0, python=args.simulator_python, pythonpath=args.simulator_pythonpath,
        output=attempt / 'server-task0', port=args.port, libero_root=args.libero_root)
    try:
        def reset_motor_rng(seed):
            import jax
            motor.policy._rng = jax.random.key(seed)
        def qualify(observation):
            zero = np.zeros((10, 32), np.float32)
            a, b = motor.infer(observation, server.metadata['instruction'], zero), motor.infer(observation, server.metadata['instruction'], zero)
            if not np.allclose(a, b, rtol=0, atol=1e-5):
                raise RuntimeError('Replication motor same-noise qualification failed')
            write_json(args.output / 'qualification.json', dict(same_noise_max_delta=float(np.abs(a-b).max()),
                features=motor.feature_encoder.qualify(observation, server.metadata['instruction']), extra_physical_resets=0))
        paired_replication(server, motor, candidates['ad_full']['policy'], cases=replication_plan(), output=args.output,
            source_provenance=dict(provenance, source_commit=args.source_commit,
                scales_sha256=args.scales_sha256, optimizer_update_scale=500),
            max_total_actions=args.max_total_actions, resume=args.resume,
            reset_motor_rng=reset_motor_rng, motor_rng_state=lambda: save_motor_rng(motor.policy._rng),
            qualify=qualify, deadline=deadline)
    finally:
        server.close()


if __name__ == '__main__':
    main()
