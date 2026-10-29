"""Fixed500 two-source-goal robot comparison on the original two target cases.

Six AD/DPT full/static/episode-reset methods, each with zero and one actual
support episode. No SAC learner, target SGD or new source data are constructed.
"""
import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import pickle
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

from .vla_history import FrozenHistoryController, file_sha256
from .vla_libero import FrozenOpenPI, write_json
from .vla_source40_corpus import save_motor_rng
from .vla_source40_evaluate import (CostLedger, TargetCase, _episode_rng, _motor_signature,
    _restore, _save_recovery, _snapshot, evaluate_case, load_candidates)
from .vla_source40_memory_replication import (CaptureMotor, CaptureSimulator, global_rng,
    public_signature, restore_global_rng, same_public, trace_rows)
from .vla_source40_schedule import content_hash
from .vla_source40_subset_evaluate import TargetServer, charged_actions, comparison_plan, load_scales
from .vla_source40_training import load_assembly


FORMAT = 'openpi_pi05_two_goal_target_comparison_v1'
MAX_ACTIONS = 16000
METHODS = ('ad_full', 'ad_static', 'ad_episode_reset', 'dpt_full', 'dpt_static', 'dpt_episode_reset')
SOURCE_TASKS = {'libero_spatial:00', 'libero_10:00'}


def two_goal_plan():
    original = {row['case']['task_id']: row['case'] for row in comparison_plan() if row['primary']}
    return [dict(case=deepcopy(original[task]), method=method, updates=500)
        for task in (0, 1) for method in METHODS]


def budget_reservation(path, run_id, commit):
    value = json.loads(Path(path).read_text())
    row = value.get('reservations', {}).get(run_id)
    if (not row or row.get('run_id') != run_id or row.get('source_commit') != commit
            or row.get('maximum_total_actions') != MAX_ACTIONS or row.get('purpose') != 'target_evaluation'
            or row.get('source_pretraining_data') is not False or row.get('fresh_ppo_training_data') is not False
            or row.get('campaign_sha256') != value.get('campaign_sha256')):
        raise ValueError('Require the existing exact16k target-evaluation escrow; this evaluator reserves no resources')
    return row


def two_goal_scale(path, sha256):
    value = load_scales(path, sha256)
    assembly = load_assembly(value['assembly_path'], allow_small_pilot=True)
    exposure = value.get('source_exposure', {})
    if (assembly.get('small_pilot') is not True or value.get('assembly_sha256') != assembly['assembly_sha256']
            or set(assembly['source_task_keys']) != SOURCE_TASKS
            or set(assembly['validation_task_keys']) != SOURCE_TASKS
            or exposure.get('training_task_keys') != assembly['source_task_keys']
            or exposure.get('validation_task_keys') != assembly['validation_task_keys']
            or exposure.get('selected_source_histories') != assembly['selected_sources']):
        raise ValueError('Require exactly two source goals with independent declared source-local validation')
    selected = next(row for row in value['scales'] if row['updates'] == 500)
    if file_sha256(selected['index_path']) != selected['index_sha256']:
        raise ValueError('Fixed500 matrix file checksum differs')
    matrix = json.loads(Path(selected['index_path']).read_text())
    if (len(matrix['controllers']) != 4 or {row['updates'] for row in matrix['controllers']} != {500}
            or matrix.get('contract', {}).get('assembly_sha256') != assembly['assembly_sha256']):
        raise ValueError('Every selected controller must be the fixed500 two-goal candidate')
    return value, selected, assembly


class TraceMotor(CaptureMotor):
    def frozen_signature(self):
        return _motor_signature(self.motor)


class RecordingController:
    """Retain every episode context and complete RNG without changing actions."""
    def __init__(self, controller, *, output, method, trial_index, motor_rng_state):
        self.controller, self.policy = controller, controller.policy
        self.output, self.method, self.trial_index = Path(output), method, trial_index
        self.motor_rng_state, self.next_episode, self.contexts = motor_rng_state, 0, []

    def set_episode_seed(self, seed):
        self.seed = seed
        _episode_rng(self.controller, seed)
        self.before = _snapshot(self.controller)

    def act(self, observation):
        return self.controller.act(observation)

    def observe(self, **outcome):
        self.controller.observe(**outcome)
        if outcome['done']:
            path = self.output / f'trial-{self.trial_index:03d}-episode-{self.next_episode:03d}-{time.time_ns()}.pkl'
            _save_recovery(path, dict(method=self.method, trial_index=self.trial_index,
                episode_position=self.next_episode, episode_random_seed=self.seed,
                controller_before_episode=self.before, controller_after_episode=_snapshot(self.controller),
                motor_rng=self.motor_rng_state(), global_rng=global_rng(),
                boundary='Actual complete physical episode; authoritative cursor is evaluation-recovery.pkl'))
            self.contexts.append(dict(path=path.name, sha256=file_sha256(path)))
            index_path = self.output / 'index.json'
            archive = json.loads(index_path.read_text()) if index_path.exists() else []
            archive.append(dict(path=path.name, sha256=file_sha256(path), method=self.method,
                trial_index=self.trial_index, episode_position=self.next_episode, episode_random_seed=self.seed))
            write_json(index_path, archive)
            self.next_episode += 1

    def snapshot(self):
        return dict(format='two_goal_recording_controller_v1', controller=_snapshot(self.controller),
            method=self.method, trial_index=self.trial_index, next_episode=self.next_episode,
            contexts=self.contexts, global_rng=global_rng())

    def restore(self, value):
        if (value['format'] != 'two_goal_recording_controller_v1' or value['method'] != self.method
                or value['trial_index'] != self.trial_index):
            raise ValueError('Episode-boundary recording context differs')
        for row in value['contexts']:
            if Path(row['path']).name != row['path'] or file_sha256(self.output / row['path']) != row['sha256']:
                raise ValueError('Complete episode context checksum differs')
        _restore(self.controller, deepcopy(value['controller']))
        self.next_episode, self.contexts = value['next_episode'], value['contexts']
        restore_global_rng(value['global_rng'])


def audit_group(path, result, group, provenance=None):
    path = Path(path)
    if (result.get('completed') is not True or result.get('case') != group['case']
            or result.get('weights_unchanged') is not True
            or [(row['method'], row['support_budget']) for row in result.get('trials', [])]
               != [(group['method'], 0), (group['method'], 1)]
            or result.get('source_provenance', {}).get('optimizer_update_scale') != 500
            or len(result.get('episodes', [])) != 3):
        raise ValueError('Saved method/case receipt differs from the fixed two-goal comparison')
    if provenance and any(result['source_provenance'].get(key) != value for key, value in provenance.items()):
        raise ValueError('Saved group uses different source exposure or fixed candidate provenance')
    episodes = result['episodes']
    arrays = [trace_rows(path, row)[0]['motor_observation'] for row in episodes]
    if (len({row['initial_state_vector_sha256'] for row in episodes}) != 1
            or len({row['initial_observation_sha256'] for row in episodes}) != 1
            or not all(same_public(arrays[0], array) for array in arrays[1:])
            or len({row['episode_random_seed'] for row in episodes if row['role'] == 'query'}) != 1):
        raise ValueError('Actual initial arrays/nominal state or paired query RNG differ within group')
    contexts_path = path / 'contexts/index.json'
    contexts = json.loads(contexts_path.read_text())
    for row in contexts:
        if Path(row['path']).name != row['path'] or file_sha256(path / 'contexts' / row['path']) != row['sha256']:
            raise ValueError('Archived complete episode context checksum differs')
    for episode in episodes:
        position = 1 if episode['support_budget'] == 1 and episode['role'] == 'query' else 0
        if not any(row['method'] == group['method'] and row['trial_index'] == episode['trial_index']
                and row['episode_position'] == position and row['episode_random_seed'] == episode['episode_random_seed']
                for row in contexts):
            raise ValueError('Committed episode lacks its full context/RNG recovery')
    return dict(nominal_state_sha256=episodes[0]['initial_state_vector_sha256'],
        initial_observation_sha256=episodes[0]['initial_observation_sha256'],
        initial_public_arrays_sha256=public_signature(arrays[0]),
        episode_context_index_sha256=file_sha256(contexts_path), initial_arrays=arrays[0])


def evaluate_groups(motor, candidates, *, server_factory, output, contract, provenance,
        resume=False, reset_motor_rng=lambda seed: None, motor_rng_state=lambda: None,
        qualify=None, controller_factory=None, deadline=float('inf')):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=resume)
    if resume:
        if json.loads((output / 'contract.json').read_text()) != contract:
            raise ValueError('Two-goal comparison resume contract differs')
        with (output / 'comparison-recovery.pkl').open('rb') as stream:
            previous = pickle.load(stream)
        if previous['contract_sha256'] != content_hash(contract):
            raise ValueError('Two-goal comparison cursor belongs to another contract')
        restore_global_rng(previous['global_rng'])
    else:
        write_json(output / 'contract.json', contract)
        _save_recovery(output / 'comparison-recovery.pkl', dict(contract_sha256=content_hash(contract),
            next_group=0, completed_groups=[], motor_rng=motor_rng_state(), global_rng=global_rng()))
    cap, plan = contract['maximum_total_actions'], contract['groups']
    if not 0 < cap <= MAX_ACTIONS or charged_actions(output) > cap:
        raise ValueError('Two-goal comparison exceeds its immutable joint action ceiling')
    server, connected, completed, initials = None, None, [], {}
    stop_reason = None
    try:
        for ordinal, group in enumerate(plan):
            path, case = output / f'group-{ordinal:03d}', TargetCase(**group['case'])
            final = path / 'index.json'
            if final.exists():
                result = json.loads(final.read_text())
            else:
                if time.monotonic() >= deadline:
                    stop_reason = 'invocation_time_reservation'
                    break
                if connected != case.task_id:
                    if server is not None:
                        server.close()
                    server, connected = server_factory(case), case.task_id
                local = CostLedger(path / 'physical-journal', max_primitive_steps=cap, max_settling_steps=cap)
                local_cost = sum(row[name] for row in local.costs().values() for name in
                    ('primitive_steps', 'uncertain_primitive_steps_ceiling', 'settling_steps', 'uncertain_settling_steps_ceiling'))
                group_cap = cap - (charged_actions(output) - local_cost)
                if not (path / 'contract.json').exists() and group_cap < 3 * (server.metadata['horizon'] + server.metadata['settle_steps']):
                    stop_reason = 'complete_group_physical_reservation'
                    break
                path.mkdir(parents=True, exist_ok=True)
                raw, contexts = path / 'raw-rpc', path / 'contexts'
                raw.mkdir(exist_ok=True)
                contexts.mkdir(exist_ok=True)
                trace_motor = TraceMotor(motor, raw)
                trace_sim = CaptureSimulator(server, trace_motor, raw)
                next_trial = 0
                recovery = path / 'evaluation-recovery.pkl'
                if recovery.exists():
                    with recovery.open('rb') as stream:
                        next_trial = pickle.load(stream)['trial_index']
                def factory(spec):
                    nonlocal next_trial
                    base = controller_factory(spec, server) if controller_factory else FrozenHistoryController(spec['policy'], mode=spec['mode'])
                    wrapped = RecordingController(base, output=contexts, method=group['method'],
                        trial_index=next_trial, motor_rng_state=motor_rng_state)
                    next_trial += 1
                    return wrapped
                source = dict(provenance, group_maximum_total_actions=group_cap, optimizer_update_scale=500,
                    source_goal_count=2, ordinary_dsrl_baseline='Separate target-online DSRL experiment; no joint SAC claim')
                def hook(observation):
                    if qualify is not None:
                        qualify(observation, server)
                    write_json(path / 'qualification.json', dict(extra_physical_resets=0))
                result = evaluate_case(trace_sim, trace_motor, candidates, case=case, output=path,
                    provenance=source, max_primitive_steps=group_cap, max_settling_steps=group_cap,
                    support_budgets=(0, 1), methods=(group['method'],), controller_factory=factory,
                    reset_motor_rng=reset_motor_rng, motor_rng_state=motor_rng_state,
                    qualify=hook if ordinal == 0 else None, resume=(path / 'contract.json').exists(),
                    max_elapsed_seconds=contract['maximum_seconds_per_invocation'], external_deadline=deadline)
            identity = audit_group(path, result, group, provenance)
            previous = initials.setdefault(case.case_id, identity)
            if (not same_public(previous['initial_arrays'], identity['initial_arrays'])
                    or previous['nominal_state_sha256'] != identity['nominal_state_sha256']
                    or previous['initial_observation_sha256'] != identity['initial_observation_sha256']):
                raise ValueError('Methods changed the actual predeclared target initial arrays or nominal state')
            completed.append(dict(group=group, result_path=str(final), result_sha256=file_sha256(final),
                audit={key: value for key, value in identity.items() if key != 'initial_arrays'}))
            _save_recovery(output / 'comparison-recovery.pkl', dict(contract_sha256=content_hash(contract),
                next_group=ordinal + 1, completed_groups=completed, motor_rng=motor_rng_state(), global_rng=global_rng()))
            write_json(output / 'progress.json', dict(completed_groups=len(completed), next_group=ordinal + 1,
                charged_total_actions=charged_actions(output), maximum_total_actions=cap))
    finally:
        if server is not None:
            server.close()
    result = dict(format=FORMAT, contract=contract, completed=len(completed) == len(plan),
        completed_groups=completed, stop_reason=stop_reason, charged_total_actions=charged_actions(output),
        actual_initial_identities={key: {name: value for name, value in row.items() if name != 'initial_arrays'} for key, row in initials.items()},
        support_query_separate=True, deployment_optimizer_updates=0)
    write_json(output / 'index.json', result)
    return result


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'checkpoint', 'openpi-root', 'libero-root', 'simulator-python', 'scales', 'budget-ledger'):
        ap.add_argument('--' + name, type=Path, required=True)
    for name in ('simulator-pythonpath', 'scales-sha256', 'source-commit', 'expected-hostname', 'run-id'):
        ap.add_argument('--' + name, required=True)
    ap.add_argument('--max-total-actions', type=int, default=MAX_ACTIONS)
    ap.add_argument('--max-elapsed-seconds', type=float, default=1320.)
    ap.add_argument('--history-device', default='cuda:0')
    ap.add_argument('--port', type=int, default=8771)
    ap.add_argument('--resume', action='store_true')
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    if (sys.platform != 'linux' or socket.gethostname() != args.expected_hostname
            or args.max_total_actions != MAX_ACTIONS or not 0 < args.max_elapsed_seconds <= 1320):
        raise ValueError('Require allocated Linux and the fixed16k-action/1320-second comparison ceiling')
    if args.output.name != 'experiment' or args.output.parent.name != args.run_id:
        raise ValueError('The reserved immutable run ID must match its exact output directory')
    root = Path(__file__).resolve().parents[3]
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() != args.source_commit
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip()):
        raise ValueError('Require the exact clean reviewed evaluation source')
    from fluffy_bassoon.gpu_execution import validate_execution
    validate_execution(os.environ.get('FLUFFY_GPU_EXECUTION', 'baige'), os.environ)
    reservation = budget_reservation(args.budget_ledger, args.run_id, args.source_commit)
    scales, selected, assembly = two_goal_scale(args.scales, args.scales_sha256)
    exposure = scales['source_exposure']
    contract = dict(format=FORMAT, source_commit=args.source_commit, groups=two_goal_plan(),
        scales_sha256=args.scales_sha256, matrix500_sha256=selected['index_sha256'],
        assembly_sha256=assembly['assembly_sha256'], assembly_file_sha256=scales['assembly_file_sha256'],
        source_exposure=exposure, source_exposure_sha256=content_hash(exposure),
        budget_reservation=reservation, maximum_total_actions=MAX_ACTIONS,
        maximum_seconds_per_invocation=args.max_elapsed_seconds, support_budgets=[0, 1],
        primary_episode_ceiling=36, primary_action_ceiling=14760,
        source_population='Exactly Spatial00 + LIBERO10:00; fixed500 candidates, full original source costs retained',
        target_population='Same predeclared LIBERO90 tasks0/1 state0 seed123 and query seeds as one-goal comparison',
        interpretation='Two-goal source-data development comparison; no full40, broad generalization or joint SAC baseline claim')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock = (args.output.parent / 'comparison.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    attempt = args.output.parent / 'runtime-attempts' / str(time.time_ns())
    attempt.mkdir(parents=True)
    deadline = time.monotonic() + args.max_elapsed_seconds
    args.model_config, args.feature_interface = 'pi05_libero', 'vla-prefix'
    motor = FrozenOpenPI(SimpleNamespace(**dict(vars(args), output=attempt)))
    candidates, provenance = load_candidates(selected['index_path'], selected['index_sha256'],
        scales['assembly_path'], motor, seed=scales['seed'], device=args.history_device, allow_small_pilot=True)
    provenance.update(two_source_assembly_sha256=assembly['assembly_sha256'],
        source_exposure_sha256=content_hash(exposure), source_exposure=exposure,
        fixed500_matrix_sha256=selected['index_sha256'], evaluation_source_commit=args.source_commit,
        ordinary_dsrl_baseline='Separate target-online DSRL experiment; no joint SAC claim')
    def reset_motor_rng(seed):
        import jax
        motor.policy._rng = jax.random.key(seed)
    def qualify(observation, server):
        zero = np.zeros((10, 32), np.float32)
        a, b = motor.infer(observation, server.metadata['instruction'], zero), motor.infer(observation, server.metadata['instruction'], zero)
        if not np.allclose(a, b, rtol=0, atol=1e-5):
            raise RuntimeError('Two-goal motor same-noise qualification failed')
        write_json(attempt / 'qualification.json', dict(same_noise_max_delta=float(np.abs(a-b).max()),
            features=motor.feature_encoder.qualify(observation, server.metadata['instruction']), extra_physical_resets=0))
    evaluate_groups(motor, candidates, server_factory=lambda case: TargetServer(task_id=case.task_id,
        python=args.simulator_python, pythonpath=args.simulator_pythonpath,
        output=attempt / f'server-{case.task_id}-{time.time_ns()}', port=args.port, libero_root=args.libero_root),
        output=args.output, contract=contract, provenance=provenance, resume=args.resume,
        reset_motor_rng=reset_motor_rng, motor_rng_state=lambda: save_motor_rng(motor.policy._rng),
        qualify=qualify, deadline=deadline)


if __name__ == '__main__':
    main()
