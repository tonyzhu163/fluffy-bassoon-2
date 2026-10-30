"""Compare source1/source2 and frozen target SAC in one frozen motor process."""
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
from .vla_libero import FrozenOpenPI, OfficialSAC, write_json
from .vla_motor_replay import parameter_signature
from .vla_source40_corpus import PI05_RECEIPT, save_motor_rng
from .vla_source40_evaluate import CostLedger, TargetCase, _save_recovery, evaluate_case, load_candidates
from .vla_source40_memory_replication import (CaptureSimulator, global_rng, public_signature,
    restore_global_rng, same_public, trace_rows)
from .vla_source40_schedule import content_hash
from .vla_source40_subset_evaluate import FrozenSAC, TargetServer, charged_actions, comparison_plan, load_scales
from .vla_source40_training import load_assembly
from .vla_source40_two_goal_evaluate import METHODS, RecordingController, TraceMotor, audit_group, two_goal_scale
from .vla_target_online_dsrl import RECIPE


FORMAT = 'openpi_pi05_common_runtime_source_scale_v1'
MAX_ACTIONS = 32000
SOURCE1_SHA = 'a52a2317f5b353d8e6afa0bc81131896426987197b49256d80b8faef354139dc'
SOURCE2_SHA = 'f1577bcbb5351743871ed2b3b0bcff627ea952f1a1aa0b724e147b7085e5d263'
TARGET_PINS = {
    0: dict(index='efdf1f359786b4d8db68178a4ca9b210ae802092136ad2f384536b70658c2154',
        contract='9ca99a5edf149d5e47838336cc5fa8fcc71917cd9168660ead4898f65f7b2b68',
        recovery='e40343bf1226a04295b6a537012678f1987e9e8248af549a6ee4306194ef587f'),
    1: dict(index='cd1c3227ab2725920064e47b84af86518ad7b3cc4d8a19967e11194af9669188',
        contract='422ff30ba28b4632a5f6124b6016d056d30a6dcb70b6f1d085e71815135e6677',
        recovery='fdf65d76037af9ae02d66ac4fe33e73dec1af5ad1cb4d0d7522dbf63b28554b1')}


def common_plan():
    cases = {row['case']['task_id']: row['case'] for row in comparison_plan() if row['primary']}
    plan = []
    for task in (0, 1):
        for method, condition in (('native', 'native'), ('frozen_target_sac', 'target_online_final')):
            plan.append(dict(case=deepcopy(cases[task]), method=method, source_condition=condition,
                updates=None, support_budgets=[0]))
        # Interleave source banks without changing the physical case or its RNG.
        for method in METHODS:
            for condition in ('source1', 'source2'):
                plan.append(dict(case=deepcopy(cases[task]), method=method, source_condition=condition,
                    updates=500, support_budgets=[0, 1]))
    return plan


def reservation(path, run_id, commit):
    value = json.loads(Path(path).read_text())
    row = value.get('reservations', {}).get(run_id, {})
    if (row.get('run_id') != run_id or row.get('source_commit') != commit
            or row.get('maximum_total_actions') != MAX_ACTIONS or row.get('purpose') != 'target_evaluation'
            or row.get('source_pretraining_data') is not False or row.get('fresh_ppo_training_data') is not False
            or row.get('campaign_sha256') != value.get('campaign_sha256')):
        raise ValueError('Require the pre-existing exact32k evaluation reservation; this command changes no budget')
    return row


def source1_scale(path):
    scales = load_scales(path, SOURCE1_SHA)
    assembly = load_assembly(scales['assembly_path'], allow_small_pilot=True)
    exposure = scales.get('source_exposure', {})
    if (assembly.get('small_pilot') is not True
            or assembly['source_task_keys'] != ['libero_spatial:00']
            or assembly['validation_task_keys'] != ['libero_spatial:00']
            or exposure.get('training_task_keys') != assembly['source_task_keys']
            or exposure.get('selected_source_histories') != assembly['selected_sources']):
        raise ValueError('Source1 must remain the original one-goal immutable corpus')
    selected = next(row for row in scales['scales'] if row['updates'] == 500)
    if file_sha256(selected['index_path']) != selected['index_sha256']:
        raise ValueError('Source1 fixed500 matrix hash differs')
    matrix = json.loads(Path(selected['index_path']).read_text())
    if len(matrix['controllers']) != 4 or {row['updates'] for row in matrix['controllers']} != {500}:
        raise ValueError('Source1 requires all four actual fixed500 candidates')
    return scales, selected, assembly


def target_manifest(root, task):
    directory = Path(root) / f'case-{task:02d}'
    paths = dict(index=directory / 'index.json', contract=directory / 'contract.json',
        recovery=directory / 'target-recovery.pkl')
    if any(file_sha256(path) != TARGET_PINS[task][name] for name, path in paths.items()):
        raise ValueError('Final target-online SAC index/contract/full recovery hash differs')
    index, contract = (json.loads(paths[name].read_text()) for name in ('index', 'contract'))
    expected_case = next(row['case'] for row in common_plan() if row['case']['task_id'] == task)
    if (not index.get('completed') or index.get('case') != expected_case or contract.get('case') != expected_case
            or index.get('contract_sha256') != content_hash(contract) or contract.get('recipe') != RECIPE
            or contract.get('motor_receipt_sha256') != PI05_RECEIPT or contract.get('target_only') is not True
            or contract.get('source_assembly_admission') is not False
            or index.get('source_checkpoint_reused') is not False or index.get('training_episodes') != 15
            or index.get('target_optimizer_updates') != 2400 or index.get('training_chunks', 0) <= 500
            or index.get('latest_recovery_sha256') != TARGET_PINS[task]['recovery']):
        raise ValueError('Require the actual completed15-episode target-only final SAC candidate')
    return dict(case=expected_case, paths={name: str(path) for name, path in paths.items()},
        hashes=TARGET_PINS[task], contract=contract, target_optimizer_updates=index['target_optimizer_updates'],
        replay_rows=index['replay_rows'], original_training_episodes=15,
        original_training_actions=index['training_primitive_steps'] + index['training_settling_steps'],
        original_physical_costs=index['costs'], original_total_action_charge=index['total_action_charge'],
        interpretation='Frozen final target-online candidate, previously adapted for15 episodes; new K0 query adds no SGD')


class FrozenTargetSAC(FrozenSAC):
    def frozen_signature(self):
        return content_hash(dict(parameters_and_optimizer=super().frozen_signature(),
            updates=self.learner.updates, replay_rows=len(self.learner.replay)))


def load_target_actor(manifest, dsrl_root):
    contract = manifest['contract']
    learner = OfficialSAC(dsrl_root, contract['learner_seed'])
    progress = learner.restore(manifest['paths']['recovery'], np.random.default_rng(contract['learner_seed']), contract)
    if (learner.updates != manifest['target_optimizer_updates'] or len(learner.replay) != manifest['replay_rows']
            or progress.get('training_episodes') != 15):
        raise ValueError('Restored target actor does not match its actual final training/replay receipt')
    return dict(policy=FrozenTargetSAC(learner), kind='frozen_target_sac', mode='frozen',
        sha256=manifest['hashes']['recovery'], target_training_actions=manifest['original_training_actions'])


def audit_control(path, result, group):
    if (result.get('completed') is not True or result.get('case') != group['case']
            or not result.get('weights_unchanged') or len(result['episodes']) != 1
            or [(row['method'], row['support_budget']) for row in result['trials']] != [(group['method'], 0)]):
        raise ValueError('Common-runtime K0 control receipt differs')
    episode = result['episodes'][0]
    arrays = trace_rows(path, episode)[0]['motor_observation']
    contexts = Path(path) / 'contexts'
    archive = json.loads((contexts / 'index.json').read_text())
    if not archive or any(Path(row['path']).name != row['path']
            or file_sha256(contexts / row['path']) != row['sha256'] for row in archive):
        raise ValueError('K0 control complete context/RNG archive differs')
    return dict(nominal_state_sha256=episode['initial_state_vector_sha256'],
        initial_observation_sha256=episode['initial_observation_sha256'],
        initial_public_arrays_sha256=public_signature(arrays),
        episode_context_index_sha256=file_sha256(contexts / 'index.json'), initial_arrays=arrays)


def run_common(motor, banks, targets, *, output, contract, server_factory, provenance,
        runtime_id, resume=False, reset_motor_rng=lambda seed: None, motor_rng_state=lambda: None,
        controller_factory=None, qualify=None, deadline=float('inf'), action_ceiling=MAX_ACTIONS,
        audit_fn=None, strict_initial_observations=True, graceful_deadline=False):
    """One shared motor object; continuations explicitly retain process lineage."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=resume)
    if resume:
        if json.loads((output / 'contract.json').read_text()) != contract:
            raise ValueError('Common-runtime resume contract differs')
        with (output / 'comparison-recovery.pkl').open('rb') as stream:
            saved = pickle.load(stream)
        if saved['contract_sha256'] != content_hash(contract):
            raise ValueError('Common-runtime cursor belongs to another contract')
        restore_global_rng(saved['global_rng'])
    else:
        write_json(output / 'contract.json', contract)
        _save_recovery(output / 'comparison-recovery.pkl', dict(contract_sha256=content_hash(contract),
            next_group=0, global_rng=global_rng(), motor_rng=motor_rng_state()))
    plan, cap = contract['groups'], contract['maximum_total_actions']
    if not 0 < cap <= action_ceiling or charged_actions(output) > cap:
        raise ValueError('Common-runtime joint physical ceiling differs')
    server, connected, completed, initials = None, None, [], {}
    stop_reason = None
    try:
        for ordinal, group in enumerate(plan):
            path, case = output / f'group-{ordinal:03d}', TargetCase(**group['case'])
            final, invocation_path = path / 'index.json', path / 'runtime-invocations.json'
            condition = group['source_condition']
            selected = banks[condition] if condition in banks else (
                {'frozen_target_sac': targets[case.task_id]} if condition == 'target_online_final' else {})
            source = dict(provenance[condition], source_condition=condition, optimizer_update_scale=group['updates'])
            if final.exists():
                result = json.loads(final.read_text())
            else:
                if time.monotonic() >= deadline:
                    stop_reason = 'invocation_time_reservation'
                    break
                if connected != case.case_id:
                    if server is not None:
                        server.close()
                    server, connected = server_factory(case), case.case_id
                local = CostLedger(path / 'physical-journal', max_primitive_steps=cap, max_settling_steps=cap)
                local_cost = sum(row[name] for row in local.costs().values() for name in
                    ('primitive_steps', 'uncertain_primitive_steps_ceiling', 'settling_steps', 'uncertain_settling_steps_ceiling'))
                group_cap = cap - (charged_actions(output) - local_cost)
                episodes = sum(budget + 1 for budget in group['support_budgets'])
                if not (path / 'contract.json').exists() and group_cap < episodes * (server.metadata['horizon'] + server.metadata['settle_steps']):
                    stop_reason = 'complete_group_physical_reservation'
                    break
                source['group_maximum_total_actions'] = group_cap
                raw, contexts = path / 'raw-rpc' / runtime_id, path / 'contexts'
                raw.mkdir(parents=True, exist_ok=True)
                contexts.mkdir(exist_ok=True)
                invocations = json.loads(invocation_path.read_text()) if invocation_path.exists() else []
                invocations.append(dict(runtime_id=runtime_id, charged_before=charged_actions(output),
                    purpose='Actual episode execution attempt in this specific motor process'))
                write_json(invocation_path, invocations)
                trace_motor, next_trial = TraceMotor(motor, raw), 0
                recovery = path / 'evaluation-recovery.pkl'
                if recovery.exists():
                    with recovery.open('rb') as stream:
                        next_trial = pickle.load(stream)['trial_index']
                def factory(spec):
                    nonlocal next_trial
                    base = controller_factory(spec, server) if controller_factory else (
                        spec['policy'] if spec.get('kind') == 'frozen_target_sac' else
                        FrozenHistoryController(spec['policy'], mode=spec['mode']))
                    value = RecordingController(base, output=contexts, method=group['method'],
                        trial_index=next_trial, motor_rng_state=motor_rng_state)
                    next_trial += 1
                    return value
                def hook(observation):
                    if qualify is not None:
                        qualify(observation, server)
                    write_json(path / 'qualification.json', dict(extra_physical_resets=0, runtime_id=runtime_id))
                try:
                    result = evaluate_case(CaptureSimulator(server, trace_motor, raw), trace_motor, selected,
                        case=case, output=path, provenance=source, max_primitive_steps=group_cap,
                        max_settling_steps=group_cap, support_budgets=tuple(group['support_budgets']),
                        methods=(group['method'],), controller_factory=factory,
                        reset_motor_rng=reset_motor_rng, motor_rng_state=motor_rng_state,
                        qualify=hook if ordinal == 0 else None, resume=(path / 'contract.json').exists(),
                        max_elapsed_seconds=contract['maximum_seconds_per_invocation'], external_deadline=deadline)
                except RuntimeError as error:
                    if (not graceful_deadline or str(error) != 'Target evaluation elapsed-time ceiling reached'
                            or time.monotonic() < deadline):
                        raise
                    # Complete episode cursor/RNG and before-RPC intents remain
                    # authoritative. Never count this partial group as a query.
                    invocations[-1].update(charged_after=charged_actions(output), graceful_deadline_stop=True)
                    write_json(invocation_path, invocations)
                    write_json(path / 'graceful-stop.json', dict(reason='invocation_time_reservation',
                        runtime_id=runtime_id, complete_episode_recovery=str(recovery),
                        physical_charge=charged_actions(output)))
                    stop_reason = 'invocation_time_reservation'
                    break
                # Unlearned controls have no controller; retain sampler/motor
                # and global RNG boundary using the authoritative recovery.
                if group['method'] in ('native', 'zero', 'tied_gaussian', 'tied_gaussian_clipped'):
                    with recovery.open('rb') as stream:
                        boundary = pickle.load(stream)
                    context = contexts / f'{group["method"]}-{time.time_ns()}.pkl'
                    _save_recovery(context, dict(
                        evaluation_recovery=boundary, global_rng=global_rng(), runtime_id=runtime_id))
                    archive_path = contexts / 'index.json'
                    archive = json.loads(archive_path.read_text()) if archive_path.exists() else []
                    archive.append(dict(path=context.name, sha256=file_sha256(context)))
                    write_json(archive_path, archive)
                invocations[-1]['charged_after'] = charged_actions(output)
                write_json(invocation_path, invocations)
            if any(result.get('source_provenance', {}).get(key) != value for key, value in source.items()
                    if key != 'group_maximum_total_actions'):
                raise ValueError('Completed group changed source bank or target-actor provenance')
            identity = (audit_fn(path, result, group) if audit_fn is not None else
                audit_group(path, result, group) if condition in banks else audit_control(path, result, group))
            identity['query_episode_random_seed'] = next(row['episode_random_seed'] for row in result['episodes'] if row['role'] == 'query')
            previous = initials.setdefault(case.case_id, identity)
            identity['public_arrays_match_first_group'] = same_public(previous['initial_arrays'], identity['initial_arrays'])
            identity['public_hash_matches_first_group'] = previous['initial_observation_sha256'] == identity['initial_observation_sha256']
            if (previous['nominal_state_sha256'] != identity['nominal_state_sha256']
                    or previous['query_episode_random_seed'] != identity['query_episode_random_seed']
                    or (strict_initial_observations and not (identity['public_arrays_match_first_group']
                        and identity['public_hash_matches_first_group']))):
                raise ValueError('Common-runtime methods changed actual initial arrays or nominal physical state')
            invocations = json.loads(invocation_path.read_text())
            completed.append(dict(group=group, result_path=str(final), result_sha256=file_sha256(final),
                runtime_ids=sorted({row['runtime_id'] for row in invocations}),
                audit={key: value for key, value in identity.items() if key != 'initial_arrays'}))
            _save_recovery(output / 'comparison-recovery.pkl', dict(contract_sha256=content_hash(contract),
                next_group=ordinal + 1, global_rng=global_rng(), motor_rng=motor_rng_state(), completed_groups=completed))
            write_json(output / 'progress.json', dict(completed_groups=len(completed), next_group=ordinal + 1,
                charged_total_actions=charged_actions(output), maximum_total_actions=cap))
            print(json.dumps(dict(group=ordinal, source_condition=condition, method=group['method'],
                task_id=case.task_id, query_success=[row['query_success'] for row in result['trials']],
                charged_total_actions=charged_actions(output))), flush=True)
    finally:
        if server is not None:
            server.close()
    runtime_ids = sorted({value for row in completed for value in row['runtime_ids']})
    result = dict(format=FORMAT, contract=contract, completed=len(completed) == len(plan),
        completed_groups=completed, stop_reason=stop_reason, charged_total_actions=charged_actions(output),
        actual_runtime_ids=runtime_ids,
        strict_common_runtime_primary_eligible=len(completed) == len(plan) and len(runtime_ids) == 1,
        support_query_separate=True, deployment_optimizer_updates=0,
        actual_initial_identities={key: {name: value for name, value in row.items() if name != 'initial_arrays'} for key, row in initials.items()})
    write_json(output / 'index.json', result)
    return result


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'checkpoint', 'openpi-root', 'dsrl-root', 'libero-root', 'simulator-python',
                 'source1-scales', 'source2-scales', 'target-final-root', 'budget-ledger'):
        ap.add_argument('--' + name, type=Path, required=True)
    for name in ('simulator-pythonpath', 'source-commit', 'expected-hostname', 'run-id'):
        ap.add_argument('--' + name, required=True)
    ap.add_argument('--max-total-actions', type=int, default=MAX_ACTIONS)
    ap.add_argument('--max-elapsed-seconds', type=float, default=2100.)
    ap.add_argument('--history-device', default='cuda:0')
    ap.add_argument('--port', type=int, default=8772)
    ap.add_argument('--resume', action='store_true')
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    if (sys.platform != 'linux' or socket.gethostname() != args.expected_hostname
            or args.max_total_actions != MAX_ACTIONS or not 0 < args.max_elapsed_seconds <= 2100
            or args.output.name != 'experiment' or args.output.parent.name != args.run_id):
        raise ValueError('Require the exact allocated Linux run and32k/2100-second common-runtime contract')
    root = Path(__file__).resolve().parents[3]
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() != args.source_commit
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip()):
        raise ValueError('Require the exact clean reviewed common-runtime source')
    from fluffy_bassoon.gpu_execution import validate_execution
    validate_execution(os.environ.get('FLUFFY_GPU_EXECUTION', 'baige'), os.environ)
    budget = reservation(args.budget_ledger, args.run_id, args.source_commit)
    scale_banks = dict(source1=source1_scale(args.source1_scales),
        source2=two_goal_scale(args.source2_scales, SOURCE2_SHA))
    targets_meta = {task: target_manifest(args.target_final_root, task) for task in (0, 1)}
    contract = dict(format=FORMAT, source_commit=args.source_commit, groups=common_plan(), budget_reservation=budget,
        maximum_total_actions=MAX_ACTIONS, maximum_seconds_per_invocation=args.max_elapsed_seconds,
        episode_ceiling=76, nominal_action_ceiling=31160, source1_scales_sha256=SOURCE1_SHA,
        source2_scales_sha256=SOURCE2_SHA, target_final_candidates=targets_meta,
        source_banks={name: dict(assembly_sha256=assembly['assembly_sha256'], assembly_file_sha256=scales['assembly_file_sha256'],
            fixed500_matrix_sha256=selected['index_sha256'], source_exposure=scales['source_exposure'],
            source_exposure_sha256=content_hash(scales['source_exposure'])) for name, (scales, selected, assembly) in scale_banks.items()},
        canonical_motor_key='jax.random.PRNGKey(seed), legacy uint32[2], shared by every comparator',
        source_condition='Separate provenance label; never part of TargetCase or stochastic seed',
        resume='Complete recoveries/costs preserved; a fresh-process continuation is non-primary for strict common-runtime inference',
        interpretation='Matched source-data development comparison in one motor process; target SAC had15 prior training episodes; no convergence or broad-transfer claim')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock = (args.output.parent / 'common-runtime.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    runtime_id = f'{socket.gethostname()}-{os.getpid()}-{time.time_ns()}'
    attempt = args.output.parent / 'runtime-attempts' / runtime_id
    attempt.mkdir(parents=True)
    deadline = time.monotonic() + args.max_elapsed_seconds
    args.model_config, args.feature_interface = 'pi05_libero', 'vla-prefix'
    motor = FrozenOpenPI(SimpleNamespace(**dict(vars(args), output=attempt)))
    banks, provenance = {}, {}
    for name, (scales, selected, assembly) in scale_banks.items():
        banks[name], provenance[name] = load_candidates(selected['index_path'], selected['index_sha256'],
            scales['assembly_path'], motor, seed=scales['seed'], device=args.history_device, allow_small_pilot=True)
        provenance[name].update(source_exposure=scales['source_exposure'],
            source_exposure_sha256=content_hash(scales['source_exposure']), fixed500_matrix_sha256=selected['index_sha256'])
    targets = {task: load_target_actor(metadata, args.dsrl_root) for task, metadata in targets_meta.items()}
    provenance['native'] = dict(steering_source_actions=0, target_training_actions=0)
    provenance['target_online_final'] = dict(target_final_candidates=targets_meta,
        source_pretraining_actions=0, target_training_costs_separate=True)
    # All steering modules and target actors are loaded before first inference.
    before = parameter_signature(motor.policy._model)
    write_json(attempt / 'motor-parameters-before.json', before)
    def reset_motor_rng(seed):
        import jax
        motor.policy._rng = jax.random.PRNGKey(seed)
    def qualify(observation, server):
        zero = np.zeros((10, 32), np.float32)
        a, b = motor.infer(observation, server.metadata['instruction'], zero), motor.infer(observation, server.metadata['instruction'], zero)
        if not np.allclose(a, b, rtol=0, atol=1e-5):
            raise RuntimeError('Common-runtime same-noise motor qualification failed')
        write_json(attempt / 'qualification.json', dict(same_noise_max_delta=float(np.abs(a-b).max()),
            features=motor.feature_encoder.qualify(observation, server.metadata['instruction']), extra_physical_resets=0))
    result = None
    try:
        result = run_common(motor, banks, targets, output=args.output, contract=contract, provenance=provenance,
            runtime_id=runtime_id, resume=args.resume, reset_motor_rng=reset_motor_rng,
            motor_rng_state=lambda: save_motor_rng(motor.policy._rng), qualify=qualify, deadline=deadline,
            server_factory=lambda case: TargetServer(task_id=case.task_id, python=args.simulator_python,
                pythonpath=args.simulator_pythonpath, output=attempt / f'server-{case.task_id}-{time.time_ns()}',
                port=args.port, libero_root=args.libero_root))
    finally:
        after = parameter_signature(motor.policy._model)
        write_json(attempt / 'motor-parameters-after.json', dict(before=before, after=after,
            equal=before == after, reference_probe_sha256='c565dd3f0c4005be728cf7597698cba937c4011cd5bd813700a89b789f9d69e9',
            matches_reference=after['sha256'] == 'c565dd3f0c4005be728cf7597698cba937c4011cd5bd813700a89b789f9d69e9'))
        if result is not None:
            result.update(motor_parameters_unchanged=before == after,
                motor_parameters_before_sha256=before['sha256'], motor_parameters_after_sha256=after['sha256'])
            result['strict_common_runtime_primary_eligible'] &= before == after
            write_json(args.output / 'index.json', result)
        if before != after:
            raise RuntimeError('Frozen motor parameter state changed during the common-runtime comparison')


if __name__ == '__main__':
    main()
