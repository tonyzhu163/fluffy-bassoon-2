"""Bounded all90 frozen-PPO deployment after successful terminal source validation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

from .vla_common_runtime_evaluate import run_common
from .vla_history import file_sha256
from .vla_libero import FrozenOpenPI, write_json
from .vla_motor_replay import parameter_signature
from .vla_source40_corpus import PI05_RECEIPT, save_motor_rng
from .vla_source40_frozen_validation import load_models
from .vla_source40_multitask_ppo import source_tasks
from .vla_source40_noise_controls import graceful_seconds
from .vla_source40_ppo_target import candidates, controller_factory
from .vla_source40_ppo_wave_plan import FORMAT as PLAN_FORMAT, LEDGER_FORMAT, build_plan, terminal_validation
from .vla_source40_schedule import content_hash, load_schedule
from .vla_source40_target_firstwave import PlannedServer, audit_firstwave


FORMAT = 'source40_frozen_ppo_target_wave_worker_v1'


def load_plan(path, expected, source):
    if file_sha256(path) != expected:
        raise ValueError('PPO target plan file checksum differs')
    value = json.loads(Path(path).read_text())
    if (value.get('format') != PLAN_FORMAT or value.get('contract', {}).get('source_commit') != source
            or value.get('ppo_wave_sha256') != content_hash({k: v for k, v in value.items() if k != 'ppo_wave_sha256'})):
        raise ValueError('PPO target source/plan content differs')
    proof = value['contract']['source_validation']
    if terminal_validation(proof['root'], proof['index_sha256']) != proof:
        raise ValueError('Terminal source-validation policy/index evidence changed')
    if value != build_plan(worktree=value['contract']['worktree'], source_commit=source, source_validation=proof):
        raise ValueError('PPO target methods, cases or balanced worker assignments differ')
    return value


def reservation(path, plan, worker, run, source):
    value = json.loads(Path(path).read_text())
    rows = value.get('reservations', {})
    if (value.get('format') != LEDGER_FORMAT or value.get('ppo_wave_sha256') != plan['ppo_wave_sha256']
            or value.get('maximum_total_actions') != 140000 or value.get('maximum_gpu_seconds') != 7200
            or value.get('source_campaign_immutable') is not True or not value.get('scope_amendment')
            or len(rows) != 2 or {r.get('worker') for r in rows.values()} != {0, 1}):
        raise ValueError('Require the pre-existing separate140k/two-hour frozen-PPO target scope')
    for key, row in rows.items():
        if (row.get('run_id') != key or row.get('source_commit') != source or row.get('ppo_wave_sha256') != plan['ppo_wave_sha256']
                or row.get('maximum_total_actions') != 70000 or row.get('worker_seconds') != 3600
                or row.get('purpose') != 'target_evaluation' or row.get('source_pretraining_data') is not False
                or row.get('fresh_ppo_training_data') is not False):
            raise ValueError('PPO target worker contract differs from its original allocation/action hold')
    record = rows.get(run)
    if record is None or record.get('worker') != worker:
        raise ValueError('Require this exact existing frozen-PPO target worker reservation')
    return dict(ledger_path=str(Path(path).resolve()), record=record, record_sha256=content_hash(record))


def load_terminal_models(plan, motor, schedule, output, device):
    proof = plan['contract']['source_validation']
    if schedule['schedule_sha256'] != proof['schedule_sha256']:
        raise ValueError('PPO source schedule differs from successful validation')
    output = Path(output)
    output.mkdir(exist_ok=True)
    for mode, spec in proof['exports'].items():
        destination = output / f'frozen-{mode}.pt'
        if destination.exists():
            if file_sha256(destination) != spec['sha256']:
                raise ValueError('Worker frozen export changed on resume')
        else:
            # Copy exact already-validated bytes, retaining the SAME model SHA.
            # load_models then re-admits R3 source recoveries and verifies these
            # existing exports; it does not re-export into the source run.
            shutil.copyfile(spec['path'], destination)
        if file_sha256(destination) != spec['sha256']:
            raise ValueError('Immutable source-validation export copy differs')
    models, provenance = load_models(proof['models_manifest_path'], proof['models_manifest_sha256'], motor=motor,
        tasks=[r['task_key'] for r in source_tasks(schedule)], schedule_sha256=schedule['schedule_sha256'],
        output=output, device=device)
    for mode, spec in proof['exports'].items():
        if (provenance[mode]['policy_sha256'] != spec['sha256'] or provenance[mode]['tensor_sha256'] != spec['tensor_sha256']
                or provenance[mode]['recovery_sha256'] != spec['recovery_sha256']
                or provenance[mode]['terminal_lineage'] != spec['terminal_lineage']):
            raise ValueError('Frozen target model differs from the actual terminal validation export/lineage')
    return candidates(models, provenance), provenance


def reuse_empty_history(plan, completed, selected):
    full, reset = selected['ppo_full'], selected['ppo_episode_reset']
    if (full['policy'] is not reset['policy'] or full['sha256'] != reset['sha256']
            or full['mode'] != 'full' or reset['mode'] != 'episode'):
        raise ValueError('K0 reset reuse must use identical full-policy weights, not the separately trained reset')
    by_id = {r['group']['group_id']: r for r in completed}
    rows = []
    for entry in plan['cases']:
        reference = entry['zero_shot_reuse']['ppo_episode_reset']
        if reference in by_id:
            saved = by_id[reference]
            result = json.loads(Path(saved['result_path']).read_text())
            query = next(e for e in result['episodes'] if e['role'] == 'query')
            rows.append(dict(case_id=entry['case']['case_id'], method='ppo_episode_reset', support_budget=0,
                reference_group_id=reference, reference_result_sha256=saved['result_sha256'],
                reference_query_episode_id=query['episode_id'], reference_trace_sha256=query['trace_sha256'],
                query_success=query['success'], policy_sha256=full['sha256'], additional_physical_actions=0,
                proof='Same full-policy weights, empty context, identical within-query transitions and paired global Torch episode RNG; trained reset remains a separate method'))
    return rows


def audit_ppo(path, result, group):
    identity = audit_firstwave(path, result, group)
    if not (identity['within_group_initial_public_arrays_equal'] and identity['within_group_initial_observation_hashes_equal']):
        raise ValueError('Frozen PPO support/query initial public observations differ')
    return identity


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'plan', 'ledger', 'schedule', 'checkpoint', 'openpi-root'):
        p.add_argument('--' + name, type=Path, required=True)
    for name in ('source-commit', 'run-id', 'expected-hostname', 'plan-sha256'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--worker', type=int, choices=(0, 1), required=True)
    p.add_argument('--stop-before-unix', type=float, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--port', type=int, default=8775)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    if (sys.platform != 'linux' or socket.gethostname() != args.expected_hostname
            or args.output.name != 'experiment' or args.output.parent.name != args.run_id
            or subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() != args.source_commit
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip()):
        raise ValueError('Require the exact clean reviewed PPO target source in this allocated Linux worker')
    from fluffy_bassoon.gpu_execution import validate_execution
    validate_execution(os.environ.get('FLUFFY_GPU_EXECUTION', 'baige'), os.environ)
    probe = json.loads(Path(os.environ['LYCO_BAIGE_GPU_MAP']).read_text())
    if probe.get('status') != 'passed' or len(probe.get('devices', [])) != 1:
        raise ValueError('Require the actual allocated single-GPU probe')
    plan = load_plan(args.plan, args.plan_sha256, args.source_commit)
    budget = reservation(args.ledger, plan, args.worker, args.run_id, args.source_commit)
    seconds = graceful_seconds(args.output.parent, budget['record_sha256'], args.stop_before_unix)
    if seconds < 60 or file_sha256(args.checkpoint / 'READY.json') != PI05_RECEIPT:
        raise ValueError('No authorized time remains or official motor receipt differs')
    deadline = time.monotonic() + seconds
    runtime = f'{socket.gethostname()}-{os.getpid()}-{time.time_ns()}'
    attempt = args.output.parent / 'runtime-attempts' / runtime
    attempt.mkdir(parents=True)
    write_json(attempt / 'invocation-deadline.json', dict(stop_before_unix=args.stop_before_unix,
        driver_seconds=seconds, shutdown_margin_seconds=60, cumulative_worker_seconds=3600, action_cap=70000))
    schedule = load_schedule(args.schedule)
    entries = {e['case']['case_id']: e for e in plan['cases'][args.worker::2]}
    groups = [g for e in entries.values() for g in e['groups']]
    contract = dict(format=FORMAT, source_commit=args.source_commit, ppo_wave_sha256=plan['ppo_wave_sha256'],
        plan_file_sha256=args.plan_sha256, worker=args.worker, groups=groups, budget_reservation=budget,
        maximum_total_actions=70000, maximum_seconds_per_invocation=3600.,
        terminal_source_validation=plan['contract']['source_validation'], schedule_file_sha256=file_sha256(args.schedule),
        canonical_motor_key='jax.random.PRNGKey(seed), legacy uint32[2], every comparator',
        controller_rng='Qualified global Torch CPU/CUDA episode seed; generator=None; complete RNG in every context/recovery',
        deployment_optimizer_updates=0, target_data_in_source_pretraining=False, qualification_extra_resets=0,
        source_costs='Full and separately trained reset source costs retained independently and counted once; same-weight reset shares full cost')
    lock = (args.output.parent / 'evaluation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    args.model_config, args.feature_interface = 'pi05_libero', 'vla-prefix'
    motor = FrozenOpenPI(SimpleNamespace(**dict(vars(args), output=attempt)))
    selected, source = load_terminal_models(plan, motor, schedule, args.output.parent / 'models', args.device)
    provenance = dict(source_ppo=source, source_validation=plan['contract']['source_validation'],
        target_optimizer_updates=0, same_weight_reset_shares_full_source_cost=True)
    before = parameter_signature(motor.policy._model)
    write_json(attempt / 'motor-parameters-before.json', before)
    def reset_rng(seed):
        import jax
        motor.policy._rng = jax.random.PRNGKey(seed)
    def qualify(observation, server):
        zero = np.zeros((10, 32), np.float32)
        a, b = motor.infer(observation, server.metadata['instruction'], zero), motor.infer(observation, server.metadata['instruction'], zero.copy())
        if not np.allclose(a, b, rtol=0, atol=1e-5):
            raise RuntimeError('Frozen motor same-noise qualification failed')
        write_json(attempt / 'qualification.json', dict(extra_physical_resets=0, same_noise_max_delta=float(np.abs(a-b).max()),
            features=motor.feature_encoder.qualify(observation, server.metadata['instruction'])))
    result = None
    try:
        result = run_common(motor, {'source_ppo': selected}, {}, output=args.output, contract=contract,
            server_factory=lambda case: PlannedServer(case, entries[case.case_id], worktree=root,
                output=attempt / f'server-{case.task_id:02d}-{time.time_ns()}', port=args.port),
            provenance={'source_ppo': provenance, 'native': dict(source_training_actions=0, target_optimizer_updates=0)},
            controller_factory=lambda spec, server: controller_factory(spec), runtime_id=runtime, resume=args.resume,
            reset_motor_rng=reset_rng, motor_rng_state=lambda: save_motor_rng(motor.policy._rng), qualify=qualify,
            deadline=deadline, action_ceiling=140000, audit_fn=audit_ppo, strict_initial_observations=True, graceful_deadline=True)
        result.update(format=FORMAT, ppo_wave_sha256=plan['ppo_wave_sha256'], worker=args.worker,
            source_provenance=provenance, zero_shot_reuse=reuse_empty_history(plan, result['completed_groups'], selected),
            target_optimizer_updates=0, interpretation='All90 one-initialization K0 screen plus fixed9 one-support panel; frozen final source models, no target checkpoint selection')
    finally:
        after = parameter_signature(motor.policy._model)
        write_json(attempt / 'motor-parameters-after.json', dict(before=before, after=after, equal=before == after))
        if result is not None:
            result.update(motor_parameters_unchanged=before == after,
                motor_parameters_before_sha256=before['sha256'], motor_parameters_after_sha256=after['sha256'])
            result['strict_common_runtime_primary_eligible'] &= before == after
            write_json(args.output / 'index.json', result)
        if before != after:
            raise RuntimeError('Frozen motor parameters changed during PPO deployment')


if __name__ == '__main__':
    main()
