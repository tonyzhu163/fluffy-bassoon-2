"""Six bounded shared-motor workers for the predeclared source40 target screen.

Reads a separately authorized target ledger. No reservation, learner, target
checkpoint selection, or source-data write occurs here.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

from .vla_common_runtime_evaluate import run_common
from .vla_history import file_sha256
from .vla_libero import FrozenOpenPI, SimulatorClient, write_json
from .vla_motor_replay import parameter_signature
from .vla_source40_corpus import PI05_RECEIPT, save_motor_rng
from .vla_source40_evaluate import load_candidates
from .vla_source40_memory_replication import public_signature, same_public, trace_rows
from .vla_source40_ppo_target import ShiftedServer
from .vla_source40_schedule import content_hash
from .vla_source40_target_firstwave_plan import (INTERIM_INPUTS, INTERIM_PROFILE, LEGACY_PROFILE,
    load_plan as load_declared_plan, profile_settings)


FORMAT = 'source40_ad_target_firstwave_worker_v1'
LEDGER_FORMAT = 'libero_target_wave_budget_v1'


def load_plan(path, expected, source):
    return load_declared_plan(path, expected, source)


def reservation(path, plan, worker, run, source):
    value = json.loads(Path(path).read_text())
    settings = profile_settings(plan['contract'].get('profile', LEGACY_PROFILE))
    rows = list(value.get('reservations', {}).values())
    if (value.get('format') != LEDGER_FORMAT or value.get('wave_sha256') != plan['firstwave_sha256']
            or value.get('maximum_total_actions') != settings['maximum_total_actions']
            or value.get('maximum_gpu_seconds') != settings['maximum_gpu_seconds']
            or value.get('source_campaign_immutable') is not True or not value.get('scope_amendment')
            or len(rows) != 6 or {row.get('worker') for row in rows} != set(range(6))):
        raise ValueError('Require the separate authorized six-worker profile-specific target scope amendment')
    for item in rows:
        planned = plan['workers'][item['worker']]
        if (item.get('source_commit') != source or item.get('wave_sha256') != plan['firstwave_sha256']
                or item.get('maximum_total_actions') != planned['action_cap']
                or item.get('worker_seconds') != planned['gpu_seconds']
                or item.get('purpose') != 'target_evaluation'
                or item.get('source_pretraining_data') is not False
                or item.get('fresh_ppo_training_data') is not False):
            raise ValueError('Target worker reservation differs from the disjoint predeclared budget')
    if (sum(row['maximum_total_actions'] for row in rows) > settings['maximum_total_actions']
            or sum(row['worker_seconds'] for row in rows) != settings['maximum_gpu_seconds']):
        raise ValueError('Target worker reservations exceed the joint action/time scope')
    selected = value['reservations'].get(run)
    if selected is None or selected.get('run_id') != run or selected.get('worker') != worker:
        raise ValueError('Require this exact existing target-only worker reservation')
    return dict(ledger_path=str(Path(path).resolve()), record=selected, record_sha256=content_hash(selected))


def worker_groups(plan, worker):
    return [dict(group, case=entry['case'], support_budgets=[group['support_budget']])
        for entry in plan['cases'][worker::6] for group in entry['groups']]


def fixed_matrix(path, expected, assembly, assembly_sha, *, updates=500):
    if updates not in (100, 500):
        raise ValueError('Only the explicit fixed100 and legacy fixed500 profiles are supported')
    if file_sha256(path) != expected or file_sha256(assembly) != assembly_sha:
        raise ValueError(f'Source40 fixed{updates} matrix/assembly file checksum differs')
    value = json.loads(Path(path).read_text())
    rows = value.get('controllers', [])
    if (value.get('completed') is not True or len(rows) != 4
            or {(r['objective'], r['mode']) for r in rows} != {(a, b) for a in ('ad', 'dpt') for b in ('full', 'static')}
            or {r['updates'] for r in rows} != {updates} or len({r['seed'] for r in rows}) != 1
            or (updates == 100 and {r['seed'] for r in rows} != {0})):
        raise ValueError(f'Require all four completed source-only fixed{updates} controllers')
    registry = []
    for row in rows:
        for field, digest in (('final_path', 'final_sha256'), ('full_recovery_path', 'full_recovery_sha256')):
            if file_sha256(row[field]) != row[digest]:
                raise ValueError('Source-only selected controller/full recovery checksum differs')
        registry.append({k: row[k] for k in ('objective', 'mode', 'updates', 'seed', 'final_path', 'final_sha256',
            'full_recovery_path', 'full_recovery_sha256')})
    return value, registry


def bind_profile_inputs(plan, matrix, matrix_sha, assembly, assembly_sha):
    """The interim profile binds the actual complete producer, never a best epoch."""
    if plan['contract'].get('profile', LEGACY_PROFILE) == LEGACY_PROFILE:
        return
    inputs = plan['contract'].get('fixed_source_inputs')
    if (inputs != INTERIM_INPUTS or Path(matrix).resolve() != Path(inputs['matrix_path']).resolve()
            or matrix_sha != inputs['matrix_sha256'] or Path(assembly).resolve() != Path(inputs['assembly_path']).resolve()
            or assembly_sha != inputs['assembly_file_sha256']):
        raise ValueError('Fixed100 profile must retain its exact producer matrix and CPU snapshot')
    if (file_sha256(matrix) != matrix_sha or file_sha256(assembly) != assembly_sha
            or file_sha256(inputs['manifest_path']) != inputs['manifest_sha256']):
        raise ValueError('Fixed100 matrix/assembly/manifest file checksum differs')
    value = json.loads(Path(matrix).read_text())
    manifest = json.loads(Path(inputs['manifest_path']).read_text())
    if (value.get('contract', {}).get('source_commit') != inputs['producer_source_commit']
            or value['contract'].get('assembly_sha256') != inputs['assembly_content_sha256']
            or manifest.get('completed') is not True
            or manifest.get('assembly_file_sha256') != assembly_sha
            or manifest.get('assembly_sha256') != inputs['assembly_content_sha256']):
        raise ValueError('Fixed100 producer or source exposure differs')


def audit_firstwave(path, result, group):
    budgets = group['support_budgets']
    episodes = result.get('episodes', [])
    if (result.get('completed') is not True or result.get('case') != group['case']
            or result.get('weights_unchanged') is not True
            or [(r['method'], r['support_budget']) for r in result.get('trials', [])] != [(group['method'], b) for b in budgets]
            or len(episodes) != sum(b + 1 for b in budgets)):
        raise ValueError('First-wave completed method/case/support receipt differs')
    arrays = [trace_rows(path, row)[0]['motor_observation'] for row in episodes]
    if (len({row['initial_state_vector_sha256'] for row in episodes}) != 1
            or len({row['episode_random_seed'] for row in episodes if row['role'] == 'query'}) != 1):
        raise ValueError('First-wave nominal initialization or paired query RNG changed')
    contexts = Path(path) / 'contexts'
    archive = json.loads((contexts / 'index.json').read_text())
    if not archive or any(Path(row['path']).name != row['path']
            or file_sha256(contexts / row['path']) != row['sha256'] for row in archive):
        raise ValueError('First-wave full context/RNG archive checksum differs')
    if group['method'] != 'native':
        for episode in episodes:
            position = episode['support_budget'] if episode['role'] == 'query' else episode['role_episode_ordinal']
            if not any(row.get('method') == group['method'] and row.get('trial_index') == episode['trial_index']
                    and row.get('episode_position') == position
                    and row.get('episode_random_seed') == episode['episode_random_seed'] for row in archive):
                raise ValueError('Committed physical episode lacks its controller/context/RNG boundary')
    # Raw arrays are retained. A screen can report nominal-case performance with
    # camera differences; a causal memory interpretation additionally needs the
    # explicit actual-observation equality flags, never an implicit waiver.
    return dict(nominal_state_sha256=episodes[0]['initial_state_vector_sha256'],
        initial_observation_sha256=episodes[0]['initial_observation_sha256'],
        initial_public_arrays_sha256=public_signature(arrays[0]), initial_arrays=arrays[0],
        within_group_initial_public_arrays_equal=all(same_public(arrays[0], a) for a in arrays[1:]),
        within_group_initial_observation_hashes_equal=len({r['initial_observation_sha256'] for r in episodes}) == 1,
        episode_context_index_sha256=file_sha256(contexts / 'index.json'))


def zero_shot_reuse(plan, completed, candidates):
    by_group = {row['group']['group_id']: row for row in completed}
    result = []
    for case in plan['cases']:
        for reset, reference in case['zero_shot_reuse'].items():
            if reference not in by_group:
                continue
            full = reset.replace('episode_reset', 'full')
            a, b = candidates[full], candidates[reset]
            if a['policy'] is not b['policy'] or a['sha256'] != b['sha256'] or a['mode'] != 'full' or b['mode'] != 'episode':
                raise ValueError('K0 reuse requires identical full-controller weights and empty-history semantics')
            row = by_group[reference]
            value = json.loads(Path(row['result_path']).read_text())
            query = next(e for e in value['episodes'] if e['role'] == 'query')
            result.append(dict(case_id=case['case']['case_id'], method=reset, support_budget=0,
                reference_group_id=reference, reference_result_path=row['result_path'],
                reference_result_sha256=row['result_sha256'], query_episode_id=query['episode_id'],
                query_trace_sha256=query['trace_sha256'], query_success=query['success'],
                policy_sha256=a['sha256'], additional_physical_actions=0,
                proof='Same policy object/weights, empty initial history, same within-query transitions and paired query seed; static weights are never substituted'))
    return result


class PlannedServer(ShiftedServer):
    def __init__(self, case, entry, *, worktree, output, port):
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.process, self.client = None, None
        self.log = output.with_suffix('.log').open('w')
        config = entry['simulator']
        paths = config['pythonpath'].split(':')
        paths[0] = str(Path(worktree) / 'src')
        command = [config['python'], '-m', config['module'], *config['arguments'], '--output', str(output), '--port', str(port)]
        try:
            self.process = subprocess.Popen(command, env=dict(os.environ, PYTHONPATH=':'.join(paths)),
                stdout=self.log, stderr=subprocess.STDOUT)
            started = time.monotonic()
            while not (output / 'simulator-ready.json').exists():
                if self.process.poll() is not None or time.monotonic() - started > 180:
                    raise RuntimeError(f'Predeclared target simulator failed; see {self.log.name}')
                time.sleep(.1)
            self.client = SimulatorClient(port)
            self.metadata = self.client.metadata
            case.validate(self.metadata)
            if any(self.metadata.get(k) != v for k, v in config['expected_metadata'].items()):
                raise ValueError('Target simulator differs from pinned case/source/horizon declaration')
        except BaseException:
            self.close()
            raise


def remaining_seconds(root, contract_record, *, worker_seconds=3600, stop_before_unix=None):
    value = json.loads((Path(root) / 'worker-time.json').read_text())
    rows = value['attempts']
    current = rows[-1]
    if (value['worker_seconds'] != worker_seconds or current['status'] != 'started'
            or value['contract'].get('escrow_record_sha256') != contract_record):
        raise ValueError('Require the active exact cumulative profile-specific target worker clock')
    charged = sum(row.get('charged_seconds', row['reserved_seconds']) for row in rows[:-1])
    if charged + current['reserved_seconds'] > worker_seconds:
        raise ValueError('Prior/active target attempts exceed the cumulative approved worker ceiling')
    seconds = current['slice_seconds'] - (time.time() - current['started_unix']) - 60
    if stop_before_unix is not None:
        if (not math.isfinite(stop_before_unix) or not time.time() < stop_before_unix
                <= current['started_unix']+current['slice_seconds']-60):
            raise ValueError('Graceful target deadline exceeds the reserved invocation')
        seconds = min(seconds, stop_before_unix-time.time())
    return seconds


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'plan', 'ledger', 'matrix', 'assembly', 'checkpoint', 'openpi-root'):
        p.add_argument('--' + name, type=Path, required=True)
    for name in ('source-commit', 'run-id', 'expected-hostname', 'plan-sha256', 'matrix-sha256', 'assembly-sha256'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--worker', type=int, choices=range(6), required=True)
    p.add_argument('--port', type=int, default=8773)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stop-before-unix', type=float)
    args = p.parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    if (sys.platform != 'linux' or socket.gethostname() != args.expected_hostname
            or args.output.name != 'experiment' or args.output.parent.name != args.run_id
            or subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() != args.source_commit
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip()):
        raise ValueError('Require the exact clean reviewed source in this explicit allocated Linux worker')
    from fluffy_bassoon.gpu_execution import validate_execution
    validate_execution(os.environ.get('FLUFFY_GPU_EXECUTION', 'baige'), os.environ)
    probe = json.loads(Path(os.environ['LYCO_BAIGE_GPU_MAP']).read_text())
    if probe.get('status') != 'passed' or len(probe.get('devices', [])) != 1:
        raise ValueError('Require the actual allocated single-GPU execution probe')
    plan = load_plan(args.plan, args.plan_sha256, args.source_commit)
    profile = plan['contract'].get('profile', LEGACY_PROFILE)
    settings = profile_settings(profile)
    if profile == INTERIM_PROFILE and args.stop_before_unix is None:
        raise ValueError('Fixed100 readout requires an explicit graceful invocation deadline')
    budget = reservation(args.ledger, plan, args.worker, args.run_id, args.source_commit)
    bind_profile_inputs(plan, args.matrix, args.matrix_sha256, args.assembly, args.assembly_sha256)
    matrix, registry = fixed_matrix(args.matrix, args.matrix_sha256, args.assembly, args.assembly_sha256,
        updates=settings['updates'])
    seconds = remaining_seconds(args.output.parent, budget['record_sha256'],
        worker_seconds=settings['worker_seconds'], stop_before_unix=args.stop_before_unix)
    if seconds < 60 or file_sha256(args.checkpoint / 'READY.json') != PI05_RECEIPT:
        raise ValueError('No remaining authorized runtime or wrong official frozen OpenPI40 motor')
    deadline = time.monotonic() + seconds
    runtime_id = f'{socket.gethostname()}-{os.getpid()}-{time.time_ns()}'
    attempt = args.output.parent / 'runtime-attempts' / runtime_id
    attempt.mkdir(parents=True)
    entries = {entry['case']['case_id']: entry for entry in plan['cases'][args.worker::6]}
    groups = worker_groups(plan, args.worker)
    contract = dict(format=FORMAT, source_commit=args.source_commit, wave_sha256=plan['firstwave_sha256'],
        plan_file_sha256=args.plan_sha256, worker=args.worker, groups=groups, budget_reservation=budget,
        maximum_total_actions=budget['record']['maximum_total_actions'], maximum_seconds_per_invocation=float(settings['worker_seconds']),
        matrix_file_sha256=args.matrix_sha256, assembly_file_sha256=args.assembly_sha256,
        canonical_motor_key='jax.random.PRNGKey(seed), legacy uint32[2], shared by every comparator',
        target_trajectories_in_pretraining=False, deployment_optimizer_updates=0,
        reporting_scope=settings['reporting_scope'],
        initialization='Exact nominal state/scene seeds; all actual initial arrays retained and equality flags reported; differences preclude causal memory claims',
        resume='Full episode/RPC/context/cursor costs retained; fresh-process continuations are non-primary for strict common-runtime claims')
    models_field = 'fixed500_models' if profile == LEGACY_PROFILE else 'fixed100_models'
    contract[models_field] = registry
    if profile == INTERIM_PROFILE:
        contract.update(profile=profile, fixed_source_inputs=plan['contract']['fixed_source_inputs'])
    lock = (args.output.parent / 'evaluation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    args.model_config, args.feature_interface = 'pi05_libero', 'vla-prefix'
    motor = FrozenOpenPI(SimpleNamespace(**dict(vars(args), output=attempt)))
    selected, source = load_candidates(args.matrix, args.matrix_sha256, args.assembly, motor,
        seed=matrix['controllers'][0]['seed'], device=args.device, allow_small_pilot=False)
    source.update(source_training_matrix_file_sha256=args.matrix_sha256, assembly_file_sha256=args.assembly_sha256,
        wave_sha256=plan['firstwave_sha256'], final_model_registry=registry)
    before = parameter_signature(motor.policy._model)
    write_json(attempt / 'motor-parameters-before.json', before)
    def reset_rng(seed):
        import jax
        motor.policy._rng = jax.random.PRNGKey(seed)
    def qualify(observation, server):
        noise = np.zeros((10, 32), np.float32)
        a = motor.infer(observation, server.metadata['instruction'], noise)
        b = motor.infer(observation, server.metadata['instruction'], noise.copy())
        if not np.allclose(a, b, rtol=0, atol=1e-5):
            raise RuntimeError('Frozen motor same-noise qualification failed')
        write_json(attempt / 'qualification.json', dict(extra_physical_resets=0,
            same_noise_max_delta=float(np.abs(a-b).max()), features=motor.feature_encoder.qualify(observation, server.metadata['instruction'])))
    result = None
    try:
        result = run_common(motor, {'source40': selected}, {}, output=args.output, contract=contract,
            server_factory=lambda case: PlannedServer(case, entries[case.case_id], worktree=root,
                output=attempt / f'server-{len(list(attempt.glob("server-*"))):03d}-{time.time_ns()}', port=args.port),
            provenance={'source40': source, 'native': dict(steering_source_actions=0, target_training_actions=0)},
            runtime_id=runtime_id, resume=args.resume, reset_motor_rng=reset_rng,
            motor_rng_state=lambda: save_motor_rng(motor.policy._rng), qualify=qualify, deadline=deadline,
            action_ceiling=settings['maximum_total_actions'], audit_fn=audit_firstwave, strict_initial_observations=False)
        result.update(format=FORMAT, zero_shot_reuse=zero_shot_reuse(plan, result['completed_groups'], selected),
            source_provenance=source, worker=args.worker,
            wave_sha256=plan['firstwave_sha256'], no_target_optimizer=True)
        result[models_field] = registry
        result['all_actual_initial_observations_equal'] = all(
            all(row['audit'][key] for key in ('within_group_initial_public_arrays_equal',
                'within_group_initial_observation_hashes_equal', 'public_arrays_match_first_group',
                'public_hash_matches_first_group')) for row in result['completed_groups'])
        result['strict_common_runtime_primary_eligible'] &= result['all_actual_initial_observations_equal']
    finally:
        after = parameter_signature(motor.policy._model)
        write_json(attempt / 'motor-parameters-after.json', dict(before=before, after=after, equal=before == after))
        if result is not None:
            result.update(motor_parameters_unchanged=before == after,
                motor_parameters_before_sha256=before['sha256'], motor_parameters_after_sha256=after['sha256'])
            write_json(args.output / 'index.json', result)
        if before != after:
            raise RuntimeError('Frozen motor parameters changed')


if __name__ == '__main__':
    main()
