"""Execute paired all90 noise controls with one common frozen motor per worker."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

from .vla_common_runtime_evaluate import audit_control, run_common
from .vla_history import file_sha256
from .vla_libero import FrozenOpenPI, write_json
from .vla_motor_replay import parameter_signature
from .vla_source40_corpus import PI05_RECEIPT, save_motor_rng
from .vla_source40_memory_replication import trace_rows
from .vla_source40_noise_plan import FORMAT as PLAN_FORMAT, LEDGER_FORMAT, METHODS, NOISE, build_plan, content_hash
from .vla_source40_target_firstwave import PlannedServer


FORMAT = 'source40_libero90_noise_controls_worker_v1'


def load_plan(path, sha, source):
    if file_sha256(path) != sha:
        raise ValueError('Noise plan file checksum differs')
    value = json.loads(Path(path).read_text())
    if (value.get('format') != PLAN_FORMAT or value.get('contract', {}).get('source_commit') != source
            or value.get('noise_plan_sha256') != content_hash({k: v for k, v in value.items() if k != 'noise_plan_sha256'})
            or value != build_plan(worktree=value['contract']['worktree'], source_commit=source)):
        raise ValueError('Exact all90 noise method/case/recipe predeclaration differs')
    return value


def reservation(path, plan, worker, run, source):
    value = json.loads(Path(path).read_text())
    rows = value.get('reservations', {})
    if (value.get('format') != LEDGER_FORMAT or value.get('noise_plan_sha256') != plan['noise_plan_sha256']
            or value.get('maximum_total_actions') != 152000 or value.get('maximum_gpu_seconds') != 7200
            or value.get('source_campaign_immutable') is not True or not value.get('scope_amendment')
            or len(rows) != 2 or {r.get('worker') for r in rows.values()} != {0, 1}):
        raise ValueError('Require the separate authorized152k/2h noise-control scope')
    for key, row in rows.items():
        if (row.get('run_id') != key or row.get('source_commit') != source
                or row.get('noise_plan_sha256') != plan['noise_plan_sha256']
                or row.get('maximum_total_actions') != 76000 or row.get('worker_seconds') != 3600
                or row.get('purpose') != 'target_evaluation' or row.get('source_pretraining_data') is not False
                or row.get('fresh_ppo_training_data') is not False):
            raise ValueError('Noise-control worker reservation does not match the immutable diagnostic')
    selected = rows.get(run)
    if selected is None or selected.get('worker') != worker:
        raise ValueError('Require this exact existing noise-control worker reservation')
    return dict(ledger_path=str(Path(path).resolve()), record=selected, record_sha256=content_hash(selected))


def remaining_seconds(root, record_sha):
    value = json.loads((Path(root) / 'worker-time.json').read_text())
    rows, current = value['attempts'], value['attempts'][-1]
    if (value['worker_seconds'] != 3600 or current['status'] != 'started'
            or value['contract'].get('escrow_record_sha256') != record_sha
            or sum(r.get('charged_seconds', r['reserved_seconds']) for r in rows[:-1]) + current['reserved_seconds'] > 3600):
        raise ValueError('Noise control requires the original active cumulative60-minute worker clock')
    return current['slice_seconds'] - (time.time() - current['started_unix']) - 60


def graceful_seconds(root, record_sha, stop_before_unix, *, now=None):
    """Absolute slice deadline is invocation metadata, never a fresh budget."""
    now = time.time() if now is None else now
    remaining = remaining_seconds(root, record_sha)
    if (not np.isfinite(stop_before_unix) or stop_before_unix <= now
            or stop_before_unix > now + remaining - 60):
        raise ValueError('Graceful stop must precede the original outer timeout by at least60 seconds')
    return min(remaining - 60, stop_before_unix - now)


def audit_noise(path, result, group):
    identity = audit_control(path, result, group)
    if group['method'] not in METHODS or group['support_budgets'] != [0]:
        raise ValueError('Noise controls require exactly one K0 query per method')
    episode = result['episodes'][0]
    rows = trace_rows(path, episode)
    rng = np.random.default_rng(episode['episode_random_seed'])
    for row in rows:
        method = group['method']
        actual = np.asarray(row['native_initial_noise'])
        if actual.shape != (10, 32) or actual.dtype != np.float32:
            raise ValueError('Noise control must retain actual float32 native H10x32 noise')
        if method == 'native':
            expected = rng.standard_normal((10, 32)).astype(np.float32)
            if row['proposed_noise'] is not None or row['executed_noise'] is not None:
                raise ValueError('Native noise must use independent rows and no steering coordinate')
        else:
            coordinate = np.zeros(32, np.float32) if method == 'zero' else rng.standard_normal(32).astype(np.float32)
            executed = np.clip(coordinate, -1., 1.).astype(np.float32) if method == 'tied_gaussian_clipped' else coordinate
            expected = np.repeat(executed[None], 10, axis=0)
            if not np.array_equal(row['proposed_noise'], coordinate) or not np.array_equal(row['executed_noise'], executed):
                raise ValueError('Tied noise coordinate differs from exact unbounded source warmup/explicit clip/zero recipe')
        if not np.array_equal(actual, expected):
            raise ValueError('Actual motor noise differs from its exact sampler/std/tie/clipping contract')
    identity.update(noise_recipe_exact=True, noise_distribution=NOISE[group['method']], actual_chunks=len(rows))
    return identity


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'plan', 'ledger', 'checkpoint', 'openpi-root'):
        p.add_argument('--' + name, type=Path, required=True)
    for name in ('source-commit', 'run-id', 'expected-hostname', 'plan-sha256'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--worker', type=int, choices=(0, 1), required=True)
    p.add_argument('--port', type=int, default=8774)
    p.add_argument('--stop-before-unix', type=float, required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    if (sys.platform != 'linux' or socket.gethostname() != args.expected_hostname
            or args.output.name != 'experiment' or args.output.parent.name != args.run_id
            or subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() != args.source_commit
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip()):
        raise ValueError('Require the exact clean reviewed noise-control source in this allocated Linux worker')
    from fluffy_bassoon.gpu_execution import validate_execution
    validate_execution(os.environ.get('FLUFFY_GPU_EXECUTION', 'baige'), os.environ)
    probe = json.loads(Path(os.environ['LYCO_BAIGE_GPU_MAP']).read_text())
    if probe.get('status') != 'passed' or len(probe.get('devices', [])) != 1:
        raise ValueError('Require the actual allocated single-GPU probe')
    plan = load_plan(args.plan, args.plan_sha256, args.source_commit)
    budget = reservation(args.ledger, plan, args.worker, args.run_id, args.source_commit)
    seconds = graceful_seconds(args.output.parent, budget['record_sha256'], args.stop_before_unix)
    if seconds < 60 or file_sha256(args.checkpoint / 'READY.json') != PI05_RECEIPT:
        raise ValueError('No authorized time remains or official OpenPI40 motor receipt differs')
    deadline = time.monotonic() + seconds
    runtime_id = f'{socket.gethostname()}-{os.getpid()}-{time.time_ns()}'
    attempt = args.output.parent / 'runtime-attempts' / runtime_id
    attempt.mkdir(parents=True)
    write_json(attempt / 'invocation-deadline.json', dict(stop_before_unix=args.stop_before_unix,
        remaining_driver_seconds=seconds, outer_shutdown_margin_seconds=60,
        original_worker_seconds=3600, action_cap=76000))
    entries = {e['case']['case_id']: e for e in plan['cases'][args.worker::2]}
    groups = [g for e in entries.values() for g in e['groups']]
    contract = dict(format=FORMAT, source_commit=args.source_commit, noise_plan_sha256=plan['noise_plan_sha256'],
        plan_file_sha256=args.plan_sha256, worker=args.worker, groups=groups, budget_reservation=budget,
        maximum_total_actions=76000, maximum_seconds_per_invocation=3600., noise_distributions=NOISE,
        source_training_actions=0, target_optimizer_updates=0, controller_weights=None,
        canonical_motor_key='jax.random.PRNGKey(seed), legacy uint32[2] for every comparator',
        qualification_extra_physical_resets=0,
        resume='Exact full recovery/raw costs retained; fresh-process continuation is non-primary for strict common-runtime claims')
    lock = (args.output.parent / 'evaluation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    args.model_config, args.feature_interface = 'pi05_libero', 'vla-prefix'
    motor = FrozenOpenPI(SimpleNamespace(**dict(vars(args), output=attempt)))
    before = parameter_signature(motor.policy._model)
    write_json(attempt / 'motor-parameters-before.json', before)
    def reset_rng(seed):
        import jax
        motor.policy._rng = jax.random.PRNGKey(seed)
    def qualify(observation, server):
        z = np.zeros((10, 32), np.float32)
        a, b = motor.infer(observation, server.metadata['instruction'], z), motor.infer(observation, server.metadata['instruction'], z.copy())
        if not np.allclose(a, b, rtol=0, atol=1e-5):
            raise RuntimeError('Frozen motor same-noise parity failed')
        write_json(attempt / 'qualification.json', dict(extra_physical_resets=0, same_noise_max_delta=float(np.abs(a-b).max())))
    result = None
    try:
        result = run_common(motor, {}, {}, output=args.output, contract=contract,
            server_factory=lambda case: PlannedServer(case, entries[case.case_id], worktree=root,
                output=attempt / f'server-{case.task_id:02d}-{time.time_ns()}', port=args.port),
            provenance={'noise_control': dict(noise_plan_sha256=plan['noise_plan_sha256'],
                exact_source_warmup=plan['contract']['source_warmup'], source_recipe_files_sha256=plan['contract']['source_recipe_files_sha256'],
                source_training_actions=0, target_optimizer_updates=0)}, runtime_id=runtime_id, resume=args.resume,
            reset_motor_rng=reset_rng, motor_rng_state=lambda: save_motor_rng(motor.policy._rng),
            qualify=qualify, deadline=deadline, action_ceiling=152000, audit_fn=audit_noise,
            strict_initial_observations=True, graceful_deadline=True)
        result.update(format=FORMAT, noise_plan_sha256=plan['noise_plan_sha256'], worker=args.worker,
            noise_distributions=NOISE, source_training_actions=0, target_optimizer_updates=0,
            interpretation='Contemporaneous paired noise-interface diagnostic on all90; no learning or checkpoint selection')
    finally:
        after = parameter_signature(motor.policy._model)
        write_json(attempt / 'motor-parameters-after.json', dict(before=before, after=after, equal=before == after))
        if result is not None:
            result.update(motor_parameters_unchanged=before == after,
                motor_parameters_before_sha256=before['sha256'], motor_parameters_after_sha256=after['sha256'])
            result['strict_common_runtime_primary_eligible'] &= before == after
            write_json(args.output / 'index.json', result)
        if before != after:
            raise RuntimeError('Frozen motor parameters changed during unlearned controls')


if __name__ == '__main__':
    main()
