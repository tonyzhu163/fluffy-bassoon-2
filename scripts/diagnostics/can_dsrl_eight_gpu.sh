#!/usr/bin/env bash
# One exclusive eight-card Baige container; scheduler must cap this retry at58min.
# Prior failed preflight reserves2min*8GPUs; (2+58)*8/60 =8GPU-hours total.
# Prefer four paired seeds with serial arms; fall back to two paired seeds with
# one independent arm per GPU when required by the measured one-hour budget.
# Never launch before the complete 4090 pilot passes the gates below.
set -euo pipefail
TASK_SOURCE=${1:?exact source commit}
TASK_RUN=${2:?fresh eight-GPU run ID}
TASK_PILOT_RUN=${3:?completed 4090 pilot run ID}
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_WORKTREE=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUTPUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_PYTHON=$TASK_ROOT/envs/dsrl-can-20261029-r2/bin/python
TASK_DSRL=$TASK_ROOT/deps/dsrl-can-20261029-r1/repo
TASK_ROBOMIMIC=$TASK_ROOT/deps/robomimic-compat-20261029-r1/repo
TASK_INPUT=$TASK_ROOT/inputs/dsrl-can-8d21b9c-20261029-r1
TASK_ALLOCATION_STARTED=$(date +%s)
[[ $TASK_SOURCE =~ ^[0-9a-f]{40}$ ]]
[[ $TASK_RUN =~ ^robomimic-can-dsrl-baige-20261029-eight-r[0-9]+$ ]]
[[ $TASK_PILOT_RUN =~ ^robomimic-can-dsrl-baige-20261029-pilot-r[0-9]+$ ]]
[[ $(hostname) =~ ^job-[a-z0-9]+-master-0$ ]]
test -n "${KUBERNETES_SERVICE_HOST:-}"
test -n "${NVIDIA_VISIBLE_DEVICES:-}"
cd "$TASK_WORKTREE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
mkdir "$TASK_OUTPUT"
exec > >(tee "$TASK_OUTPUT/console.log") 2>&1
trap 'TASK_EXIT=$?; printf "%s\n" "$TASK_EXIT" > "$TASK_OUTPUT/exit-code.txt"' EXIT
export PYTHONPATH="$TASK_WORKTREE/src:$TASK_DSRL:$TASK_DSRL/dppo:$TASK_DSRL/stable-baselines3"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 FLUFFY_GPU_EXECUTION=baige
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl WANDB_MODE=disabled NCCL_P2P_DISABLE=1
export TASK_SOURCE TASK_RUN TASK_PILOT_RUN TASK_ROOT TASK_WORKTREE TASK_OUTPUT TASK_DSRL TASK_ROBOMIMIC TASK_INPUT TASK_ALLOCATION_STARTED
"$TASK_PYTHON" - <<'PY'
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from importlib import metadata
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import time

root, output, worktree = (Path(os.environ[name]) for name in ('TASK_ROOT', 'TASK_OUTPUT', 'TASK_WORKTREE'))
pilot = root / 'outputs' / os.environ['TASK_PILOT_RUN']
source, started = os.environ['TASK_SOURCE'], float(os.environ['TASK_ALLOCATION_STARTED'])
METHODS = ['dsrl-na', 'dsrl-na-retry', 'rl2', 'episode-reset']
PAIRS = [['dsrl-na', 'rl2'], ['dsrl-na-retry', 'episode-reset']]
HELPER = 'src/fluffy_bassoon/meta_rl/can_dsrl_gpu.py'
PROBE_SCRIPT = 'scripts/[train]gpu_probe.py'
WALL, FINAL, PROBE, TRANSFER = 3480, 180, 180, 2.0
lock, records = threading.Lock(), []
plan = dict(source_commit=source, run_id=os.environ['TASK_RUN'], pilot=str(pilot),
            hostname=socket.gethostname(), admitted=False, wall_budget_seconds=WALL,
            allocated_gpu_hours_ceiling=8, final_audit_reserve_seconds=FINAL,
            required_scheduler_wall_minutes=58, prior_failed_allocation_reserve_seconds=120,
            allocation_budget=dict(prior_reserved_gpu_minutes=16,new_gpu_minutes_ceiling=464,
                                   combined_gpu_minutes_ceiling=480,combined_gpu_hours_ceiling=8),
            full_probe_reserve_seconds=PROBE, rtx6000_timing_factor=TRANSFER,
            methods=METHODS, seeds=None, selected_seed_count=None,
            envs=4, attempts=4, eval_trials=8, minimum_verified_target_steps=28800,
            predeclared_designs=[dict(seed_count=4, required_arms=16, layout='balanced serial pairs'),
                                 dict(seed_count=2, required_arms=8, layout='one independent arm per GPU')],
            selection_rule='prefer four matched seeds if feasible; otherwise two matched seeds with every method and control',
            core=dict(core='gtrxl', width=128, layers=2, heads=4, memory_length=300))


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    pending = path.with_suffix(path.suffix + '.tmp')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    pending.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def event(stage, **details):
    row = dict(stage=stage, time=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.time() - started, **details)
    with lock:
        with (output / 'stages.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps(row, allow_nan=False), flush=True)


def positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Pilot timing must be finite and positive')
    return value


def approved_baige_sku(devices):
    """Use the user-approved raw SKU without guessing its marketing model."""
    names = [row['cuda_name'] for row in devices]
    if len(names) != 8 or len(set(names)) != 1:
        raise ValueError('Require eight cards with the same approved raw Baige SKU')
    name = names[0]
    if name == 'NVIDIA RPBZZZ6':
        return dict(raw_name=name, raw_names=names, queue='user-approved Pro6K queue',
                    identity_basis='exact raw SKU in the user-provided Baige screenshot and explicit eight-card approval',
                    marketing_model_decoded=False)
    if re.search(r'RTX\s*(?:PRO\s*)?6000', name, re.I):
        return dict(raw_name=name, raw_names=names, queue='user-approved Pro6K queue',
                    identity_basis='existing RTX6000-family CUDA/NVML name rule',
                    marketing_model_decoded=False)
    raise ValueError(f'Unapproved raw GPU SKU: {name}')


def runtime_probe_migration(pilot_source, qualified_source):
    """Allow only the explicit Baige admission change, protecting every other probe helper."""
    previous = subprocess.check_output(['git', 'show', f'{qualified_source}:{PROBE_SCRIPT}'], cwd=worktree)
    trained = subprocess.check_output(['git', 'show', f'{pilot_source}:{PROBE_SCRIPT}'], cwd=worktree)
    current = (worktree / PROBE_SCRIPT).read_bytes()
    if previous != trained:
        raise ValueError('Qualification and pilot used different allocation probes')
    if hashlib.sha256(previous).hexdigest() != 'b1db1d3e306cfb7e637f0d5d5befab7b54339014f755fe4c4793d39caaca5c2e':
        raise ValueError('Unreviewed original allocation-probe baseline')
    old_tree, new_tree = ast.parse(previous), ast.parse(current)
    allowed = {'run_baige_probe', 'main'}
    protected_old = ast.Module(body=[node for node in old_tree.body
        if not (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in allowed)], type_ignores=[])
    protected_new = ast.Module(body=[node for node in new_tree.body
        if not (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in allowed)], type_ignores=[])
    old_dump, new_dump = ast.dump(protected_old), ast.dump(protected_new)
    if old_dump != new_dump:
        raise ValueError('Probe migration changed protected Slurm/direct/shared probe code')
    if previous == current:
        raise ValueError('Expected an explicit reviewed Baige training-admission migration')
    return dict(qualified_source_commit=qualified_source, pilot_source_commit=pilot_source,
                path=PROBE_SCRIPT, old_sha256=hashlib.sha256(previous).hexdigest(),
                new_sha256=hashlib.sha256(current).hexdigest(),
                permitted_changed_functions=sorted(allowed),
                protected_probe_ast_sha256=hashlib.sha256(new_dump.encode()).hexdigest(),
                science_changed=False, purpose='training', opt_in='--baige-exclusive-training',
                rule='exclusive scheduler UUID+PCI+CUDA+ownPID+memory; utilization is retained telemetry',
                cpu_fixtures='pending', full_eight_gpu_cuda_probe='pending')


def recovery(directory, method):
    if method.startswith('dsrl-'):
        manifest = read(directory / 'recovery-manifest.json')
        payload = (directory / manifest['directory']).resolve()
        if not payload.is_relative_to(directory.resolve()):
            raise ValueError('Recovery generation escaped its run directory')
        if set(manifest['files']) != {'recovery.zip', 'replay.pkl', 'runtime.pt'}:
            raise ValueError('Incomplete DSRL recovery generation')
        for name, expected in manifest['files'].items():
            if digest(payload / name) != expected:
                raise ValueError('DSRL recovery payload hash differs')
        return dict(manifest_sha256=digest(directory / 'recovery-manifest.json'), **manifest)
    return dict(recovery_sha256=digest(directory / 'recovery.pt'))


def evaluation(directory, method):
    data, paired = read(directory / 'evaluation.json'), {}
    conditions = ['frozen'] if method.startswith('dsrl-') else ['normal', 'memory_reset', 'reward_blind']
    for condition in conditions:
        rows = data[condition]
        if len(rows) != 8:
            raise ValueError('All eight held-out scenes are required')
        for index, row in enumerate(rows):
            scene = 2_000_000 + index
            if (row['scene_seeds'] != [scene] or row['attempts'] != 4 or row['primitive_steps'] != 1200
                    or len(row['successes']) != 1 or len(row['successes'][0]) != 4
                    or len(row['release_logged_successes']) != 1 or len(row['release_logged_successes'][0]) != 4):
                raise ValueError('Incomplete paired retry evaluation or success metrics')
            if not method.startswith('dsrl-'):
                expected_reset = method == 'episode-reset' or condition == 'memory_reset'
                if row.get('reset_each_attempt') is not expected_reset:
                    raise ValueError('Evaluation changed the declared episode memory reset rule')
            initial = row['initial_observation_sha256']
            if len(initial) != 4 or any(item != initial[0] for item in initial):
                raise ValueError('Retry evaluation changed its initial scene')
            if str(scene) in paired and paired[str(scene)] != initial[0]:
                raise ValueError('Evaluation controls changed the paired scene')
            paired[str(scene)] = initial[0]
    return paired


def phase_estimate(method, target, measured, qualified):
    # Both the real complete pilot phase and qualification micro-timings bound
    # each phase. A smaller candidate never discounts fixed evaluation costs.
    episode = positive(qualified['native_elapsed_seconds']) / 32
    update = positive(qualified['released_batch_update_seconds'])
    if method.startswith('dsrl-'):
        train = target / 300 * episode + (target - 24016) / 16 * update
        evaluate = 32 * episode
        boundary = 4 * episode + 75 * update
    else:
        train = target / 4800 * positive(qualified['ppo_full_trial_total_seconds'])
        # The B1K2 proof measures serial evaluator episodes. Dividing the B4K4
        # rollout by four would count a complete four-attempt lane as one episode.
        one_episode = max(episode, positive(qualified['ppo_small_trial_seconds']) / 2)
        evaluate = 96 * one_episode
        boundary = positive(qualified['ppo_full_trial_total_seconds'])
    deadline = math.ceil(TRANSFER * train + 60)
    total = max(math.ceil(TRANSFER * (positive(measured) + 60)),
                math.ceil(deadline + TRANSFER * (evaluate + boundary) + 60))
    return dict(training_deadline_seconds=deadline, total_seconds=total,
                evaluation_reserve_seconds=math.ceil(TRANSFER * evaluate),
                boundary_reserve_seconds=math.ceil(TRANSFER * boundary),
                measured_complete_4090_phase_seconds=measured)


def helper_migration(old_plan, pilot_source, qualified_source):
    """Validate the recorded count2-only runtime migration without reusing it blindly."""
    migration = old_plan.get('helper_migration')
    if not migration:
        return None
    if (migration.get('qualified_source_commit') != qualified_source
            or migration.get('permitted_changed_source') != [HELPER]
            or migration.get('cpu_tests', {}).get('passed') is not True):
        raise ValueError('The helper-only compatibility migration is incomplete')
    changes = subprocess.check_output(
        ['git', 'diff', '--name-only', qualified_source, pilot_source, '--', 'src'],
        cwd=worktree, text=True).splitlines()
    if set(changes) != {HELPER}:
        raise ValueError('The two-GPU compatibility migration changed scientific source')
    previous = subprocess.check_output(['git', 'show', f'{qualified_source}:{HELPER}'], cwd=worktree)
    pilot_helper = subprocess.check_output(['git', 'show', f'{pilot_source}:{HELPER}'], cwd=worktree)
    current = digest(worktree / HELPER)
    if (hashlib.sha256(previous).hexdigest() != migration['old_sha256']
            or hashlib.sha256(pilot_helper).hexdigest() != migration['new_sha256']
            or current != migration['new_sha256']):
        raise ValueError('GPU helper source does not match the documented migration')
    tests = migration['cpu_tests']
    if (digest(pilot / 'runtime-tests.log') != tests['log_sha256']
            or digest(worktree / 'tests/test_can_dsrl_gpu.py') != tests['source_sha256']):
        raise ValueError('The allocation fixture proof or its source changed')
    scientific = old_plan.get('scientific_source_sha256', {})
    if not scientific or any(digest(worktree / name) != expected for name, expected in scientific.items()):
        raise ValueError('The byte-identical scientific source manifest changed after the pilot')
    parent = read(pilot / 'parent-gpu-probe.json')
    parent_sha = digest(pilot / 'parent-gpu-probe.json')
    assigned = parent.get('gpu_identity_mapping', {}).get('assigned_nvml_uuids', [])
    if (parent_sha != old_plan['parent_probe_sha256'] or parent.get('status') != 'passed'
            or parent.get('purpose') != 'training' or parent.get('allocation_type') != 'baige_container'
            or parent.get('source_commit') != pilot_source
            or parent.get('scientific_source_commit') != qualified_source
            or parent.get('helper_sha256') != current or len(parent.get('devices', [])) != 2
            or len(assigned) != 2 or set(assigned) != {row['uuid'] for row in parent['devices']}):
        raise ValueError('The migration lacks its complete two-GPU parent qualification')
    receipts = migration.get('actual_selected_gpu_probes')
    if not isinstance(receipts, dict) or set(receipts) != set(METHODS):
        raise ValueError('All four actual selected-GPU worker receipts are required')
    for method in METHODS:
        path = pilot / method / 'gpu-probe.json'
        receipt = read(path)
        selected = receipt.get('selected_device', {})
        ordinal = int(receipt.get('cuda_visible_devices', '-1'))
        if (digest(path) != receipts[method] or receipt.get('status') != 'passed'
                or receipt.get('source_commit') != pilot_source or receipt.get('parent_gpu_count') != 2
                or receipt.get('parent_probe_sha256') != parent_sha
                or receipt.get('source_binding') != 'parent_and_worker'
                or receipt.get('hostname') != parent['hostname']
                or receipt.get('scheduler_nvidia_visible_devices') != assigned
                or ordinal not in (0, 1) or selected.get('uuid') != parent['devices'][ordinal]['uuid']
                or receipt.get('tensor_result') != 8192. or receipt.get('elements') != 4096
                or receipt.get('pid_allocation', {}).get('available') is not True
                or receipt['pid_allocation'].get('own_gpu_uuids') != [selected['uuid']]):
            raise ValueError(f'Incomplete actual worker GPU migration evidence: {method}')
    return dict(verified=True, qualified_source_commit=qualified_source, pilot_source_commit=pilot_source,
                old_helper_sha256=migration['old_sha256'], helper_sha256=current,
                cpu_tests=tests, parent_probe_sha256=parent_sha, worker_receipts=receipts)


def admit():
    admission, old_plan = read(pilot / 'pilot-admission.json'), read(pilot / 'plan.json')
    if (admission.get('core_training_interactions_complete') is not True
            or old_plan.get('scene_pairing_verified') is not True
            or old_plan.get('admitted') is not True):
        raise ValueError('All four paired 4090 pilot arms must have completed')
    pilot_source, pilot_target = admission['source_commit'], int(admission['target_steps'])
    if old_plan['source_commit'] != pilot_source or old_plan['target_steps'] != pilot_target:
        raise ValueError('Pilot admission and predeclared plan disagree')
    qual = Path(old_plan['qualification'])
    for name, expected in old_plan['qualification_artifact_sha256'].items():
        if digest(qual / name) != expected:
            raise ValueError('Qualification evidence changed after the pilot admission')
    q, native = read(qual / 'qualification/summary.json'), read(qual / 'native/summary.json')
    native_rows = read(qual / 'native/evaluation.json')['native']
    outcomes = [value for row in native_rows for trial in row['successes'] for value in trial]
    if (q.get('passed') is not True or q['environment_parity'].get('passed') is not True
            or q['ppo_recovery_roundtrip'].get('passed') is not True
            or q.get('dsrl_recovery_roundtrip') is not True or native.get('completed') is not True
            or len(outcomes) != 32 or sum(outcomes) / len(outcomes) >= .95):
        raise ValueError('Qualified simulator/recovery/complete native headroom evidence required')
    if q['ppo_full_trial_geometry'] != dict(lanes=4, attempts=4, chunks=300, epochs=4, minibatch_trials=4):
        raise ValueError('Pilot qualification did not measure the declared GTrXL update geometry')
    checked = ['src/fluffy_bassoon/meta_rl/' + name for name in
               ('can_dsrl.py', 'can_dsrl_run.py', 'can_dsrl_compare.py', 'can_dsrl_parity.py', 'can_dsrl_gpu.py', 'ppo.py', 'gtrxl.py')]
    checked += ['src/fluffy_bassoon/baige_gpu.py', 'src/fluffy_bassoon/gpu_execution.py']
    for name in checked:
        previous = subprocess.check_output(['git', 'show', f'{pilot_source}:{name}'], cwd=worktree)
        if hashlib.sha256(previous).hexdigest() != digest(worktree / name):
            raise ValueError(f'Runtime source changed after pilot: {name}')
    stage = {row['method']: row for row in admission['stage_records'] if row['method'] in METHODS}
    baseline, scene_hashes, old_recoveries = None, None, {}
    for method in METHODS:
        if stage.get(method, {}).get('completed') is not True:
            raise ValueError('A required pilot arm is missing or incomplete')
        directory = pilot / method
        summary, contract = read(directory / 'summary.json'), read(directory / 'contract.json')
        if summary.get('completed') is not True or summary['primitive_steps'] != pilot_target:
            raise ValueError('Pilot arms did not meet one common training target')
        if contract['source_commit'] != pilot_source or contract['seed'] != 1:
            raise ValueError('Pilot run contract does not match its admission')
        if any(contract[key] != value for key, value in plan['core'].items()):
            raise ValueError('Pilot must use the predeclared GTrXL architecture')
        identity = {key: contract[key] for key in ('runtime', 'robomimic_commit', 'robomimic_env_sha256',
                    'checkpoint_sha256', 'normalization_sha256', 'frozen_motor_sha256')}
        if baseline is not None and baseline != identity:
            raise ValueError('Pilot arms used different frozen inputs or runtime identities')
        baseline = identity
        probe = read(directory / 'gpu-probe.json')
        devices = probe.get('devices', [probe.get('selected_device', {})])
        if len(devices) != 1 or re.search(r'RTX\s*4090', devices[0].get('cuda_name', ''), re.I) is None:
            raise ValueError('Each measured pilot phase must use one selected actual4090')
        paired = evaluation(directory, method)
        if scene_hashes is not None and paired != scene_hashes:
            raise ValueError('Pilot methods did not evaluate the same scenes')
        scene_hashes, old_recoveries[method] = paired, recovery(directory, method)
    qualified_contract = read(qual / 'qualification/contract.json')
    migration_evidence = helper_migration(old_plan, pilot_source, qualified_contract['source_commit'])
    probe_migration = runtime_probe_migration(pilot_source, qualified_contract['source_commit'])
    scientific_changes = subprocess.check_output(
        ['git', 'diff', '--name-only', qualified_contract['source_commit'], source, '--', 'src'],
        cwd=worktree, text=True).splitlines()
    if set(scientific_changes) - ({HELPER} if migration_evidence else set()):
        raise ValueError('Scientific source changed after qualification')
    for directory in (qual / 'qualification', qual / 'native'):
        contract = read(directory / 'contract.json')
        if any(contract[key] != value for key, value in baseline.items()):
            raise ValueError('Qualification and pilot used different runtime or frozen motor identities')
        if any(contract[key] != value for key, value in plan['core'].items()):
            raise ValueError('Qualification architecture differs from the pilot architecture')
        for name in checked:
            qualified_source = subprocess.check_output(
                ['git', 'show', f'{contract["source_commit"]}:{name}'], cwd=worktree)
            if hashlib.sha256(qualified_source).hexdigest() != digest(worktree / name):
                if name == HELPER and migration_evidence:
                    continue
                raise ValueError(f'Runtime source changed after qualification: {name}')
    current_runtime = {name: metadata.version(name) for name in baseline['runtime']}
    installed = metadata.distribution('robomimic').locate_file('robomimic/envs/env_robosuite.py')
    compatibility_source = Path(os.environ['TASK_ROBOMIMIC'])
    if (current_runtime != baseline['runtime']
            or digest(installed) != baseline['robomimic_env_sha256']
            or digest(compatibility_source / 'robomimic/envs/env_robosuite.py') != baseline['robomimic_env_sha256']
            or subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=compatibility_source, text=True).strip()
               != baseline['robomimic_commit']
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=compatibility_source, text=True).strip()):
        raise ValueError('Current runtime differs from the qualified compatibility source')
    for name, expected in (('state_5000.pt', baseline['checkpoint_sha256']),
                           ('normalization.npz', baseline['normalization_sha256'])):
        if digest(Path(os.environ['TASK_INPUT']) / name) != expected:
            raise ValueError('Current frozen input artifact differs from the pilot')
    qualified = dict(q, native_elapsed_seconds=native['elapsed_seconds'])
    candidates = []
    remaining = WALL - (time.time() - started) - PROBE - FINAL
    for seed_count in (4, 2):
        if seed_count == 4:
            assignment = [dict(gpu=2*i+j, seed=i+1, methods=PAIRS[j])
                          for i in range(4) for j in range(2)]
            layout = 'balanced serial pairs'
        else:
            assignment = [dict(gpu=4*i+j, seed=i+1, methods=[method])
                          for i in range(2) for j, method in enumerate(METHODS)]
            layout = 'one independent arm per GPU'
        for target in (48000, 28800):
            if target > pilot_target or target < plan['minimum_verified_target_steps']:
                continue
            phases = {method: phase_estimate(method, target, stage[method]['phase_elapsed_seconds'], qualified)
                      for method in METHODS}
            longest = max(sum(phases[method]['total_seconds'] for method in lane['methods'])
                          for lane in assignment)
            fits = longest <= remaining
            candidates.append(dict(seed_count=seed_count, seeds=list(range(1, seed_count+1)),
                                   required_arms=seed_count*len(METHODS), layout=layout,
                                   gpu_assignment=assignment, target_steps=target, phases=phases,
                                   longest_worker_seconds=longest, available_worker_seconds=remaining,
                                   fits=fits, admission_reason='fits measured one-hour envelope' if fits else
                                       'complete worker workload plus reserves exceeds one-hour envelope'))
    plan.update(candidates=candidates, candidate_available_worker_seconds=remaining)
    chosen = next((candidate for candidate in candidates if candidate['fits']), None)
    if chosen is None:
        raise ValueError('Neither predeclared seed geometry fits at the full verified target and controls')
    selection_reason = ('Four matched seeds fit with every required method and control'
                        if chosen['seed_count'] == 4 else
                        'Four-seed serial lanes do not fit; two matched seeds preserve the full target, '
                        'DSRL retry and trained-reset controls within one hour')
    plan.update(pilot_source_commit=pilot_source, runtime_identity=baseline, paired_scenes=scene_hashes,
                pilot_recoveries=old_recoveries, candidates=candidates, estimate=chosen,
                helper_migration_verification=migration_evidence,
                runtime_probe_migration=probe_migration,
                selected_seed_count=chosen['seed_count'], seeds=chosen['seeds'],
                selected_layout=chosen['layout'], selection_reason=selection_reason,
                gpu_assignment=chosen['gpu_assignment'],
                target_steps=chosen['target_steps'], pilot_artifact_sha256={name: digest(pilot / name)
                    for name in ('pilot-admission.json', 'plan.json')},
                source_files_sha256={name: digest(worktree / name) for name in checked + [PROBE_SCRIPT]})
    return chosen


def worker(ordinal, seed, methods, chosen, parent_probe):
    worker_records = []
    for index, method in enumerate(methods):
        budgets = chosen['phases']
        required = sum(budgets[name]['total_seconds'] for name in methods[index:]) + FINAL
        if WALL - (time.time() - started) < required:
            event('phase_refused', gpu=ordinal, seed=seed, method=method, reason='complete remaining phases no longer fit')
            break
        directory = output / f'seed{seed}' / method
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(ordinal), LYCO_BAIGE_GPU_MAP=str(parent_probe),
                   NUMBA_CACHE_DIR=str(root / 'cache' / f'{os.environ["TASK_RUN"]}-gpu{ordinal}'))
        env.pop('LYCO_BAIGE_ISOLATED_UUID', None)
        Path(env['NUMBA_CACHE_DIR']).mkdir(parents=True, exist_ok=False) if index == 0 else None
        command = [sys.executable, '-m', 'fluffy_bassoon.meta_rl.can_dsrl_run', '--method', method,
                   '--dsrl-root', os.environ['TASK_DSRL'], '--robomimic-root', os.environ['TASK_ROBOMIMIC'],
                   '--checkpoint', str(Path(os.environ['TASK_INPUT']) / 'state_5000.pt'),
                   '--normalization', str(Path(os.environ['TASK_INPUT']) / 'normalization.npz'),
                   '--source-commit', source, '--gpu-attestation', str(parent_probe),
                   '--checkpoint-sha256', plan['runtime_identity']['checkpoint_sha256'],
                   '--normalization-sha256', plan['runtime_identity']['normalization_sha256'],
                   '--output', str(directory), '--seed', str(seed), '--envs', '4', '--attempts', '4',
                   '--core', 'gtrxl', '--width', '128', '--layers', '2', '--heads', '4', '--memory-length', '300',
                   '--steps', str(chosen['target_steps']), '--eval-trials', '8',
                   '--max-seconds', str(budgets[method]['training_deadline_seconds'])]
        directory.parent.mkdir(parents=True, exist_ok=True)
        event('phase_started', gpu=ordinal, seed=seed, method=method, command=command)
        phase_start = time.monotonic()
        # No timeout/signal is sent to a model process. The CLI stops at a
        # complete resumable boundary; the scheduler owns its58-minute cap.
        with (directory.parent / f'{method}.console.log').open('x') as log:
            process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        record = dict(gpu=ordinal, seed=seed, method=method, returncode=process.returncode,
                      phase_elapsed_seconds=time.monotonic() - phase_start, completed=False)
        try:
            summary, contract = read(directory / 'summary.json'), read(directory / 'contract.json')
            if process.returncode or summary.get('completed') is not True or summary['primitive_steps'] != chosen['target_steps']:
                raise ValueError('Phase failed or stopped below the common target')
            if any(contract[key] != value for key, value in plan['runtime_identity'].items()):
                raise ValueError('Phase changed qualified runtime or frozen motor identity')
            worker_probe = read(directory / 'gpu-probe.json')
            if (worker_probe.get('status') != 'passed' or worker_probe.get('parent_gpu_count') != 8
                    or worker_probe.get('source_commit') != source
                    or worker_probe.get('parent_probe_sha256') != plan['parent_probe_sha256']
                    or worker_probe.get('cuda_visible_devices') != str(ordinal)
                    or worker_probe.get('pid_allocation', {}).get('available') is not True):
                raise ValueError('Worker lost its complete parent allocation and selected-GPU proof')
            record.update(completed=True, summary=summary, paired_scenes=evaluation(directory, method),
                          recovery=recovery(directory, method), contract_sha256=digest(directory / 'contract.json'),
                          evaluation_sha256=digest(directory / 'evaluation.json'),
                          gpu_probe_sha256=digest(directory / 'gpu-probe.json'))
            if record['paired_scenes'] != plan['paired_scenes']:
                raise ValueError('Phase changed the paired held-out scenes')
        except Exception as error:
            record.update(completed=False, error=f'{type(error).__name__}: {error}')
        worker_records.append(record)
        with lock:
            records.append(record)
            write(output / 'phase-records.json', records)
        event('phase_ended', **record)
        if not record['completed']:
            break
    return worker_records


exit_code, reason = 0, None
try:
    chosen = admit()
    write(output / 'plan.json', plan)
    runtime_started = time.monotonic()
    fixture_paths = ['tests/test_gpu_probe.py', 'tests/test_gpu_identity_mapping.py', 'tests/test_can_dsrl_gpu.py']
    fixture_env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    with (output / 'runtime-probe-tests.log').open('x') as log:
        subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *fixture_paths],
                       env=fixture_env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60)
    plan['runtime_probe_migration']['cpu_fixtures'] = dict(passed=True,
        log_sha256=digest(output / 'runtime-probe-tests.log'),
        test_source_sha256={name: digest(worktree / name) for name in fixture_paths},
        timeout_seconds=60, elapsed_seconds=time.monotonic()-runtime_started)
    write(output / 'plan.json', plan)
    remaining_probe = PROBE - (time.monotonic()-runtime_started)
    if remaining_probe <= 0:
        raise ValueError('CPU fixtures exhausted the combined runtime qualification reserve')
    probe_path = output / 'parent-gpu-probe.json'
    subprocess.run([sys.executable, str(worktree / 'scripts/[train]gpu_probe.py'), '--expected-count', '8',
                    '--output', str(probe_path), '--purpose', 'training', '--baige-container',
                    '--baige-exclusive-training'], check=True, timeout=remaining_probe)
    probe = read(probe_path)
    if (probe.get('status') != 'passed' or probe.get('purpose') != 'training'
            or probe.get('exclusive_training_admission') is not True
            or probe.get('admission_policy') != 'exclusive_scheduler_cuda_pci_pid_memory'
            or probe.get('utilization_role') != 'telemetry_only'):
        raise ValueError('Full eight-GPU training admission did not use the reviewed explicit policy')
    resource_identity = approved_baige_sku(probe['devices'])
    assigned = os.environ['NVIDIA_VISIBLE_DEVICES'].split(',')
    if len(assigned) != 8 or set(assigned) != {row['uuid'] for row in probe['devices']}:
        raise ValueError('Complete scheduler NVIDIA allocation changed during parent probe')
    probe.update(source_commit=source, source_files_sha256=plan['source_files_sha256'],
                 approved_resource_identity=resource_identity,
                 runtime_probe_migration={key: value for key,value in plan['runtime_probe_migration'].items()
                                          if key != 'full_eight_gpu_cuda_probe'})
    write(probe_path, probe)
    if WALL - (time.time() - started) < chosen['longest_worker_seconds'] + FINAL:
        raise ValueError('Startup/probe consumed the remaining full-campaign reserve')
    parent_sha256 = digest(probe_path)
    plan['runtime_probe_migration']['full_eight_gpu_cuda_probe'] = dict(passed=True,
        parent_probe_sha256=parent_sha256, combined_fixture_and_probe_seconds=time.monotonic()-runtime_started)
    plan.update(admitted=True, parent_probe_sha256=parent_sha256, resource_identity=resource_identity)
    write(output / 'plan.json', plan)
    event('campaign_admitted', target_steps=chosen['target_steps'], seed_count=chosen['seed_count'],
          required_arms=chosen['required_arms'], layout=chosen['layout'],
          selection_reason=plan['selection_reason'], gpu_assignment=plan['gpu_assignment'],
          raw_gpu_sku=resource_identity['raw_name'])
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(worker, row['gpu'], row['seed'], row['methods'], chosen, probe_path)
                   for row in plan['gpu_assignment']]
        for future in futures:
            future.result()
except Exception as error:
    reason, exit_code = f'{type(error).__name__}: {error}', 2
    event('campaign_refused_or_failed', reason=reason)
finally:
    expected = {(seed, method) for seed in (plan.get('seeds') or []) for method in METHODS}
    completed = {(row['seed'], row['method']) for row in records if row['completed']}
    matched = plan['admitted'] and completed == expected
    final = dict(source_commit=source, admitted=plan['admitted'], matched_training_interactions=matched,
                 target_steps=plan.get('target_steps'), selected_seed_count=plan.get('selected_seed_count'),
                 selected_layout=plan.get('selected_layout'), selection_reason=plan.get('selection_reason'),
                 completed_arms=len(completed), required_arms=len(expected) if expected else None,
                 incomplete_arms=[dict(seed=seed, method=method) for seed, method in sorted(expected-completed)],
                 stage_records=records, elapsed_seconds=time.time()-started, wall_budget_seconds=WALL,
                 allocation_gpu_hours=(time.time()-started)*8/3600, reason=reason,
                 allocation_budget=plan['allocation_budget'],
                 combined_gpu_hours_with_prior_reserve=((time.time()-started)+120)*8/3600,
                 resource_identity=plan.get('resource_identity'),
                 runtime_probe_migration=plan.get('runtime_probe_migration'),
                 comparison=f"{plan['selected_seed_count']} matched training seeds" if matched else
                     ('incomplete; no matched comparison' if expected else 'not admitted; no geometry selected'))
    write(output / 'plan.json', plan)
    write(output / 'campaign-admission.json', final)
    event('campaign_ended', matched_training_interactions=matched, completed_arms=len(completed),
          selected_seed_count=plan.get('selected_seed_count'), required_arms=final['required_arms'], reason=reason)
    if not matched:
        exit_code = exit_code or 3
raise SystemExit(exit_code)
PY
