#!/usr/bin/env bash
# Initial small test: one 4090 or RTX 6000 GPU, at most 90 minutes after qualification.
set -euo pipefail
TASK_SOURCE=${1:?exact source commit}
TASK_RUN=${2:?fresh pilot run ID}
TASK_QUAL_RUN=${3:?completed qualification run ID}
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_WORKTREE=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUTPUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_PYTHON=$TASK_ROOT/envs/dsrl-can-20261029-r2/bin/python
TASK_DSRL=$TASK_ROOT/deps/dsrl-can-20261029-r1/repo
TASK_ROBOMIMIC=$TASK_ROOT/deps/robomimic-compat-20261029-r1/repo
TASK_INPUT=$TASK_ROOT/inputs/dsrl-can-8d21b9c-20261029-r1
[[ $TASK_SOURCE =~ ^[0-9a-f]{40}$ ]]
[[ $TASK_RUN =~ ^robomimic-can-dsrl-baige-20261029-pilot-r[0-9]+$ ]]
[[ $TASK_QUAL_RUN =~ ^robomimic-can-dsrl-baige-20261029-qual-r[0-9]+$ ]]
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
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export FLUFFY_GPU_EXECUTION=baige CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl WANDB_MODE=disabled
export NUMBA_CACHE_DIR="$TASK_ROOT/cache/dsrl-can-numba-r1"
export TASK_SOURCE TASK_RUN TASK_QUAL_RUN TASK_ROOT TASK_OUTPUT TASK_WORKTREE TASK_DSRL TASK_ROBOMIMIC TASK_INPUT
mkdir -p "$NUMBA_CACHE_DIR"
"$TASK_PYTHON" - <<'PY'
"""Plan from qualification timing only, then invoke the existing pinned CLI."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

started = time.monotonic()
output = Path(os.environ['TASK_OUTPUT'])
worktree = Path(os.environ['TASK_WORKTREE'])
qual = Path(os.environ['TASK_ROOT']) / 'outputs' / os.environ['TASK_QUAL_RUN']
source = os.environ['TASK_SOURCE']
methods = ['dsrl-na', 'dsrl-na-retry', 'rl2', 'episode-reset']
records = []
plan = dict(source_commit=source, run_id=os.environ['TASK_RUN'], qualification=str(qual),
            hostname=socket.gethostname(), gpu_count=1, wall_budget_seconds=5400,
            qualification_budget_seconds=1200, admission_ceiling_seconds=4800,
            allocation_slack_seconds=600, prior_failed_qualification_reserve_seconds=360,
            finalization_reserve_seconds=120, admitted=False, target_steps=None,
            methods=methods, seed=1, envs=4, attempts=4, eval_trials=8,
            selection_rule='largest of 48000,28800 whose full timing estimate fits 80 minutes',
            timing_factor=1.75,
            recurrent_core=dict(core='gtrxl', width=128, layers=2, heads=4, memory_length=300),
            startup_io_reserve_per_phase_seconds=60,
            native_success_threshold=.95, online_requested_trials=2,
            scene_pairing_verified=False)


def stamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text())


def event(name, state, **details):
    record = dict(stage=name, state=state, time=stamp(), elapsed_seconds=time.monotonic() - started, **details)
    with (output / 'stages.jsonl').open('a') as stream:
        stream.write(json.dumps(record, allow_nan=False) + '\n')
    print(json.dumps(record), flush=True)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'Invalid qualification measurement: {name}')
    return value


def estimates(target, episode, update, ppo_trial, ppo_episode):
    dsrl_calls = (target - 24016) // 16
    ppo_updates = target // 4800
    phases = {}
    for method in methods:
        dsrl = method.startswith('dsrl-')
        train = target / 300 * episode + dsrl_calls * update if dsrl else ppo_updates * ppo_trial
        evaluation = 32 * episode if dsrl else 96 * ppo_episode
        boundary = 4 * episode + 75 * update if dsrl else ppo_trial
        # CLI deadlines include startup. Its unbounded frozen evaluation and
        # one whole episode/trial overrun are reserved after that deadline.
        training_deadline = math.ceil(train + 30)
        phases[method] = dict(training_estimate_seconds=train,
                              evaluation_reserve_seconds=evaluation,
                              boundary_reserve_seconds=boundary,
                              training_deadline_seconds=training_deadline,
                              total_seconds=math.ceil(training_deadline + evaluation + boundary + 30))
    online_work = 8 * episode + 600 * update
    phases['dsrl-online'] = dict(work_estimate_seconds=online_work,
                                training_deadline_seconds=math.ceil(online_work + 30),
                                boundary_reserve_seconds=episode + 75 * update,
                                total_seconds=math.ceil(online_work + episode + 75 * update + 60))
    return dict(target_steps=target, dsrl_training_calls=dsrl_calls, ppo_updates=ppo_updates,
                phases=phases, total_seconds=sum(p['total_seconds'] for p in phases.values()) + 120)


def validate_evaluation(method, directory):
    data = read_json(directory / 'evaluation.json')
    conditions = ['frozen'] if method.startswith('dsrl-') else ['normal', 'memory_reset', 'reward_blind']
    hashes = {}
    for condition in conditions:
        rows = data[condition]
        if len(rows) != 8:
            raise ValueError(f'{method}/{condition}: incomplete held-out scenes')
        for index, row in enumerate(rows):
            if (row['scene_seeds'] != [2_000_000 + index] or row['attempts'] != 4
                    or row['primitive_steps'] != 1200 or len(row['successes']) != 1
                    or len(row['successes'][0]) != 4):
                raise ValueError(f'{method}/{condition}: incomplete retry trial')
            if not method.startswith('dsrl-'):
                expected_reset = method == 'episode-reset' or condition == 'memory_reset'
                if row.get('reset_each_attempt') is not expected_reset:
                    raise ValueError(f'{method}/{condition}: recurrent reset contract differs')
            initial = row['initial_observation_sha256']
            if len(initial) != 4 or any(item != initial[0] for item in initial):
                raise ValueError(f'{method}/{condition}: retry scene changed')
            hashes.setdefault(str(2_000_000 + index), initial[0])
            if hashes[str(2_000_000 + index)] != initial[0]:
                raise ValueError(f'{method}: evaluation conditions use different scenes')
    return hashes


def run_phase(method, budget, target):
    directory = output / method
    command = [sys.executable, '-m', 'fluffy_bassoon.meta_rl.can_dsrl_run', '--method', method,
               '--dsrl-root', os.environ['TASK_DSRL'],
               '--robomimic-root', os.environ['TASK_ROBOMIMIC'],
               '--checkpoint', str(Path(os.environ['TASK_INPUT']) / 'state_5000.pt'),
               '--normalization', str(Path(os.environ['TASK_INPUT']) / 'normalization.npz'),
               '--source-commit', source,
               '--checkpoint-sha256', '61851045e6b516807826e3bda4270c9e4a086023f2dd85af00eadb59d9b98a1b',
               '--normalization-sha256', 'a4bb04c498625bfad0ee6faae21674c0a06879cd9f9614bbd2da41a7d0dc1c1a',
               '--output', str(directory), '--seed', '1', '--envs', '4', '--attempts', '4',
               '--core', 'gtrxl', '--width', '128', '--layers', '2', '--heads', '4', '--memory-length', '300',
               '--steps', str(target), '--eval-trials', '2' if method == 'dsrl-online' else '8',
               '--max-seconds', str(budget['training_deadline_seconds'])]
    if method == 'dsrl-online':
        command += ['--dsrl-reference', str(output / 'dsrl-na')]
    event(method, 'started', command=command, estimate=budget)
    phase_start = time.monotonic()
    # Scheduler owns the 90-minute allocation. No timeout sends a signal to a
    # learner; the existing CLI saves complete boundaries before evaluating.
    completed = subprocess.run(command, check=False)
    record = dict(method=method, returncode=completed.returncode,
                  phase_elapsed_seconds=time.monotonic() - phase_start, completed=False)
    try:
        if (directory / 'summary.json').exists():
            summary = read_json(directory / 'summary.json')
            record['summary'] = summary
            record['completed'] = completed.returncode == 0 and summary.get('completed') is True
            if method != 'dsrl-online':
                record['completed'] &= summary.get('primitive_steps') == target
                if record['completed']:
                    record['initial_observation_sha256'] = validate_evaluation(method, directory)
    except Exception as error:
        record.update(completed=False, validation_error=f'{type(error).__name__}: {error}')
    records.append(record)
    event(method, 'ended', **record)
    return record


def main():
    reason = None
    exit_code = 0
    try:
        gpu_lines = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name,uuid', '--format=csv,noheader'], text=True).strip().splitlines()
        if len(gpu_lines) != 1 or not re.search(r'RTX\s*(?:4090|(?:PRO\s*)?6000)', gpu_lines[0], re.I):
            raise ValueError(f'Small pilot requires exactly one RTX 4090 or RTX 6000-family GPU: {gpu_lines}')
        plan['gpu'] = gpu_lines[0]
        admission = read_json(qual / 'admission.json')
        qualification = read_json(qual / 'qualification' / 'summary.json')
        contract = read_json(qual / 'qualification' / 'contract.json')
        native_contract = read_json(qual / 'native' / 'contract.json')
        native = read_json(qual / 'native' / 'summary.json')
        native_rows = read_json(qual / 'native' / 'evaluation.json')['native']
        successes = [value for row in native_rows for trial in row['successes'] for value in trial]
        if (admission.get('qualified') is not True or qualification.get('passed') is not True
                or native.get('completed') is not True or len(successes) != 32):
            raise ValueError('Qualification admission or complete 32-episode native evaluation missing')
        native_success = sum(successes) / len(successes)
        if abs(native_success - float(admission['native_success'])) > 1e-12 or native_success >= .95:
            raise ValueError('Native success gate failed or qualification admission disagrees')
        # A launcher-only commit can follow qualification. A model/environment
        # change requires a fresh qualification rather than reusing old timing.
        qualified_source = contract['source_commit']
        for key in ('source_commit', 'checkpoint_sha256', 'normalization_sha256',
                    'core', 'width', 'layers', 'heads', 'memory_length'):
            if native_contract[key] != contract[key]:
                raise ValueError(f'Qualification and native contracts disagree: {key}')
        for key, expected in (
                ('checkpoint_sha256', '61851045e6b516807826e3bda4270c9e4a086023f2dd85af00eadb59d9b98a1b'),
                ('normalization_sha256', 'a4bb04c498625bfad0ee6faae21674c0a06879cd9f9614bbd2da41a7d0dc1c1a')):
            if contract[key] != expected:
                raise ValueError(f'Qualified inputs differ from pilot inputs: {key}')
        if any(contract[key] != value for key, value in plan['recurrent_core'].items()):
            raise ValueError('Qualification must use the declared full GTrXL architecture')
        checked = ['src/fluffy_bassoon/meta_rl/' + name for name in
                   ('can_dsrl.py', 'can_dsrl_run.py', 'can_dsrl_compare.py', 'can_dsrl_parity.py',
                    'can_dsrl_gpu.py', 'ppo.py', 'gtrxl.py')]
        for name in checked:
            qualified = subprocess.check_output(['git', 'show', f'{qualified_source}:{name}'], cwd=worktree)
            if hashlib.sha256(qualified).hexdigest() != file_hash(worktree / name):
                raise ValueError(f'Runtime changed after qualification: {name}')
        full_update = positive(qualification['released_batch_update_seconds'], 'released batch update')
        native_elapsed = positive(native['elapsed_seconds'], 'native elapsed')
        full_ppo = positive(qualification['ppo_full_trial_total_seconds'], 'full PPO trial and update')
        ppo_rollout = positive(qualification['ppo_full_trial_rollout_seconds'], 'full PPO rollout')
        small_ppo = positive(qualification['ppo_small_trial_seconds'], 'single-lane PPO proof')
        positive(qualification['ppo_full_trial_update_seconds'], 'full four-epoch PPO update')
        if (qualification['ppo_full_trial_geometry'] !=
                dict(lanes=4, attempts=4, chunks=300, epochs=4, minibatch_trials=4)
                or qualification['ppo_recovery_roundtrip'].get('passed') is not True):
            raise ValueError('Complete GTrXL training geometry and recovery proof required')
        episode, update = 1.75 * native_elapsed / 32, 1.75 * full_update
        ppo_trial = 1.75 * full_ppo
        # Frozen evaluation is serial by scene. Reserve up to four times the
        # batched per-episode rollout cost, as well as the measured B1 proof.
        ppo_episode = 1.75 * max(native_elapsed / 32, small_ppo / 2, ppo_rollout / 4)
        candidates = [estimates(target, episode, update, ppo_trial, ppo_episode) for target in (48000, 28800)]
        plan.update(qualification_source_commit=qualified_source, native_success=native_success,
                    measured_native_elapsed_seconds=native_elapsed,
                    measured_released_batch_update_seconds=full_update,
                    measured_ppo_full_trial_total_seconds=full_ppo,
                    measured_ppo_full_trial_rollout_seconds=ppo_rollout,
                    measured_ppo_full_trial_update_seconds=qualification['ppo_full_trial_update_seconds'],
                    measured_ppo_full_trial_peak_cuda_memory_gib=qualification['ppo_full_trial_peak_cuda_memory_gib'],
                    conservative_ppo_trial_seconds=ppo_trial, conservative_ppo_episode_seconds=ppo_episode,
                    conservative_episode_seconds=episode, conservative_update_seconds=update,
                    timing_note='1.75 factor covers timing transfer from qualification hardware and runtime variation',
                    qualification_artifact_sha256={name: file_hash(qual / name) for name in
                        ('admission.json', 'qualification/summary.json', 'qualification/contract.json',
                         'native/summary.json', 'native/contract.json', 'native/evaluation.json')}, candidates=candidates)
        chosen = next((candidate for candidate in candidates if candidate['total_seconds'] <= 4800), None)
        if chosen is None:
            raise ValueError('Neither common target fits the conservative 80-minute admission estimate')
        plan.update(admitted=True, target_steps=chosen['target_steps'], estimate=chosen, planned_at=stamp())
        write_json(output / 'plan.json', plan)
        event('pilot', 'admitted', target_steps=chosen['target_steps'], estimate_seconds=chosen['total_seconds'])
        phases = chosen['phases']
        for index, method in enumerate(methods):
            remaining = 5400 - (time.monotonic() - started)
            required = sum(phases[name]['total_seconds'] for name in methods[index:]) + 120
            if remaining < required:
                reason = f'Remaining core phases refused: {remaining:.1f}s available, {required:.1f}s reserved'
                break
            record = run_phase(method, phases[method], chosen['target_steps'])
            if not record['completed']:
                reason = f'{method} failed or stopped below the declared common target'
                break
        if len(records) == 4 and all(record['completed'] for record in records):
            scene_hashes = [record['initial_observation_sha256'] for record in records]
            if any(hashes != scene_hashes[0] for hashes in scene_hashes[1:]):
                raise ValueError('Training arms evaluated different initial scenes')
            plan['scene_pairing_verified'] = True
            remaining = 5400 - (time.monotonic() - started)
            if remaining >= phases['dsrl-online']['total_seconds'] + 120:
                online = run_phase('dsrl-online', phases['dsrl-online'], chosen['target_steps'])
                if not online['completed']:
                    reason = 'Core arms complete; online evaluation incomplete'
            else:
                reason = 'Core arms complete; online evaluation deferred because its full workload does not fit'
                event('dsrl-online', 'deferred', reason=reason, remaining_seconds=remaining)
    except Exception as error:
        reason, exit_code = f'{type(error).__name__}: {error}', 2
        event('pilot', 'refused_or_failed', reason=reason)
    finally:
        write_json(output / 'plan.json', plan)
        core = {record['method']: record for record in records if record['method'] in methods}
        incomplete = [method for method in methods if not core.get(method, {}).get('completed')]
        core_complete = plan['admitted'] and not incomplete and plan['scene_pairing_verified']
        online = next((record for record in records if record['method'] == 'dsrl-online'), None)
        if online and not online['completed']:
            incomplete.append('dsrl-online')
        matched = core_complete and not incomplete
        final = dict(source_commit=source, target_steps=plan['target_steps'], admitted=plan['admitted'],
                     core_training_interactions_complete=core_complete,
                     matched_training_interactions=matched, incomplete_arms=incomplete,
                     comparison='matched interaction-budget feasibility pilot' if matched else 'incomplete; no matched comparison',
                     stage_records=records, online_status='completed' if online and online['completed'] else
                         ('incomplete' if online else 'deferred'),
                     reason=reason, elapsed_seconds=time.monotonic() - started,
                     wall_budget_seconds=5400, finished_at=stamp())
        write_json(output / 'pilot-admission.json', final)
        event('pilot', 'ended', matched_training_interactions=matched, incomplete_arms=incomplete,
              online_status=final['online_status'], reason=reason)
        if not matched:
            exit_code = exit_code or 3
    return exit_code


raise SystemExit(main())
PY
