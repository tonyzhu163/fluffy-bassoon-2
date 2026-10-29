#!/usr/bin/env bash
# Fresh matched pilot: one exclusive two-4090 container, <=50 minutes /100 GPU-minutes.
# Scientific source stays identical to the passed single-GPU qualification.
set -euo pipefail
TASK_SOURCE=${1:?exact source commit}
TASK_RUN=${2:?fresh pilot run ID}
TASK_QUAL_RUN=${3:?passed qualification run ID}
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_WORKTREE=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUTPUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_PYTHON=$TASK_ROOT/envs/dsrl-can-20261029-r2/bin/python
TASK_DSRL=$TASK_ROOT/deps/dsrl-can-20261029-r1/repo
TASK_ROBOMIMIC=$TASK_ROOT/deps/robomimic-compat-20261029-r1/repo
TASK_INPUT=$TASK_ROOT/inputs/dsrl-can-8d21b9c-20261029-r1
TASK_ALLOCATION_STARTED=$(date +%s)
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
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 FLUFFY_GPU_EXECUTION=baige
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl WANDB_MODE=disabled
export TASK_SOURCE TASK_RUN TASK_QUAL_RUN TASK_ROOT TASK_WORKTREE TASK_OUTPUT TASK_DSRL TASK_ROBOMIMIC TASK_INPUT TASK_ALLOCATION_STARTED
"$TASK_PYTHON" - <<'PY'
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from importlib import metadata
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
qual = root / 'outputs' / os.environ['TASK_QUAL_RUN']
source, started = os.environ['TASK_SOURCE'], float(os.environ['TASK_ALLOCATION_STARTED'])
METHODS = ['dsrl-na', 'dsrl-na-retry', 'rl2', 'episode-reset']
WALL, WORK, PROBE, FINAL, FACTOR = 3000, 2700, 90, 120, 1.25
HELPER = 'src/fluffy_bassoon/meta_rl/can_dsrl_gpu.py'
lock, records = threading.Lock(), []
plan = dict(source_commit=source, run_id=os.environ['TASK_RUN'], qualification=str(qual),
            hostname=socket.gethostname(), gpu_count=2, wall_budget_seconds=WALL,
            admission_ceiling_seconds=WORK, startup_and_probe_reserve_seconds=PROBE,
            finalization_reserve_seconds=FINAL, timing_factor=FACTOR, admitted=False,
            target_steps=None, methods=METHODS, seed=1, envs=4, attempts=4, eval_trials=8,
            scene_pairing_verified=False, native_success_threshold=.95,
            recurrent_core=dict(core='gtrxl', width=128, layers=2, heads=4, memory_length=300),
            lanes=[dict(gpu=0, methods=['dsrl-na', 'rl2']),
                   dict(gpu=1, methods=['dsrl-na-retry', 'episode-reset'])],
            selection_rule='largest of48000,28800 whose balanced lanes fit45min work and50min allocation',
            online_requested_trials=0)


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
    row = dict(stage=stage, time=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.time()-started, **details)
    with lock:
        with (output / 'stages.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps(row, allow_nan=False), flush=True)


def positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Qualification timing must be finite and positive')
    return value


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
                raise ValueError('Incomplete retry trial or success metrics')
            if not method.startswith('dsrl-'):
                expected_reset = method == 'episode-reset' or condition == 'memory_reset'
                if row.get('reset_each_attempt') is not expected_reset:
                    raise ValueError('Evaluation reset semantics changed')
            initial = row['initial_observation_sha256']
            if len(initial) != 4 or any(item != initial[0] for item in initial):
                raise ValueError('A retry changed its initial scene')
            if str(scene) in paired and paired[str(scene)] != initial[0]:
                raise ValueError('Memory controls changed the paired scene')
            paired[str(scene)] = initial[0]
    return paired


def recovery(directory, method):
    if method.startswith('dsrl-'):
        manifest = read(directory / 'recovery-manifest.json')
        payload = (directory / manifest['directory']).resolve()
        if not payload.is_relative_to(directory.resolve()):
            raise ValueError('Recovery generation escaped its run')
        if set(manifest['files']) != {'recovery.zip', 'replay.pkl', 'runtime.pt'}:
            raise ValueError('Incomplete DSRL recovery payload')
        for name, expected in manifest['files'].items():
            if digest(payload / name) != expected:
                raise ValueError('DSRL recovery hash differs')
        return dict(manifest_sha256=digest(directory / 'recovery-manifest.json'), **manifest)
    return dict(recovery_sha256=digest(directory / 'recovery.pt'))


def estimate(target, q, native_elapsed):
    native_episode = positive(native_elapsed) / 32
    serial_ppo_episode = positive(q['ppo_small_trial_seconds']) / 2
    full_trial = positive(q['ppo_full_trial_total_seconds'])
    update = positive(q['released_batch_update_seconds'])
    phases = {}
    for method in METHODS:
        dsrl = method.startswith('dsrl-')
        train = target / 300 * native_episode + (target-24016) / 16 * update if dsrl else target / 4800 * full_trial
        evaluate = 32 * native_episode if dsrl else 96 * serial_ppo_episode
        boundary = 4 * native_episode + 75 * update if dsrl else full_trial
        deadline = math.ceil(FACTOR * train + 60)
        phases[method] = dict(training_estimate_seconds=FACTOR*train,
                              training_deadline_seconds=deadline,
                              evaluation_reserve_seconds=math.ceil(FACTOR*evaluate),
                              boundary_reserve_seconds=math.ceil(FACTOR*boundary),
                              startup_io_reserve_seconds=60,
                              total_seconds=deadline+math.ceil(FACTOR*evaluate)+math.ceil(FACTOR*boundary))
    lanes = [sum(phases[name]['total_seconds'] for name in lane['methods']) for lane in plan['lanes']]
    return dict(target_steps=target, dsrl_training_calls=(target-24016)//16,
                ppo_updates=target//4800, phases=phases, lane_seconds=lanes,
                longest_worker_seconds=max(lanes))


def admit():
    admission = read(qual / 'admission.json')
    q, native = read(qual / 'qualification/summary.json'), read(qual / 'native/summary.json')
    qualified, native_contract = read(qual / 'qualification/contract.json'), read(qual / 'native/contract.json')
    rows = read(qual / 'native/evaluation.json')['native']
    outcomes = [value for row in rows for trial in row['successes'] for value in trial]
    if (admission.get('qualified') is not True or q.get('passed') is not True
            or q['environment_parity'].get('passed') is not True
            or q['ppo_recovery_roundtrip'].get('passed') is not True
            or q.get('dsrl_recovery_roundtrip') is not True
            or native.get('completed') is not True or len(rows) != 8 or len(outcomes) != 32):
        raise ValueError('Complete original simulator, learner/recovery and native qualification required')
    success = sum(outcomes)/32
    if abs(success-float(admission['native_success'])) > 1e-12 or success >= .95:
        raise ValueError('Native headroom gate failed')
    if q['ppo_full_trial_geometry'] != dict(lanes=4, attempts=4, chunks=300, epochs=4, minibatch_trials=4):
        raise ValueError('Qualification did not measure the declared complete GTrXL geometry')
    keys = ['source_commit', 'runtime', 'robomimic_commit', 'robomimic_env_sha256',
            'checkpoint_sha256', 'normalization_sha256', 'frozen_motor_sha256'] + list(plan['recurrent_core'])
    if any(qualified[key] != native_contract[key] for key in keys):
        raise ValueError('Native and learner qualifications used different source/runtime/input contracts')
    if any(qualified[key] != value for key,value in plan['recurrent_core'].items()):
        raise ValueError('Qualification did not use the declared GTrXL architecture')
    qualified_source = qualified['source_commit']
    changes = subprocess.check_output(['git','diff','--name-only',qualified_source,source,'--','src'],cwd=worktree,text=True).splitlines()
    if set(changes)-{HELPER}:
        raise ValueError(f'Scientific code changed after qualification: {changes}')
    files = subprocess.check_output(['git','ls-tree','-r','--name-only',source,'--','src'],cwd=worktree,text=True).splitlines()
    scientific_hashes = {name:digest(worktree/name) for name in files if name != HELPER and name.endswith('.py')}
    old_helper = subprocess.check_output(['git','show',f'{qualified_source}:{HELPER}'],cwd=worktree)
    identity = {key:qualified[key] for key in ('runtime','robomimic_commit','robomimic_env_sha256',
                'checkpoint_sha256','normalization_sha256','frozen_motor_sha256')}
    current_runtime = {name:metadata.version(name) for name in identity['runtime']}
    installed = metadata.distribution('robomimic').locate_file('robomimic/envs/env_robosuite.py')
    compatibility = Path(os.environ['TASK_ROBOMIMIC'])
    if (current_runtime != identity['runtime'] or digest(installed) != identity['robomimic_env_sha256']
            or digest(compatibility/'robomimic/envs/env_robosuite.py') != identity['robomimic_env_sha256']
            or subprocess.check_output(['git','rev-parse','HEAD'],cwd=compatibility,text=True).strip()!=identity['robomimic_commit']
            or subprocess.check_output(['git','status','--porcelain'],cwd=compatibility,text=True).strip()):
        raise ValueError('Current packages differ from the qualified runtime')
    for name,expected in [('state_5000.pt',identity['checkpoint_sha256']),('normalization.npz',identity['normalization_sha256'])]:
        if digest(Path(os.environ['TASK_INPUT'])/name)!=expected:
            raise ValueError('Frozen input artifact changed after qualification')
    probe = read(qual/'native/gpu-probe.json')
    devices = probe.get('devices',[probe.get('selected_device',{})])
    if len(devices)!=1 or re.search(r'RTX\s*4090',devices[0].get('cuda_name',''),re.I) is None:
        raise ValueError('Qualification timings must come from an actual single4090')
    candidates = [estimate(target,q,native['elapsed_seconds']) for target in (48000,28800)]
    available = WALL-(time.time()-started)-PROBE-FINAL
    chosen = next((row for row in candidates if row['longest_worker_seconds']<=min(WORK,available)),None)
    if chosen is None:
        raise ValueError('No common target fits45min work within the50min allocation')
    plan.update(qualification_source_commit=qualified_source, runtime_identity=identity,
                native_success=success, measured_native_elapsed_seconds=native['elapsed_seconds'],
                measured_ppo_serial_episode_seconds=q['ppo_small_trial_seconds']/2,
                measured_released_batch_update_seconds=q['released_batch_update_seconds'],
                measured_ppo_full_trial_total_seconds=q['ppo_full_trial_total_seconds'],
                evaluation_timing_rule='B1K2 measured rollout/2; fullK4 update and1.25margin cover late attention cost',
                scientific_source_sha256=scientific_hashes, candidates=candidates, estimate=chosen,
                target_steps=chosen['target_steps'],
                helper_migration=dict(qualified_source_commit=qualified_source, permitted_changed_source=[HELPER],
                    old_sha256=hashlib.sha256(old_helper).hexdigest(), new_sha256=digest(worktree/HELPER),
                    change='admit complete2GPU parent reports alongside1and8; no scientific source changes',
                    cpu_tests='pending', actual_selected_gpu_probes='pending'),
                qualification_artifact_sha256={name:digest(qual/name) for name in
                    ('admission.json','qualification/summary.json','qualification/contract.json',
                     'native/summary.json','native/contract.json','native/evaluation.json')})
    return chosen


def worker(ordinal,methods,chosen,parent_probe):
    for index,method in enumerate(methods):
        phases=chosen['phases']
        required=sum(phases[name]['total_seconds'] for name in methods[index:])+FINAL
        if WALL-(time.time()-started)<required:
            event('phase_refused',gpu=ordinal,method=method,reason='remaining complete phases and audit no longer fit')
            break
        directory=output/method
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(ordinal),LYCO_BAIGE_GPU_MAP=str(parent_probe),
                 NUMBA_CACHE_DIR=str(root/'cache'/f'{os.environ["TASK_RUN"]}-gpu{ordinal}'))
        env.pop('LYCO_BAIGE_ISOLATED_UUID',None)
        if index==0:
            Path(env['NUMBA_CACHE_DIR']).mkdir(parents=True,exist_ok=False)
        command=[sys.executable,'-m','fluffy_bassoon.meta_rl.can_dsrl_run','--method',method,
            '--dsrl-root',os.environ['TASK_DSRL'],'--robomimic-root',os.environ['TASK_ROBOMIMIC'],
            '--checkpoint',str(Path(os.environ['TASK_INPUT'])/'state_5000.pt'),
            '--normalization',str(Path(os.environ['TASK_INPUT'])/'normalization.npz'),
            '--source-commit',source,'--gpu-attestation',str(parent_probe),
            '--checkpoint-sha256',plan['runtime_identity']['checkpoint_sha256'],
            '--normalization-sha256',plan['runtime_identity']['normalization_sha256'],
            '--output',str(directory),'--seed','1','--envs','4','--attempts','4',
            '--core','gtrxl','--width','128','--layers','2','--heads','4','--memory-length','300',
            '--steps',str(chosen['target_steps']),'--eval-trials','8',
            '--max-seconds',str(phases[method]['training_deadline_seconds'])]
        event('phase_started',gpu=ordinal,method=method,command=command,estimate=phases[method])
        phase_start=time.monotonic()
        # Model processes receive no timeout or signal. The original driver
        # owns complete recovery boundaries; the scheduler has the50min cap.
        with (output/f'{method}.console.log').open('x') as log:
            process=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=False)
        record=dict(gpu=ordinal,method=method,returncode=process.returncode,
                    phase_elapsed_seconds=time.monotonic()-phase_start,completed=False)
        try:
            summary,contract=read(directory/'summary.json'),read(directory/'contract.json')
            if process.returncode or summary.get('completed') is not True or summary['primitive_steps']!=chosen['target_steps']:
                raise ValueError('Method failed or stopped below the common target')
            if any(contract[key]!=value for key,value in plan['runtime_identity'].items()):
                raise ValueError('Qualified runtime or frozen motor changed')
            if contract['source_commit']!=source or contract['seed']!=1:
                raise ValueError('Method source or paired training seed changed')
            worker_probe=read(directory/'gpu-probe.json')
            if (worker_probe['status']!='passed' or worker_probe['parent_gpu_count']!=2
                    or worker_probe['parent_probe_sha256']!=plan['parent_probe_sha256']
                    or worker_probe['cuda_visible_devices']!=str(ordinal)):
                raise ValueError('Actual worker GPU qualification differs from parent allocation')
            record.update(completed=True,summary=summary,initial_observation_sha256=evaluation(directory,method),
                          recovery=recovery(directory,method),gpu_probe_sha256=digest(directory/'gpu-probe.json'),
                          contract_sha256=digest(directory/'contract.json'),evaluation_sha256=digest(directory/'evaluation.json'))
        except Exception as error:
            record.update(completed=False,validation_error=f'{type(error).__name__}: {error}')
        with lock:
            records.append(record)
            write(output/'phase-records.json',records)
        event('phase_ended',**record)
        if not record['completed']:
            break


exit_code,reason=0,None
try:
    chosen=admit()
    # This migration requalifies GPU accounting only. All scientific source
    # files remain byte-identical to the passed c103 qualification.
    test_env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
    with (output/'runtime-tests.log').open('x') as log:
        subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider','tests/test_can_dsrl_gpu.py'],
                       env=test_env,stdout=log,stderr=subprocess.STDOUT,check=True)
    plan['helper_migration']['cpu_tests']=dict(passed=True,log_sha256=digest(output/'runtime-tests.log'),
                                              source_sha256=digest(worktree/'tests/test_can_dsrl_gpu.py'))
    parent_probe=output/'parent-gpu-probe.json'
    subprocess.run([sys.executable,str(worktree/'scripts/[train]gpu_probe.py'),'--expected-count','2',
                    '--output',str(parent_probe),'--purpose','training','--baige-container'],check=True,timeout=PROBE)
    probe=read(parent_probe)
    if len(probe['devices'])!=2 or any(re.search(r'RTX\s*4090',row['cuda_name'],re.I) is None for row in probe['devices']):
        raise ValueError('This pilot requires two actual4090GPUs')
    assigned=os.environ['NVIDIA_VISIBLE_DEVICES'].split(',')
    if len(assigned)!=2 or set(assigned)!={row['uuid'] for row in probe['devices']}:
        raise ValueError('Complete scheduler NVIDIA allocation changed')
    probe.update(source_commit=source,scientific_source_commit=plan['qualification_source_commit'],
                 source_files_sha256=plan['scientific_source_sha256'],helper_sha256=plan['helper_migration']['new_sha256'])
    write(parent_probe,probe)
    if WALL-(time.time()-started)<chosen['longest_worker_seconds']+FINAL:
        raise ValueError('Runtime qualification consumed the full remaining phase reserve')
    plan.update(admitted=True,parent_probe_sha256=digest(parent_probe))
    write(output/'plan.json',plan)
    event('pilot_admitted',target_steps=chosen['target_steps'],lane_seconds=chosen['lane_seconds'])
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,lane['gpu'],lane['methods'],chosen,parent_probe) for lane in plan['lanes']]
        for future in futures:
            future.result()
except Exception as error:
    reason,exit_code=f'{type(error).__name__}: {error}',2
    event('pilot_refused_or_failed',reason=reason)
finally:
    completed={row['method']:row for row in records if row['completed']}
    if len(completed)==4:
        hashes=[completed[name]['initial_observation_sha256'] for name in METHODS]
        plan['scene_pairing_verified']=all(value==hashes[0] for value in hashes[1:])
        plan['helper_migration']['actual_selected_gpu_probes']={name:completed[name]['gpu_probe_sha256'] for name in METHODS}
    matched=plan['admitted'] and len(completed)==4 and plan['scene_pairing_verified']
    final=dict(source_commit=source,target_steps=plan['target_steps'],admitted=plan['admitted'],
               core_training_interactions_complete=matched,matched_training_interactions=matched,
               incomplete_arms=[name for name in METHODS if name not in completed],
               comparison='matched interaction-budget feasibility pilot' if matched else 'incomplete; no matched comparison',
               stage_records=records,online_status='outside this bounded four-arm pilot',reason=reason,
               elapsed_seconds=time.time()-started,wall_budget_seconds=WALL,
               allocated_gpu_minutes=(time.time()-started)*2/60,
               finished_at=datetime.now(timezone.utc).isoformat())
    write(output/'plan.json',plan)
    write(output/'pilot-admission.json',final)
    event('pilot_ended',matched_training_interactions=matched,incomplete_arms=final['incomplete_arms'],reason=reason)
    if not matched:
        exit_code=exit_code or 3
raise SystemExit(exit_code)
PY
