#!/usr/bin/env bash
# Optional original-DSRL online control: one4090, <=15 minutes, no automatic retry.
# The scheduler must request one GPU/16CPU/64GiB/shm8GiB with a15-minute hard cap.
set -euo pipefail
TASK_SOURCE=${1:?exact latest-main source commit}
TASK_RUN=${2:?fresh online run ID}
TASK_PILOT_RUN=${3:-robomimic-can-dsrl-baige-20261029-pilot-r3}
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_WORKTREE=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUTPUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_PYTHON=$TASK_ROOT/envs/dsrl-can-20261029-r2/bin/python
TASK_DSRL=$TASK_ROOT/deps/dsrl-can-20261029-r1/repo
TASK_ROBOMIMIC=$TASK_ROOT/deps/robomimic-compat-20261029-r1/repo
TASK_INPUT=$TASK_ROOT/inputs/dsrl-can-8d21b9c-20261029-r1
TASK_ALLOCATION_STARTED=$(date +%s)
[[ $TASK_SOURCE =~ ^[0-9a-f]{40}$ ]]
[[ $TASK_RUN =~ ^robomimic-can-dsrl-baige-20261029-online-r[0-9]+$ ]]
[[ $TASK_PILOT_RUN == robomimic-can-dsrl-baige-20261029-pilot-r3 ]]
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
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl WANDB_MODE=disabled
export NUMBA_CACHE_DIR="$TASK_ROOT/cache/$TASK_RUN"
export TASK_SOURCE TASK_RUN TASK_PILOT_RUN TASK_ROOT TASK_WORKTREE TASK_OUTPUT TASK_DSRL TASK_ROBOMIMIC TASK_INPUT TASK_ALLOCATION_STARTED
mkdir "$NUMBA_CACHE_DIR"
"$TASK_PYTHON" - <<'PY'
from datetime import datetime, timezone
from importlib import metadata
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

root, output, worktree = (Path(os.environ[name]) for name in ('TASK_ROOT','TASK_OUTPUT','TASK_WORKTREE'))
source, started = os.environ['TASK_SOURCE'], float(os.environ['TASK_ALLOCATION_STARTED'])
pilot = root/'outputs'/os.environ['TASK_PILOT_RUN']
reference, online = pilot/'dsrl-na', output/'online'
methods = ['dsrl-na','dsrl-na-retry','rl2','episode-reset']
plan = dict(source_commit=source, run_id=os.environ['TASK_RUN'], pilot=str(pilot), reference=str(reference),
            hostname=socket.gethostname(), admitted=False, gpu_count=1, wall_budget_seconds=900,
            cli_deadline_seconds=660, startup_and_complete_boundary_reserve_seconds=240,
            seed=1, attempts=4, requested_scenes=2, training_reference_steps=28800,
            recurrent_core=dict(core='gtrxl',width=128,layers=2,heads=4,memory_length=300),
            budget_gpu_minutes=dict(prior_failed_reserve=6,qualification_reserve=12,completed_pilot_reserve=51,
                                    optional_online_cap=15,total_ceiling=84,authorized_initial_ceiling=120),
            comparison_metadata=dict(
                scope='two paired held-out scenes only; eight complete retry episodes',
                interaction_budget='same two scenes, K4 and2400 new primitive transitions as paired frozen evaluations',
                deployment='stronger conventional test-time adaptation control with different optimizer compute',
                released_training_lanes=4, online_adaptation_lanes=1,
                gradient_steps_per_call=20, noise_critic_steps_per_call=10,
                online_updates_per_new_transition_relative_to_released_training=4,
                update_ratio_note='20+10 updates follow each single-lane chunk online; training uses20+10 per four-lane vector step',
                interpretation='matched interaction budget; optimizer compute and update-to-data ratio differ'),
            automatic_retry=False)


def read(path):
    return json.loads(path.read_text())


def write(path,value):
    pending=path.with_suffix(path.suffix+'.tmp')
    pending.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    pending.replace(path)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def reference_hashes():
    manifest=read(reference/'recovery-manifest.json')
    directory=(reference/manifest['directory']).resolve()
    if not directory.is_relative_to(reference.resolve()):
        raise ValueError('Reference recovery escaped its immutable run')
    if set(manifest['files'])!={'recovery.zip','replay.pkl','runtime.pt'}:
        raise ValueError('Reference recovery lacks complete learner/replay/runtime payloads')
    hashes={name:digest(directory/name) for name in manifest['files']}
    if hashes!=manifest['files']:
        raise ValueError('Reference recovery hash differs')
    hashes['recovery-manifest.json']=digest(reference/'recovery-manifest.json')
    return hashes


def preflight():
    admission, prior=read(pilot/'pilot-admission.json'),read(pilot/'plan.json')
    if (admission.get('core_training_interactions_complete') is not True
            or admission.get('matched_training_interactions') is not True
            or prior.get('scene_pairing_verified') is not True
            or admission['target_steps']!=28800 or prior['target_steps']!=28800):
        raise ValueError('A complete paired four-arm28,800-step pilot is required')
    training_source=admission['source_commit']
    if prior['source_commit']!=training_source:
        raise ValueError('Pilot source contracts disagree')
    changes=subprocess.check_output(['git','diff','--name-only',training_source,source,'--','src'],
                                    cwd=worktree,text=True).splitlines()
    if changes:
        raise ValueError(f'Scientific/runtime source changed after pilot: {changes}')
    baseline=None
    paired={}
    for method in methods:
        directory=pilot/method
        summary,contract=read(directory/'summary.json'),read(directory/'contract.json')
        if (summary.get('completed') is not True or summary['primitive_steps']!=28800
                or contract['source_commit']!=training_source or contract['seed']!=1
                or any(contract[key]!=value for key,value in plan['recurrent_core'].items())):
            raise ValueError('Pilot arm is incomplete or its source/seed/core differs')
        identity={key:contract[key] for key in ('runtime','robomimic_commit','robomimic_env_sha256',
                    'checkpoint_sha256','normalization_sha256','frozen_motor_sha256')}
        if baseline is not None and identity!=baseline:
            raise ValueError('Pilot arms used different runtime or frozen input identities')
        baseline=identity
        condition='frozen' if method.startswith('dsrl-') else 'normal'
        rows=read(directory/'evaluation.json')[condition]
        if len(rows)!=8:
            raise ValueError('Pilot held-out evaluation is incomplete')
        for index,row in enumerate(rows[:2]):
            scene=2_000_000+index
            initial=row['initial_observation_sha256']
            if (row['scene_seeds']!=[scene] or row['attempts']!=4 or row['primitive_steps']!=1200
                    or len(initial)!=4 or any(value!=initial[0] for value in initial)):
                raise ValueError('Pilot retry geometry or scene identity differs')
            if str(scene) in paired and paired[str(scene)]!=initial[0][0]:
                raise ValueError('Pilot methods evaluated different paired scenes')
            paired[str(scene)]=initial[0][0]
    versions={name:metadata.version(name) for name in baseline['runtime']}
    installed=metadata.distribution('robomimic').locate_file('robomimic/envs/env_robosuite.py')
    compatibility=Path(os.environ['TASK_ROBOMIMIC'])
    if (versions!=baseline['runtime'] or digest(installed)!=baseline['robomimic_env_sha256']
            or digest(compatibility/'robomimic/envs/env_robosuite.py')!=baseline['robomimic_env_sha256']
            or subprocess.check_output(['git','rev-parse','HEAD'],cwd=compatibility,text=True).strip()!=baseline['robomimic_commit']
            or subprocess.check_output(['git','status','--porcelain'],cwd=compatibility,text=True).strip()):
        raise ValueError('Current runtime differs from the completed pilot')
    for name,expected in [('state_5000.pt',baseline['checkpoint_sha256']),('normalization.npz',baseline['normalization_sha256'])]:
        if digest(Path(os.environ['TASK_INPUT'])/name)!=expected:
            raise ValueError('Frozen input changed after the pilot')
    pins={'.':'8d21b9cf55459f022c0e769bca1343edd9d30e5a',
          'dppo':'86ce51834055c02f9013e60dd4c4275606d82df7',
          'stable-baselines3':'10e5d311b5c42fe571ff3eeb72f622ca616a149d'}
    for name,expected in pins.items():
        path=Path(os.environ['TASK_DSRL'])/name
        if (subprocess.check_output(['git','rev-parse','HEAD'],cwd=path,text=True).strip()!=expected
                or subprocess.check_output(['git','status','--porcelain'],cwd=path,text=True).strip()):
            raise ValueError('Pinned original DSRL source changed')
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid','--format=csv,noheader'],text=True).strip().splitlines()
    if len(gpu)!=1 or re.search(r'RTX\s*4090',gpu[0],re.I) is None or len(os.environ['NVIDIA_VISIBLE_DEVICES'].split(','))!=1:
        raise ValueError('Online control requires one actual allocated4090')
    if 900-(time.time()-started)<660+180:
        raise ValueError('Startup consumed the complete online and boundary budget')
    plan.update(admitted=True, training_source_commit=training_source, runtime_identity=baseline,
                paired_scene_sha256=paired, reference_sha256=reference_hashes(), gpu=gpu[0],
                pilot_artifact_sha256={name:digest(pilot/name) for name in ('pilot-admission.json','plan.json')})


exit_code,reason,complete=0,None,False
try:
    preflight()
    write(output/'plan.json',plan)
    command=[sys.executable,'-m','fluffy_bassoon.meta_rl.can_dsrl_run','--method','dsrl-online',
        '--dsrl-root',os.environ['TASK_DSRL'],'--robomimic-root',os.environ['TASK_ROBOMIMIC'],
        '--checkpoint',str(Path(os.environ['TASK_INPUT'])/'state_5000.pt'),
        '--normalization',str(Path(os.environ['TASK_INPUT'])/'normalization.npz'),
        '--source-commit',source,'--checkpoint-sha256',plan['runtime_identity']['checkpoint_sha256'],
        '--normalization-sha256',plan['runtime_identity']['normalization_sha256'],
        '--output',str(online),'--dsrl-reference',str(reference),'--eval-trials','2','--attempts','4',
        '--seed','1','--envs','4','--steps','28800','--core','gtrxl','--width','128',
        '--layers','2','--heads','4','--memory-length','300','--max-seconds','660']
    plan['command']=command
    write(output/'plan.json',plan)
    # The unchanged CLI performs the full one-GPU probe. No timeout/signal is
    # sent to its model process; incomplete scenes keep existing partial recovery.
    result=subprocess.run(command,check=False)
    plan['cli_returncode']=result.returncode
    if result.returncode:
        raise RuntimeError('Online CLI failed; preserve its complete/partial artifacts for diagnosis')
    summary,data=read(online/'summary.json'),read(online/'online-evaluation.json')
    contract=read(online/'contract.json')
    if (contract['source_commit']!=source or contract['method']!='dsrl-online' or contract['seed']!=1
            or contract['eval_trials']!=2 or contract['attempts']!=4
            or any(contract[key]!=value for key,value in plan['runtime_identity'].items())
            or data['training_source_contract']['source_commit']!=plan['training_source_commit']
            or data['training_source_contract']['method']!='dsrl-na'):
        raise ValueError('Online source/runtime or saved training reference contract differs')
    if reference_hashes()!=plan['reference_sha256'] or data['reference_sha256']!=plan['reference_sha256']:
        raise ValueError('The immutable DSRL reference recovery changed')
    for name,expected in [('state_5000.pt',plan['runtime_identity']['checkpoint_sha256']),
                          ('normalization.npz',plan['runtime_identity']['normalization_sha256'])]:
        if digest(Path(os.environ['TASK_INPUT'])/name)!=expected:
            raise ValueError('The source diffusion checkpoint or normalization artifact changed')
    if (data.get('reference_unchanged') is not True or data.get('motor_unchanged') is not True
            or data['frozen_motor_sha256']!=plan['runtime_identity']['frozen_motor_sha256']):
        raise ValueError('Frozen reference or motor integrity check failed')
    if not summary.get('completed') or not data.get('completed'):
        raise RuntimeError('Online control is incomplete; existing partial recoveries remain preserved')
    if (len(data['scenes'])!=2 or data['primitive_steps']!=2400 or data['adaptation_learner_updates']!=12000
            or data['adaptation_optimizer_steps_total']!=42000):
        raise ValueError('Complete online aggregate counters differ from the declared two-scene control')
    for index,row in enumerate(data['scenes']):
        scene=2_000_000+index
        if (row['scene_seed']!=scene or row['attempts']!=4 or row['completed'] is not True
                or row['primitive_steps']!=1200 or row['adaptation_train_calls']!=300
                or row['adaptation_learner_updates']!=6000 or len(row['attempt_records'])!=4
                or row['adaptation_optimizer_updates']!=dict(actor=6000,critic=6000,noise_critic=3000,entropy=6000)
                or row['initial_observation_sha256']!=[plan['paired_scene_sha256'][str(scene)]]*4):
            raise ValueError('Online scene geometry, updates or paired initial state differs')
    complete=True
    plan.update(online_evaluation_sha256=digest(online/'online-evaluation.json'),
                online_summary_sha256=digest(online/'summary.json'),
                adaptation_train_calls=600,actor_updates=12000,noise_critic_updates=6000,
                immutable_reference_unchanged=True,source_checkpoint_unchanged=True)
except Exception as error:
    reason,exit_code=f'{type(error).__name__}: {error}',2
finally:
    final=dict(source_commit=source,admitted=plan['admitted'],completed=complete,reason=reason,
               requested_scenes=2,requested_episodes=8,requested_primitive_steps=2400,
               elapsed_seconds=time.time()-started,wall_budget_seconds=900,
               allocated_gpu_minutes=(time.time()-started)/60,budget_gpu_minutes=plan['budget_gpu_minutes'],
               comparison_metadata=plan['comparison_metadata'],
               reference=str(reference),partial_artifacts_preserved=True,automatic_retry=False,
               finished_at=datetime.now(timezone.utc).isoformat())
    write(output/'plan.json',plan)
    write(output/'online-admission.json',final)
    print(json.dumps(final,allow_nan=False),flush=True)
raise SystemExit(exit_code)
PY
