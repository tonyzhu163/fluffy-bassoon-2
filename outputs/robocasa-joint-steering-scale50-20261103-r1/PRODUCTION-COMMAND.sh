#!/usr/bin/env bash
# The launch wrapper may be a newer receipt-only Git commit than TASK_META.
# TASK_META remains the exact native/learner/worker source admitted below.
set -Eeuo pipefail
: "${TASK_META:?}" "${TASK_CODE_PIN:?}" "${TASK_ARM:?}" "${TASK_RUN:?}"
: "${TASK_ADMISSION:?}" "${TASK_ADMISSION_SHA256:?}" "${TASK_SUBMITTED_UNIX:?}"
: "${RANK:?}" "${WORLD_SIZE:?}" "${MASTER_ADDR:?}"
test "$RANK" = 0
test "$WORLD_SIZE" = 1
case "$TASK_ARM" in velocity|kv_all_layers) ;; *) exit 2 ;; esac
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_PY=$TASK_ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
export TASK_PROCESS_STARTED=$(date +%s)
test ! -e "$TASK_RUN"
mkdir -p "$TASK_RUN"
exec > >(tee "$TASK_RUN/job-console.log") 2>&1
finish() {
  TASK_EXIT_CODE=$?
  trap - EXIT
  TASK_FINISHED=$(date +%s)
  printf '{"started_unix":%s,"finished_unix":%s,"allocated_gpus":8,"process_gpu_seconds":%s,"exit_code":%s,"allocation_release_not_included":true}\n' \
    "$TASK_PROCESS_STARTED" "$TASK_FINISHED" "$((8*(TASK_FINISHED-TASK_PROCESS_STARTED)))" "$TASK_EXIT_CODE" > "$TASK_RUN/clock.json"
  exit "$TASK_EXIT_CODE"
}
trap finish EXIT
export RLINF_PATH=$TASK_ROOT/sources/robocasa365-public-pi05-nnx-leaves-20261101-r1/RLinf
export PYTHONPATH=$TASK_META/src:$RLINF_PATH
export ROBOCASA_ASSETS_PATH=$TASK_ROOT/setups/robocasa365-native-20261101-r1/assets
export OPENPI_DATA_HOME=$TASK_ROOT/setups/robocasa365-native-20261101-r1/cache/.cache/openpi
export FLUFFY_GPU_EXECUTION=baige MUJOCO_GL=egl PYOPENGL_PLATFORM=egl NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export WANDB_MODE=offline WANDB_CONSOLE=off WANDB_DISABLE_GIT=true WANDB_DISABLE_CODE=true
# No credential is needed or serialized for the qualified offline exporter.
unset WANDB_API_KEY WANDB_ACCESS_TOKEN WANDB_IDENTITY_TOKEN_FILE
export LYCO_BAIGE_GPU_MAP=$TASK_RUN/GPU-PROBE.json
"$TASK_PY" -B "$TASK_META/scripts/[train]gpu_probe.py" --expected-count 8 \
  --output "$LYCO_BAIGE_GPU_MAP" --baige-container --purpose training --baige-exclusive-training
"$TASK_PY" -B - "$TASK_RUN" <<'PY'
import hashlib,json,math,os,pathlib,socket,subprocess,time
from fluffy_bassoon.meta_rl.joint_steering_ppo import JointActionJournal
run=pathlib.Path(__import__('sys').argv[1]); admission_path=pathlib.Path(os.environ['TASK_ADMISSION'])
assert socket.gethostname()==os.environ['MASTER_ADDR']
assert hashlib.sha256(admission_path.read_bytes()).hexdigest()==os.environ['TASK_ADMISSION_SHA256']
a=json.loads(admission_path.read_text()); arm=os.environ['TASK_ARM']; row=a['arms'][arm]
assert a['completed'] and a['worker_lanes']==48 and row['passed']
code=pathlib.Path(os.environ['TASK_META'])
assert a['source_commit']==os.environ['TASK_CODE_PIN']==subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip()
assert not subprocess.check_output(['git','status','--porcelain'],cwd=code,text=True).strip()
storage=os.statvfs(run)
assert storage.f_bavail*storage.f_frsize >= a['required_free_storage_bytes_at_launch']
quota,period=pathlib.Path('/sys/fs/cgroup/cpu.max').read_text().split()
available_CPU=len(os.sched_getaffinity(0))
if quota!='max':
 available_CPU=min(available_CPU,int(quota)/int(period))
assert available_CPU>=168
assert int(pathlib.Path('/sys/fs/cgroup/memory.max').read_text())>=1664*2**30
assert os.statvfs('/dev/shm').f_blocks*os.statvfs('/dev/shm').f_frsize>=384*2**30
probe=json.loads((run/'GPU-PROBE.json').read_text())
assert probe['status']=='passed' and probe['hostname']==socket.gethostname() and len(probe['devices'])==8
prior=row['prior_conservative_GPU_seconds']; submitted=float(os.environ['TASK_SUBMITTED_UNIX'])
assert 0<=prior<40*3600 and submitted<=time.time()
deadline=math.floor(submitted+(40*3600-prior)/8-150)
assert deadline-time.time()>120
native_env={k:os.environ[k] for k in ('RLINF_PATH','ROBOCASA_ASSETS_PATH','OPENPI_DATA_HOME','MUJOCO_GL','PYOPENGL_PLATFORM','NCCL_P2P_DISABLE')}
profile=dict(code=str(code),source_commit=a['source_commit'],runtime_path=os.environ['RLINF_PATH'],
 runtime_commit=a['runtime_commit'],model_directory=a['model_directory'],model_sha256=a['model_sha256'],
 model_config_sha256=a['model_config_sha256'],norm_path=a['norm_path'],norm_sha256=a['norm_sha256'],
 arm=arm,seed=921801 if arm=='velocity' else 921901,updates=50,episodes_per_update=384,lanes_per_worker=48,
 submitted_unix=submitted,prior_conservative_gpu_seconds=prior,native_env=native_env,
 canonical_root=str(run/'canonical'),admission_reference=dict(path=str(admission_path),sha256=os.environ['TASK_ADMISSION_SHA256']),
 wandb=dict(enabled=True,entity='tonyzoo-brown-university',project='fluffy-bassoon-2',
 group='joint-steering-scale50-20261103-r1',online_authorized=False))
(run/'PROFILE.json').write_text(json.dumps(profile,indent=2)+'\n')
(run/'deadline-unix.txt').write_text(str(deadline)+'\n')
# Carry earlier physical calls into the same hard TRAIN ceiling once; these
# ledger entries explicitly reference external qualification, not new dispatch.
prior_controls=row['prior_train_controls']
assert type(prior_controls) is int and 0<=prior_controls<=130000
journal=JointActionJournal(run/'accounting',rank=98,category='train',scope='carried-native-qualification')
if prior_controls:
 token=journal.begin(prior_controls,phase='carried_qualification_receipt')
 journal.finish(token,prior_controls)
(run/'CARRIED-ACTION-ACCOUNTING.json').write_text(json.dumps(dict(controls=prior_controls,
 scientific_updates=0,new_physical_dispatch=False,source=a['qualification_reference'],
 source_arm=arm,allocation_GPU_seconds=prior),indent=2)+'\n')
PY
TASK_DEADLINE=$(cat "$TASK_RUN/deadline-unix.txt")
TASK_REMAINING=$((TASK_DEADLINE+90-$(date +%s)))
test "$TASK_REMAINING" -gt 90
timeout --signal=TERM --kill-after=30s "${TASK_REMAINING}s" "$TASK_PY" -B -m \
  fluffy_bassoon.meta_rl.joint_steering_ppo_driver --run "$TASK_RUN" --profile "$TASK_RUN/PROFILE.json" \
  --deadline-unix "$TASK_DEADLINE"
