#!/usr/bin/env bash
# Common corrected PPO recipe: four fresh sources and one exact continuation.
set -Eeuo pipefail
TASK_PIN=${1:?exact reviewed commit}
TASK_NAME=${2:?source task}
TASK_ATTEMPT=${3:?fresh attempt name}
TASK_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_RUN=robocasa-six-task-scale-20261113-r1
TASK_PY=$TASK_ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
case "$TASK_NAME" in
  OpenDrawer) TASK_TRAIN=2100000; TASK_EVAL=610100 ;;
  CloseToasterOvenDoor) TASK_TRAIN=2700000; TASK_EVAL=670100 ;;
  CoffeeSetupMug) TASK_TRAIN=2500000; TASK_EVAL=650100 ;;
  TurnOffStove) TASK_TRAIN=2400000; TASK_EVAL=640100 ;;
  PickPlaceCounterToStove) TASK_TRAIN=2600000; TASK_EVAL=660100 ;;
  *) exit 2 ;;
esac
[[ $TASK_ATTEMPT =~ ^[a-z0-9-]+$ ]]
[[ $(uname -s) == Linux && ${HOSTNAME:-} == job-* ]]
[[ $(git -C "$TASK_CODE" rev-parse HEAD) == "$TASK_PIN" ]]
[[ -z $(git -C "$TASK_CODE" status --porcelain --untracked-files=no) ]]
export TASK_ATTEMPT
if [[ $TASK_NAME == PickPlaceCounterToStove ]]; then
  [[ -n ${ROBOCASA_SIM_PPO_RESUME:-} && -n ${ROBOCASA_SIM_PPO_BEFORE_REFERENCE:-} && -n ${ROBOCASA_SIM_PPO_BEFORE_SHA256:-} ]]
  TASK_INITIAL_STEP=10; TASK_CHECKPOINT_SECONDS=5400; TASK_MAXIMUM_GPU_HOURS=16
else
  [[ -z ${ROBOCASA_SIM_PPO_RESUME:-} && -z ${ROBOCASA_SIM_PPO_BEFORE_REFERENCE:-} && -z ${ROBOCASA_SIM_PPO_BEFORE_SHA256:-} ]]
  unset ROBOCASA_SIM_PPO_RESUME ROBOCASA_SIM_PPO_BEFORE_REFERENCE ROBOCASA_SIM_PPO_BEFORE_SHA256
  TASK_INITIAL_STEP=0; TASK_CHECKPOINT_SECONDS=7200; TASK_MAXIMUM_GPU_HOURS=20
fi
export TASK_INITIAL_STEP TASK_MAXIMUM_GPU_HOURS
export TASK_CODE_PIN=$TASK_PIN
export RLINF_PATH=$TASK_ROOT/sources/critic-selected-a1229239-r1
[[ $(git -C "$RLINF_PATH" rev-parse HEAD) == 8f9bfa7661928ef8c17494b04f4d7bf3125634e1 ]]
export PYTHONPATH=$TASK_CODE/src:$RLINF_PATH
export ROBOCASA_ASSETS_PATH=$TASK_ROOT/setups/robocasa365-native-20261101-r1/assets
export OPENPI_DATA_HOME=$TASK_ROOT/setups/robocasa365-native-20261101-r1/cache/.cache/openpi
export ROBOCASA_PI05_TORCH_MODEL=$TASK_ROOT/inputs/robocasa365-public-pi05-c484-torch-20261101-r2
export ROBOCASA_PI05_NATIVE_NORM=$TASK_ROOT/inputs/robocasa365-public-pi05-c484-75000-20261101-r1/assets/norm_stats.json
export ROBOCASA_SIM_PPO_OUTPUT=${TASK_LOCAL_OUTPUT:-$TASK_ROOT/outputs/$TASK_RUN/teachers/$TASK_NAME/$TASK_ATTEMPT}
export ROBOCASA_SIM_PPO_ACCOUNTING=$ROBOCASA_SIM_PPO_OUTPUT/accounting
export ROBOCASA_FLOW_TASK=$TASK_NAME ROBOCASA_FLOW_TRAIN_SEED=$TASK_TRAIN ROBOCASA_FLOW_EVAL_SEED=$TASK_EVAL
export ROBOCASA_SIM_PPO_ACTOR_PLACEMENT=0-1 ROBOCASA_SIM_PPO_ROLLOUT_PLACEMENT=0-7
export ROBOCASA_SIM_PPO_MICRO_BATCH=16 ROBOCASA_SIM_PPO_WORLD_SIZE=8
export ROBOCASA_FLOW_TRAIN_ENVS=128 ROBOCASA_FLOW_ROUNDS=1
export ROBOCASA_FLOW_MAX_EPOCHS=30 ROBOCASA_FLOW_VAL_INTERVAL=10 ROBOCASA_FLOW_SAVE_INTERVAL=5
export ROBOCASA_FLOW_FULL_MODEL_EXPORT_STEPS='[30]'
# Future attempts only: keep initial/evaluation/teacher checkpoints plus the two
# newest complete recovery boundaries. Existing detached launches are immutable.
export ROBOCASA_FLOW_KEEP_LATEST_CHECKPOINTS=2
export ROBOCASA_FLOW_CANONICAL_COMPRESSION=gzip
export ROBOCASA_FLOW_SAMPLING_RECORD=full ROBOCASA_FLOW_ACTOR_BATCH_DROP='[]'
export ROBOCASA_FLOW_NOISE_LEVEL=0.3 ROBOCASA_FLOW_IGNORE_LAST=true
export ROBOCASA_SIM_PPO_GLOBAL_ESCROW=true ROBOCASA_FLOW_FRESH_EVAL_PROCESSES=true
export ROBOCASA_FLOW_MAX_ACTIONS=4100000
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl NCCL_P2P_DISABLE=1 FLUFFY_GPU_EXECUTION=baige
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 WANDB_MODE=disabled
# Preserve any scheduler CUDA mask. Some Baige images supply only NVIDIA UUIDs;
# use container ordinals only after proving that its entire inventory is assigned.
if [[ ${CUDA_VISIBLE_DEVICES+x} != x ]]; then
  "$TASK_PY" -B - <<'PY'
import os,subprocess
assigned=os.environ['NVIDIA_VISIBLE_DEVICES'].split(',')
visible=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
assert len(assigned)==len(set(assigned))==len(visible)==8
assert all(x.startswith('GPU-') for x in assigned) and set(assigned)==set(visible)
PY
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fi
TASK_STARTED=$(date +%s)
export TASK_DEADLINE=$((TASK_STARTED+TASK_CHECKPOINT_SECONDS+600))
[[ ! -e $ROBOCASA_SIM_PPO_OUTPUT ]]
mkdir -p "$ROBOCASA_SIM_PPO_OUTPUT"
exec > >(tee "$ROBOCASA_SIM_PPO_OUTPUT/console.log") 2>&1
TASK_WATCHER=
finish() {
  TASK_RC=$?
  trap - EXIT
  if [[ -n $TASK_WATCHER ]]; then kill "$TASK_WATCHER" 2>/dev/null || true; fi
  TASK_ENDED=$(date +%s)
  printf '{"exit_code":%s,"started_unix":%s,"ended_unix":%s,"worker_GPU_hours":%.8f}\n' \
    "$TASK_RC" "$TASK_STARTED" "$TASK_ENDED" \
    "$(awk -v s="$TASK_STARTED" -v e="$TASK_ENDED" 'BEGIN {print 8*(e-s)/3600}')" \
    > "$ROBOCASA_SIM_PPO_OUTPUT/EXIT.json"
  exit "$TASK_RC"
}
trap finish EXIT
"$TASK_PY" -B - "$TASK_PIN" <<'PY'
import hashlib,json,os,shutil,socket,sys,time
from pathlib import Path
p=Path(os.environ['ROBOCASA_SIM_PPO_OUTPUT'])
assert shutil.disk_usage(p).free >= 270*2**30, 'Need 70 GiB source allowance above the shared 200 GiB floor'
cpu=Path('/sys/fs/cgroup/cpu.max').read_text().split()
assert cpu[0]=='max' or int(cpu[0])/int(cpu[1])>=64
memory=Path('/sys/fs/cgroup/memory.max').read_text().strip()
assert memory=='max' or int(memory)>=1024*2**30
assert shutil.disk_usage('/dev/shm').total>=32*2**30
probe=p/'.write-probe';probe.write_text('source-pilot');assert probe.read_text()=='source-pilot';probe.unlink()
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
model=sha(Path(os.environ['ROBOCASA_PI05_TORCH_MODEL'])/'model.safetensors')
norm=sha(os.environ['ROBOCASA_PI05_NATIVE_NORM'])
assert model=='661bf172a2ae9dfca75216697eeaa543408bcbdfa53c98132759b93e0e8993a5'
assert norm=='4aed1af411bd0e0f49d2b0e6d6832b11b8682917231a07173fbc30fd493bbdee'
(p/'MANIFEST.json').write_text(json.dumps(dict(source=sys.argv[1],hostname=socket.gethostname(),
    task=os.environ['ROBOCASA_FLOW_TASK'],model_sha256=model,norm_sha256=norm,
    maximum_job_GPU_hours=int(os.environ['TASK_MAXIMUM_GPU_HOURS']),
    resumed_step=int(os.environ['TASK_INITIAL_STEP']),target_iterations=30,
    additional_train_episodes=(30-int(os.environ['TASK_INITIAL_STEP']))*128,
    additional_train_controls=(30-int(os.environ['TASK_INITIAL_STEP']))*128*600,
    free_bytes=shutil.disk_usage(p).free,checked_unix=time.time()),indent=2)+'\n')
PY
"$TASK_PY" -B "$TASK_CODE/scripts/[train]gpu_probe.py" --expected-count 8 --purpose training \
  --baige-container --baige-exclusive-training --output "$ROBOCASA_SIM_PPO_OUTPUT/gpu-probe.json"
(
  while sleep 20; do
    TASK_FREE=$(df -Pk "$ROBOCASA_SIM_PPO_OUTPUT" | awk 'NR==2 {print $4}')
    if (( TASK_FREE < 230686720 || $(date +%s) >= TASK_STARTED+TASK_CHECKPOINT_SECONDS )); then
      touch "$ROBOCASA_SIM_PPO_OUTPUT/CHECKPOINT_AND_EXIT"
      break
    fi
  done
) &
TASK_WATCHER=$!
timeout --signal=TERM --kill-after=30s "$((TASK_CHECKPOINT_SECONDS+700))" "$TASK_PY" -B \
  "$TASK_CODE/benchmarks/robocasa/train_flow_ppo.py" \
  --config-name robocasa365_pi05_flow_sde_ppo_reuse4
