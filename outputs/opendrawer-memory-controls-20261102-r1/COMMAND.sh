#!/usr/bin/env bash
set -Eeuo pipefail
: "${TASK_CODE:?}" "${TASK_PIN:?}" "${RANK:?}" "${WORLD_SIZE:?}"
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_OUT=${TASK_MEMORY_OUTPUT:-$TASK_ROOT/outputs/opendrawer-memory-controls-20261102-r1}
TASK_PY=$TASK_ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
TASK_STARTED=$(date +%s)
TASK_LIMIT_SECONDS=${TASK_MEMORY_LIMIT_SECONDS:-10800}
TASK_LOCAL_GPUS=${TASK_MEMORY_GPUS_PER_NODE:-2}
test "$TASK_LIMIT_SECONDS" -le 10800
TASK_DEADLINE=$((TASK_STARTED+TASK_LIMIT_SECONDS-300))
export PYTHONPATH=$TASK_CODE/src:$TASK_ROOT/sources/robocasa365-public-pi05-nnx-leaves-20261101-r1/RLinf
export RLINF_PATH=$TASK_ROOT/sources/robocasa365-public-pi05-nnx-leaves-20261101-r1/RLinf
export ROBOCASA_ASSETS_PATH=$TASK_ROOT/setups/robocasa365-native-20261101-r1/assets
export OPENPI_DATA_HOME=$TASK_ROOT/setups/robocasa365-native-20261101-r1/cache/.cache/openpi
export ROBOCASA_NATIVE_OUTPUT=$TASK_OUT/native/node-$RANK
export ROBOCASA_PI05_TORCH_MODEL=$TASK_ROOT/inputs/robocasa365-public-pi05-c484-torch-20261101-r2
export ROBOCASA_PI05_NATIVE_NORM=$TASK_ROOT/inputs/robocasa365-public-pi05-c484-75000-20261101-r1/assets/norm_stats.json
export FLUFFY_GPU_EXECUTION=baige NCCL_P2P_DISABLE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
test "$(git -C "$TASK_CODE" rev-parse HEAD)" = "$TASK_PIN"
mkdir -p "$TASK_OUT/logs"
exec > >(tee "$TASK_OUT/logs/node-$RANK.log") 2>&1
finish() {
 TASK_RC=$?
 trap - EXIT
 TASK_FINISHED=$(date +%s)
 printf '{"started_unix":%s,"finished_unix":%s,"allocated_gpus":%s,"gpu_seconds":%s,"exit_code":%s}\n' \
  "$TASK_STARTED" "$TASK_FINISHED" "$TASK_LOCAL_GPUS" "$(((TASK_FINISHED-TASK_STARTED)*TASK_LOCAL_GPUS))" "$TASK_RC" > "$TASK_OUT/logs/node-$RANK-clock.json"
 exit "$TASK_RC"
}
trap finish EXIT
timeout --signal=TERM --kill-after=60s "$((TASK_LIMIT_SECONDS-180))" "$TASK_PY" -B \
 "$TASK_CODE/outputs/opendrawer-memory-controls-20261102-r1/CONTROL.py" --deadline "$TASK_DEADLINE"
if [ "$RANK" = 0 ]; then
 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 "$TASK_PY" -B \
  "$TASK_CODE/outputs/opendrawer-memory-controls-20261102-r1/SUMMARIZE.py"
fi
