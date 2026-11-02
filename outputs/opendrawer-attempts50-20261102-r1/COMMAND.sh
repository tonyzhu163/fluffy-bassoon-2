#!/usr/bin/env bash
set -Eeuo pipefail
: "${TASK_CURVE_CODE:?}" "${TASK_CURVE_OUTPUT:?}" "${TASK_CURVE_METHOD:?}" "${TASK_ATTEMPT:?}" "${TASK_BINDING_SHA:?}" "${RANK:?}"
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_PY=$TASK_ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
TASK_STARTED=$(date +%s)
TASK_DEADLINE=$((TASK_STARTED+27900))
export TASK_ATTEMPT_OUT=$TASK_CURVE_OUTPUT/attempts/$TASK_ATTEMPT/node-$(printf '%02d' "$RANK")
export PYTHONPATH=$TASK_ROOT/worktrees/opendrawer-ad-dpt-20261102-r1/src:$TASK_ROOT/sources/robocasa365-public-pi05-nnx-leaves-20261101-r1/RLinf
export RLINF_PATH=$TASK_ROOT/sources/robocasa365-public-pi05-nnx-leaves-20261101-r1/RLinf
export ROBOCASA_ASSETS_PATH=$TASK_ROOT/setups/robocasa365-native-20261101-r1/assets
export OPENPI_DATA_HOME=$TASK_ROOT/setups/robocasa365-native-20261101-r1/cache/.cache/openpi
export ROBOCASA_NATIVE_OUTPUT=$TASK_ATTEMPT_OUT/native
export ROBOCASA_PI05_TORCH_MODEL=$TASK_ROOT/inputs/robocasa365-public-pi05-c484-torch-20261101-r2
export ROBOCASA_PI05_NATIVE_NORM=$TASK_ROOT/inputs/robocasa365-public-pi05-c484-75000-20261101-r1/assets/norm_stats.json
export FLUFFY_GPU_EXECUTION=baige NCCL_P2P_DISABLE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
mkdir -p "$TASK_CURVE_OUTPUT/attempts/$TASK_ATTEMPT"
exec > >(tee "$TASK_CURVE_OUTPUT/attempts/$TASK_ATTEMPT/node-$RANK-console.log") 2>&1
finish() {
 TASK_RC=$?
 trap - EXIT
 TASK_FINISHED=$(date +%s)
 printf '{"started_unix":%s,"finished_unix":%s,"allocated_gpus":2,"observed_gpu_seconds":%s,"exit_code":%s}\n' \
  "$TASK_STARTED" "$TASK_FINISHED" "$(((TASK_FINISHED-TASK_STARTED)*2))" "$TASK_RC" > "$TASK_ATTEMPT_OUT/clock.json"
 exit "$TASK_RC"
}
trap finish EXIT
timeout --signal=TERM --kill-after=60s 28020 "$TASK_PY" -B \
 "$TASK_CURVE_CODE/outputs/opendrawer-attempts50-20261102-r1/ATTEMPTS.py" --deadline "$TASK_DEADLINE"
