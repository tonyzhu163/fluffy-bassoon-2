#!/usr/bin/env bash
set -Eeuo pipefail
: "${TASK_META:?}" "${TASK_CODE_PIN:?}" "${RANK:?}" "${WORLD_SIZE:?}"
export TASK_STARTED=$(date +%s)
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_OUT=$TASK_ROOT/outputs/robocasa-interface-render-qualification-20261102-r1
TASK_PY=$TASK_ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
test ! -e "$TASK_OUT"
mkdir -p "$TASK_OUT/job"
export PYTHONPATH=$TASK_META/src:$TASK_ROOT/sources/robocasa365-public-pi05-nnx-leaves-20261101-r1/RLinf
export RLINF_PATH=$TASK_ROOT/sources/robocasa365-public-pi05-nnx-leaves-20261101-r1/RLinf
export ROBOCASA_ASSETS_PATH=$TASK_ROOT/setups/robocasa365-native-20261101-r1/assets
export OPENPI_DATA_HOME=$TASK_ROOT/setups/robocasa365-native-20261101-r1/cache/.cache/openpi
export ROBOCASA_PI05_TORCH_MODEL=$TASK_ROOT/inputs/robocasa365-public-pi05-c484-torch-20261101-r2
export ROBOCASA_PI05_NATIVE_NORM=$TASK_ROOT/inputs/robocasa365-public-pi05-c484-75000-20261101-r1/assets/norm_stats.json
export ROBOCASA_NATIVE_OUTPUT=$TASK_OUT/native
export FLUFFY_GPU_EXECUTION=baige NCCL_P2P_DISABLE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
exec > >(tee "$TASK_OUT/job/console.log") 2>&1
finish() {
 TASK_RC=$?; trap - EXIT
 TASK_FINISHED=$(date +%s)
 printf '{"started_unix":%s,"finished_unix":%s,"allocated_gpus":1,"observed_gpu_seconds":%s,"exit_code":%s}\n' \
  "$TASK_STARTED" "$TASK_FINISHED" "$((TASK_FINISHED-TASK_STARTED))" "$TASK_RC" > "$TASK_OUT/job/clock.json"
 exit "$TASK_RC"
}
trap finish EXIT
timeout --signal=TERM --kill-after=20s 480 "$TASK_PY" -B \
 "$TASK_META/outputs/robocasa-interface-render-qualification-20261102-r1/RUN.py"
