#!/usr/bin/env bash
set -Eeuo pipefail
: "${TASK_FLOW_CODE:?}" "${TASK_CODE_PIN:?}" "${MASTER_ADDR:?}" "${RANK:?}" "${WORLD_SIZE:?}"
test "$RANK" = 0
test "$WORLD_SIZE" = 1
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_PY=$TASK_ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
TASK_OUT=$TASK_ROOT/outputs/flow-four-method-source-gpu-20261103-r2
TASK_STARTED=$(date +%s)
test ! -e "$TASK_OUT"
mkdir -p "$TASK_OUT"
exec > >(tee "$TASK_OUT/job-console.log") 2>&1
finish() {
  TASK_RC=$?
  trap - EXIT
  TASK_ENDED=$(date +%s)
  printf '{"started_unix":%s,"ended_unix":%s,"process_gpu_seconds":%s,"gpu_count":1,"exit_code":%s,"scheduler_release_not_included":true}\n' \
    "$TASK_STARTED" "$TASK_ENDED" "$((TASK_ENDED-TASK_STARTED))" "$TASK_RC" > "$TASK_OUT/clock.json"
  exit "$TASK_RC"
}
trap finish EXIT
export RLINF_PATH=$TASK_ROOT/sources/robocasa365-public-pi05-nnx-leaves-20261101-r1/RLinf
export PYTHONPATH=$TASK_FLOW_CODE/src:$RLINF_PATH
export ROBOCASA_ASSETS_PATH=$TASK_ROOT/setups/robocasa365-native-20261101-r1/assets
export OPENPI_DATA_HOME=$TASK_ROOT/setups/robocasa365-native-20261101-r1/cache/.cache/openpi
export FLUFFY_GPU_EXECUTION=baige MUJOCO_GL=egl PYOPENGL_PLATFORM=egl NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 WANDB_MODE=disabled
# Baige exposes this scheduled container's sole accelerator at CUDA index0.
# Keep an explicit scheduler selection when present; the native probe still
# checks expected-count, exclusive ownership, CUDA/PCI/UUID/PID agreement.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
timeout --signal=TERM --kill-after=20s 2350s "$TASK_PY" -B \
  "$TASK_FLOW_CODE/outputs/flow-sde-four-methods-20261103-r1/PREPARE.py" \
  --code "$TASK_FLOW_CODE" --source-commit "$TASK_CODE_PIN" --output "$TASK_OUT/preparation" \
  --expected-hostname "$MASTER_ADDR" --max-seconds 2300
