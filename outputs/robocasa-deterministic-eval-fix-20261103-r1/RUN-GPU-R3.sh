#!/usr/bin/env bash
set -Eeuo pipefail
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_CODE=${1:?pass exact detached code directory}
TASK_PIN=${2:?pass exact reviewed Git commit}
TASK_RUN=$TASK_ROOT/outputs/robocasa-deterministic-eval-fix-20261103-r1/gpu-r3
TASK_PY=$TASK_ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
test ! -e "$TASK_RUN"
mkdir "$TASK_RUN"
exec > >(tee "$TASK_RUN/console.log") 2>&1
TASK_STARTED=$(date +%s)
finish() {
  TASK_EXIT=$?
  trap - EXIT
  TASK_ENDED=$(date +%s)
  printf '{"started_unix":%s,"ended_unix":%s,"GPU_seconds":%s,"exit_code":%s,"allocated_GPUs":1}\n' \
    "$TASK_STARTED" "$TASK_ENDED" "$((TASK_ENDED-TASK_STARTED))" "$TASK_EXIT" > "$TASK_RUN/clock.json"
  exit "$TASK_EXIT"
}
trap finish EXIT
test "$(git -C "$TASK_CODE" rev-parse HEAD)" = "$TASK_PIN"
test -z "$(git -C "$TASK_CODE" status --porcelain)"
export RLINF_PATH=$TASK_ROOT/sources/robocasa365-public-pi05-nnx-leaves-20261101-r1/RLinf
export PYTHONPATH=$TASK_CODE/src:$RLINF_PATH
export ROBOCASA_ASSETS_PATH=$TASK_ROOT/setups/robocasa365-native-20261101-r1/assets
export OPENPI_DATA_HOME=$TASK_ROOT/setups/robocasa365-native-20261101-r1/cache/.cache/openpi
export ROBOCASA_NATIVE_OUTPUT=$TASK_RUN/native-runtime
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1 FLUFFY_GPU_EXECUTION=baige
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export LYCO_BAIGE_GPU_MAP=$TASK_RUN/GPU-PROBE.json
"$TASK_PY" -B "$TASK_CODE/scripts/[train]gpu_probe.py" --expected-count 1 \
  --output "$LYCO_BAIGE_GPU_MAP" --baige-container --purpose evaluation
timeout --signal=TERM --kill-after=30s 840s "$TASK_PY" -B \
  "$TASK_CODE/outputs/robocasa-deterministic-eval-fix-20261103-r1/qualify_native.py" \
  --output "$TASK_RUN/qualification" --code "$TASK_CODE"
