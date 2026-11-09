#!/usr/bin/env bash
set -euo pipefail
TASK_CODE=$1
TASK_PIN=$2
TASK_BASE=/mowenjie/fluffy-bassoon-2/outputs/robocasa-sft-reproduction-20261109-r1
TASK_OUTPUT=$TASK_BASE/${3:?fresh campaign identity required}
TASK_PY=/mowenjie/fluffy-bassoon-2/setups/robocasa365-native-20261101-r1/env/bin/python
test "$(git -C "$TASK_CODE" rev-parse HEAD)" = "$TASK_PIN"
git -C "$TASK_CODE" diff --quiet
test "$(git -C "$TASK_BASE/sources/openpi" rev-parse HEAD)" = ca4c6d710db75e276bc7c866a57bd7e4aee5b6e8
test "$(git -C "$TASK_BASE/sources/robocasa" rev-parse HEAD)" = 29f7ce8814c1547f5af762a0997fbd4b64848dd7
export FLUFFY_GPU_EXECUTION=baige MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70 OPENPI_DATA_HOME="$TASK_BASE/openpi-cache"
export JAX_COMPILATION_CACHE_DIR="$TASK_BASE/jax-cache"
export PYTHONPATH="$TASK_BASE/sources/openpi/src:$TASK_BASE/sources/openpi/packages/openpi-client/src:$TASK_BASE/sources/robocasa:$TASK_CODE/src"
if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
  [[ $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) -eq 1 ]]
  export CUDA_VISIBLE_DEVICES=0
fi
exec "$TASK_PY" -B "$TASK_CODE/outputs/robocasa-sft-reproduction-20261109-r1/campaign_worker.py" \
  --source "$TASK_CODE" --pin "$TASK_PIN" --output "$TASK_OUTPUT" \
  --manifest "$TASK_CODE/outputs/robocasa-sft-reproduction-20261109-r1/CAMPAIGN-MANIFEST.json"
