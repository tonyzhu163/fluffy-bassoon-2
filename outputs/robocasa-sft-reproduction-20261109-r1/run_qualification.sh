#!/usr/bin/env bash
set -euo pipefail
TASK_CODE=$1
TASK_PIN=$2
TASK_BASE=/mowenjie/fluffy-bassoon-2/outputs/robocasa-sft-reproduction-20261109-r1
TASK_PY=/mowenjie/fluffy-bassoon-2/setups/robocasa365-native-20261101-r1/env/bin/python
TASK_OUT=$TASK_BASE/gpu01
test "$(git -C "$TASK_CODE" rev-parse HEAD)" = "$TASK_PIN"
git -C "$TASK_CODE" diff --quiet
test "$(git -C "$TASK_BASE/sources/openpi" rev-parse HEAD)" = ca4c6d710db75e276bc7c866a57bd7e4aee5b6e8
test "$(git -C "$TASK_BASE/sources/robocasa" rev-parse HEAD)" = 29f7ce8814c1547f5af762a0997fbd4b64848dd7
mkdir "$TASK_OUT"
TASK_STARTED=$(date +%s)
trap 'TASK_EXIT=$?; TASK_FINISHED=$(date +%s); printf "{\"exit_code\":%s,\"start_unix\":%s,\"end_unix\":%s,\"allocated_gpu_count\":1}\n" "$TASK_EXIT" "$TASK_STARTED" "$TASK_FINISHED" > "$TASK_OUT/EXIT.json"' EXIT
exec > >(tee "$TASK_OUT/console.log") 2>&1
export FLUFFY_GPU_EXECUTION=baige MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONDONTWRITEBYTECODE=1 MUJOCO_EGL_DEVICE_ID=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70
export JAX_COMPILATION_CACHE_DIR="$TASK_BASE/jax-cache"
export OPENPI_DATA_HOME="$TASK_BASE/openpi-cache"
export PYTHONPATH="$TASK_BASE/sources/openpi/src:$TASK_BASE/sources/openpi/packages/openpi-client/src:$TASK_BASE/sources/robocasa:$TASK_CODE/src"
"$TASK_PY" -B "$TASK_CODE/scripts/[train]gpu_probe.py" --expected-count 1 \
  --purpose evaluation --baige-container --output "$TASK_OUT/GPU-PROBE.json"
timeout --signal=TERM --kill-after=45 3240 "$TASK_PY" -B \
  "$TASK_CODE/outputs/robocasa-sft-reproduction-20261109-r1/reference_eval.py" \
  --checkpoint /mowenjie/fluffy-bassoon-2/inputs/robocasa365-public-pi05-c484-75000-20261101-r1 \
  --output "$TASK_OUT/eval" --tasks OpenDrawer CloseToasterOvenDoor CoffeeSetupMug \
  --trials 1 --maximum-seconds 3000 --version 1.0.1 --replan 5
