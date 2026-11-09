#!/usr/bin/env bash
set -euo pipefail
TASK_CODE=$1
TASK_PIN=$2
TASK_BASE=/mowenjie/fluffy-bassoon-2/outputs/robocasa-sft-reproduction-20261109-r1
TASK_OUTPUT=$TASK_BASE/${3:?fresh campaign identity required}
TASK_PY=/mowenjie/fluffy-bassoon-2/setups/robocasa365-native-20261101-r1/env/bin/python
TASK_MANIFEST_NAME=${4:-CAMPAIGN-MANIFEST.json}
[[ $TASK_MANIFEST_NAME =~ ^[A-Z0-9-]+\.json$ ]]
TASK_MANIFEST=$TASK_CODE/outputs/robocasa-sft-reproduction-20261109-r1/$TASK_MANIFEST_NAME
read -r TASK_SIM_FOLDER TASK_SIM_PIN < <("$TASK_PY" - "$TASK_MANIFEST" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
print(p.get('robocasa_source_directory','robocasa'),
      p.get('robocasa_commit','29f7ce8814c1547f5af762a0997fbd4b64848dd7'))
PY
)
[[ $TASK_SIM_FOLDER =~ ^robocasa(-v100)?$ ]]
[[ $TASK_SIM_PIN =~ ^[0-9a-f]{40}$ ]]
test "$(git -C "$TASK_CODE" rev-parse HEAD)" = "$TASK_PIN"
git -C "$TASK_CODE" diff --quiet
test "$(git -C "$TASK_BASE/sources/openpi" rev-parse HEAD)" = ca4c6d710db75e276bc7c866a57bd7e4aee5b6e8
test "$(git -C "$TASK_BASE/sources/$TASK_SIM_FOLDER" rev-parse HEAD)" = "$TASK_SIM_PIN"
export FLUFFY_GPU_EXECUTION=baige MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70 OPENPI_DATA_HOME="$TASK_BASE/openpi-cache"
export JAX_COMPILATION_CACHE_DIR="$TASK_BASE/jax-cache"
export PYTHONPATH="$TASK_BASE/sources/openpi/src:$TASK_BASE/sources/openpi/packages/openpi-client/src:$TASK_BASE/sources/$TASK_SIM_FOLDER:$TASK_CODE/src"
if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
  [[ $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) -eq 1 ]]
  export CUDA_VISIBLE_DEVICES=0
fi
exec "$TASK_PY" -B "$TASK_CODE/outputs/robocasa-sft-reproduction-20261109-r1/campaign_worker.py" \
  --source "$TASK_CODE" --pin "$TASK_PIN" --output "$TASK_OUTPUT" \
  --manifest "$TASK_MANIFEST"
