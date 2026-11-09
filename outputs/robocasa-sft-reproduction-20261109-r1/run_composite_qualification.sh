#!/usr/bin/env bash
set -euo pipefail
TASK_CODE=$1
TASK_PIN=$2
TASK_BASE=/mowenjie/fluffy-bassoon-2/outputs/robocasa-sft-reproduction-20261109-r1
TASK_OUT=$TASK_BASE/${3:?fresh output identity required}
TASK_PY=/mowenjie/fluffy-bassoon-2/setups/robocasa365-native-20261101-r1/env/bin/python
TASK_CHECKPOINT=/mowenjie/fluffy-bassoon-2/inputs/robocasa365-public-pi05-c484-75000-20261101-r1
TASK_SCRIPTS=$TASK_CODE/outputs/robocasa-sft-reproduction-20261109-r1
TASK_VERSION=${4:-1.0.1}
case "$TASK_VERSION" in
  1.0.1) TASK_SIM_FOLDER=robocasa; TASK_SIM_PIN=29f7ce8814c1547f5af762a0997fbd4b64848dd7 ;;
  1.0.0) TASK_SIM_FOLDER=robocasa-v100; TASK_SIM_PIN=8f3c96ec8d1bfcd8126cad2bca887da98d30e997 ;;
  *) exit 2 ;;
esac
test "$(git -C "$TASK_BASE/sources/openpi" rev-parse HEAD)" = ca4c6d710db75e276bc7c866a57bd7e4aee5b6e8
test "$(git -C "$TASK_BASE/sources/$TASK_SIM_FOLDER" rev-parse HEAD)" = "$TASK_SIM_PIN"
test "$(git -C "$TASK_CODE" rev-parse HEAD)" = "$TASK_PIN"
git -C "$TASK_CODE" diff --quiet
mkdir "$TASK_OUT"
TASK_STARTED=$(date +%s)
TASK_SERVER=''
TASK_CLIENTS=()
cleanup() {
  TASK_EXIT=$?
  for TASK_CHILD in "${TASK_CLIENTS[@]}"; do kill -TERM "$TASK_CHILD" 2>/dev/null || true; done
  if [[ -n $TASK_SERVER ]]; then kill -TERM "$TASK_SERVER" 2>/dev/null || true; wait "$TASK_SERVER" || true; fi
  printf '{"exit_code":%s,"start_unix":%s,"end_unix":%s,"allocated_gpu_count":1}\n' "$TASK_EXIT" "$TASK_STARTED" "$(date +%s)" > "$TASK_OUT/EXIT.json"
}
trap cleanup EXIT
exec > >(tee "$TASK_OUT/console.log") 2>&1
export FLUFFY_GPU_EXECUTION=baige MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70 OPENPI_DATA_HOME="$TASK_BASE/openpi-cache"
export JAX_COMPILATION_CACHE_DIR="$TASK_BASE/jax-cache"
export PYTHONPATH="$TASK_BASE/sources/openpi/src:$TASK_BASE/sources/openpi/packages/openpi-client/src:$TASK_BASE/sources/$TASK_SIM_FOLDER:$TASK_CODE/src"
if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
  [[ $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) -eq 1 ]]
  export CUDA_VISIBLE_DEVICES=0
fi
"$TASK_PY" -B "$TASK_CODE/scripts/[train]gpu_probe.py" --expected-count 1 --purpose evaluation --baige-container --output "$TASK_OUT/GPU-PROBE.json"
"$TASK_PY" -B "$TASK_SCRIPTS/serve_native.py" --checkpoint "$TASK_CHECKPOINT" --output "$TASK_OUT/server" --port 18405 > "$TASK_OUT/server.log" 2>&1 &
TASK_SERVER=$!
while [[ ! -f $TASK_OUT/server/READY.json ]]; do
  kill -0 "$TASK_SERVER"
  [[ $(date +%s) -lt $((TASK_STARTED+900)) ]]
  sleep 2
done
for TASK_NAME in SearingMeat ArrangeBreadBasket WaffleReheat PackIdenticalLunches; do
  "$TASK_PY" -B "$TASK_SCRIPTS/reference_eval.py" --checkpoint "$TASK_CHECKPOINT" \
    --output "$TASK_OUT/$TASK_NAME" --tasks "$TASK_NAME" --trials 1 \
    --maximum-seconds 1500 --version "$TASK_VERSION" --video-episodes 0 --replan 5 --server-port 18405 > "$TASK_OUT/$TASK_NAME.log" 2>&1 &
  TASK_CLIENTS+=("$!")
done
TASK_FAILURE=0
for TASK_CHILD in "${TASK_CLIENTS[@]}"; do wait "$TASK_CHILD" || TASK_FAILURE=1; done
TASK_CLIENTS=()
exit "$TASK_FAILURE"
