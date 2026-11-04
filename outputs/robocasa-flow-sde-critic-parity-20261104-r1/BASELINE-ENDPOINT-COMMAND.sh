#!/usr/bin/env bash
set -Eeuo pipefail
: "${TASK_META:?}" "${TASK_CODE_PIN:?}"
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_RUN=$TASK_ROOT/outputs/robocasa-flow-sde-critic-parity-20261104-r1
TASK_RUNTIME=$TASK_ROOT/sources/flow10m-0988b5f9
TASK_VENDOR=$TASK_ROOT/sources/robocasa365-public-pi05-nnx-leaves-20261101-r1/RLinf
TASK_PY=$TASK_ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
TASK_OUTPUT=$TASK_RUN/baseline-endpoint-r2/gpu
TASK_STARTED=$(date +%s)
test ! -e "$TASK_OUTPUT"
mkdir -p "$TASK_OUTPUT"
exec > >(tee "$TASK_OUTPUT/console.log") 2>&1
finish() {
  TASK_RC=$?
  trap - EXIT
  TASK_FINISHED=$(date +%s)
  printf '{"started_unix":%s,"finished_unix":%s,"allocated_gpus":1,"observed_gpu_seconds":%s,"exit_code":%s}\n' \
    "$TASK_STARTED" "$TASK_FINISHED" "$((TASK_FINISHED-TASK_STARTED))" "$TASK_RC" > "$TASK_OUTPUT/clock.json"
  exit "$TASK_RC"
}
trap finish EXIT
test "$(git -C "$TASK_META" rev-parse HEAD)" = "$TASK_CODE_PIN"
test -z "$(git -C "$TASK_META" status --porcelain)"
test "$(git -C "$TASK_RUNTIME" rev-parse HEAD)" = 0988b5f9e9bf31ac3db673cc2860959611b9ae12
test -z "$(git -C "$TASK_RUNTIME" status --porcelain)"
test "$(git -C "$TASK_VENDOR" rev-parse HEAD)" = 8a1233c3fae6f5905adfda7684c455842b0ecfb8
test -z "$(git -C "$TASK_VENDOR" status --porcelain)"
export PYTHONPATH=$TASK_META/outputs/robocasa-flow-sde-critic-parity-20261104-r1:$TASK_META/outputs/robocasa-flow-sde-checkpoint-selection-20261104-r1:$TASK_RUNTIME/src:$TASK_VENDOR
export OPENPI_DATA_HOME=$TASK_ROOT/setups/robocasa365-native-20261101-r1/cache/.cache/openpi
export ROBOCASA_ASSETS_PATH=$TASK_ROOT/setups/robocasa365-native-20261101-r1/assets
export NCCL_P2P_DISABLE=1 FLUFFY_GPU_EXECUTION=baige CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 WANDB_MODE=disabled
"$TASK_PY" -B - "$TASK_RUN" <<'PY'
import json,pathlib,sys
r=pathlib.Path(sys.argv[1]);b=json.loads((r/'RECOVERY-ADMISSION.json').read_text())
assert b['completed'] and b['prior_conservative_gpu_seconds']+10560+1200<=21600
assert json.loads((r/'comparison-input-r2/RESTORED.json').read_text())['completed']
assert json.loads((r/'BASELINE-ENDPOINT-ADMISSION.json').read_text())['completed']
PY
"$TASK_PY" -B "$TASK_META/scripts/[train]gpu_probe.py" --expected-count 1 --purpose evaluation \
  --baige-container --output "$TASK_OUTPUT/gpu-probe.json"
TASK_DEADLINE=$((TASK_STARTED+1050))
timeout --signal=TERM --kill-after=30s 1080 "$TASK_PY" -B -c \
  'from baseline_endpoint import main; main()' --root "$TASK_RUN" --output "$TASK_OUTPUT" --deadline "$TASK_DEADLINE"
