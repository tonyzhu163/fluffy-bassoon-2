#!/usr/bin/env bash
set -Eeuo pipefail
: "${TASK_FLOW_CODE:?}" "${TASK_CODE_PIN:?}" "${TASK_FIT_ATTEMPT:?}" "${MASTER_ADDR:?}" "${RANK:?}" "${WORLD_SIZE:?}"
test "$RANK" = 0
test "$WORLD_SIZE" = 1
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_PY=$TASK_ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
TASK_FITS=$TASK_ROOT/outputs/flow-four-method-fits-20261103-r1
TASK_LOGS=$TASK_FITS/allocation-$TASK_FIT_ATTEMPT
TASK_STARTED=$(date +%s)
test ! -e "$TASK_LOGS"
mkdir -p "$TASK_LOGS"
exec > >(tee "$TASK_LOGS/console.log") 2>&1
finish() {
  TASK_RC=$?; trap - EXIT
  TASK_ENDED=$(date +%s)
  printf '{"started_unix":%s,"ended_unix":%s,"process_gpu_seconds":%s,"gpu_count":1,"exit_code":%s,"scheduler_release_not_included":true}\n' \
    "$TASK_STARTED" "$TASK_ENDED" "$((TASK_ENDED-TASK_STARTED))" "$TASK_RC" > "$TASK_LOGS/clock.json"
  exit "$TASK_RC"
}
trap finish EXIT
export PYTHONPATH=$TASK_FLOW_CODE/src
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 WANDB_MODE=disabled
timeout --signal=TERM --kill-after=30s 3500s "$TASK_PY" -B \
  "$TASK_FLOW_CODE/outputs/flow-sde-four-methods-20261103-r1/FITS.py" \
  --assembly "$TASK_ROOT/outputs/flow-four-method-source-gpu-20261103-r2/preparation/ASSEMBLY.json" \
  --output "$TASK_FITS" --source-commit "$TASK_CODE_PIN" --expected-hostname "$MASTER_ADDR" \
  --attempt "$TASK_FIT_ATTEMPT"
