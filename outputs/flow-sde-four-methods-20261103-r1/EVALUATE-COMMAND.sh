#!/usr/bin/env bash
set -Eeuo pipefail
: "${TASK_FLOW_CODE:?}" "${TASK_CODE_PIN:?}" "${TASK_EVAL_METHOD:?}" "${TASK_EVAL_MODE:?}" "${TASK_EVAL_ATTEMPT:?}" "${TASK_EVAL_ADMISSION:?}" "${TASK_CAMPAIGN_SHA:?}" "${TASK_EVAL_SECONDS:?}" "${TASK_CASE_LIMIT:?}" "${MASTER_ADDR:?}" "${RANK:?}" "${WORLD_SIZE:?}"
test "$RANK" = 0
test "$WORLD_SIZE" = 1
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_PY=$TASK_ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
TASK_RUN=$TASK_ROOT/outputs/flow-four-method-readout-20261103-r1
TASK_LOG=$TASK_RUN/job-logs/$TASK_EVAL_METHOD-$TASK_EVAL_MODE-$TASK_EVAL_ATTEMPT
TASK_STARTED=$(date +%s)
test ! -e "$TASK_LOG"
mkdir -p "$TASK_LOG"
exec > >(tee "$TASK_LOG/console.log") 2>&1
finish() {
  TASK_RC=$?; trap - EXIT
  TASK_ENDED=$(date +%s)
  printf '{"started_unix":%s,"ended_unix":%s,"process_gpu_seconds":%s,"gpu_count":1,"exit_code":%s,"scheduler_release_not_included":true}\n' \
    "$TASK_STARTED" "$TASK_ENDED" "$((TASK_ENDED-TASK_STARTED))" "$TASK_RC" > "$TASK_LOG/clock.json"
  exit "$TASK_RC"
}
trap finish EXIT
export RLINF_PATH=$TASK_ROOT/sources/robocasa365-public-pi05-nnx-leaves-20261101-r1/RLinf
export PYTHONPATH=$TASK_FLOW_CODE/src:$RLINF_PATH
export ROBOCASA_ASSETS_PATH=$TASK_ROOT/setups/robocasa365-native-20261101-r1/assets
export OPENPI_DATA_HOME=$TASK_ROOT/setups/robocasa365-native-20261101-r1/cache/.cache/openpi
export FLUFFY_GPU_EXECUTION=baige MUJOCO_GL=egl PYOPENGL_PLATFORM=egl NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 WANDB_MODE=disabled
test -f /mnt/bos/adgfyl/strong-plant-archives/20261103-joint-steering-owned-flow-checkpoints-r1/checkpoint-00/actor/model_state_dict/full_weights.pt
timeout --signal=TERM --kill-after=20s "$((TASK_EVAL_SECONDS-30))s" "$TASK_PY" -B \
  "$TASK_FLOW_CODE/outputs/flow-sde-four-methods-20261103-r1/EVALUATE.py" \
  --code "$TASK_FLOW_CODE" --source-commit "$TASK_CODE_PIN" \
  --campaign "$TASK_FLOW_CODE/outputs/flow-sde-four-methods-20261103-r1/CAMPAIGN.json" \
  --campaign-sha256 "$TASK_CAMPAIGN_SHA" --admission "$TASK_EVAL_ADMISSION" \
  --output "$TASK_RUN" --method "$TASK_EVAL_METHOD" --mode "$TASK_EVAL_MODE" \
  --attempt "$TASK_EVAL_ATTEMPT" --expected-hostname "$MASTER_ADDR" --case-limit "$TASK_CASE_LIMIT"
