#!/usr/bin/env bash
set -euo pipefail
ROOT=/mowenjie/fluffy-bassoon-2
PY=$ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
: "${TASK_CODE_DIR:?Set TASK_CODE_DIR to the clean reviewed, deployed commit worktree}"
CODE=$TASK_CODE_DIR
RUN=$ROOT/outputs/robocasa-three-task-meta-baseline-20261104-r2
export PYTHONPATH="$CODE/src:$ROOT/sources/critic-selected-a1229239-r1"
TASK=${ROBOCASA_META_TASK:?Set a source task for preparation, or joint for pooled fitting}
if [[ "$TASK" == "joint" ]]; then
  STAGE=fit
  WALL=${ROBOCASA_META_WALL_SECONDS:-28800}
else
  STAGE=prepare
  WALL=14400
fi
exec "$PY" -B "$CODE/outputs/robocasa-three-task-meta-baseline-20261104-r2/RUN.py" \
  --task "$TASK" --stage "$STAGE" \
  --output "$RUN/$ROBOCASA_META_TASK" \
  --attempt "${ROBOCASA_META_ATTEMPT:-${STAGE}-4gpu-01}" \
  --files-per-batch "${ROBOCASA_META_FILES_PER_BATCH:-4}" \
  --wall-seconds "$WALL" --prepare-seconds 12600
