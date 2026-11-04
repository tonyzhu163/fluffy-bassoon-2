#!/usr/bin/env bash
# Launch commands for the exploratory 3-task meta-learning baseline on Baige (1 GPU per task)
set -euo pipefail

ROOT=/mowenjie/fluffy-bassoon-2
PY=$ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
CODE=${TASK_CODE_DIR:-$ROOT/sources/ppo-grpo-meta-pipeline-ad918c9f}
COMMIT=$(git -C "$CODE" rev-parse HEAD)
RUN=$ROOT/outputs/robocasa-three-task-meta-baseline-20261104-r1
TASK=${ROBOCASA_META_TASK:-OpenDrawer}
MODE=${ROBOCASA_META_STAGE:-prepare}

export PYTHONPATH="$CODE/src:$ROOT/sources/critic-selected-a1229239-r1"

if [[ "$MODE" == "prepare" ]]; then
  exec "$PY" -B "$RUN/PREPARE-STREAM.py" \
    --task "$TASK" \
    --code "$CODE" \
    --source-commit "$COMMIT" \
    --output "$RUN/$TASK/preparation" \
    --expected-hostname "$(hostname)" \
    --max-seconds "${TASK_MAX_SECONDS:-10800}"
elif [[ "$MODE" == "fit" ]]; then
  ASSEMBLY="$RUN/$TASK/preparation/ASSEMBLY.json"
  ASSEMBLY_SHA=$(sha256sum "$ASSEMBLY" | awk '{print $1}')
  exec "$PY" -B "$RUN/FITS.py" \
    --assembly "$ASSEMBLY" \
    --assembly-sha256 "$ASSEMBLY_SHA" \
    --output "$RUN/$TASK/fits" \
    --source-commit "$COMMIT" \
    --expected-hostname "$(hostname)" \
    --attempt "${TASK_ATTEMPT:-r1}" \
    --max-elapsed-seconds "${TASK_MAX_SECONDS:-6600}"
else
  echo "Unknown ROBOCASA_META_STAGE=$MODE (expected prepare or fit)" >&2
  exit 2
fi
