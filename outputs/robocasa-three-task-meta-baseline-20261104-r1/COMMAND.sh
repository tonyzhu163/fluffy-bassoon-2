#!/usr/bin/env bash
# Launch commands for the exploratory 3-task meta-learning baseline on Baige (1 GPU per task)
set -euo pipefail

ROOT=/mowenjie/fluffy-bassoon-2
PY=$ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
CODE=${TASK_CODE_DIR:-$ROOT/sources/three-task-meta-baseline-f125e65c}
COMMIT=$(git -C "$CODE" rev-parse HEAD)
RUN=$ROOT/outputs/robocasa-three-task-meta-baseline-20261104-r1
TASK=${ROBOCASA_META_TASK:-OpenDrawer}
MODE=${ROBOCASA_META_STAGE:-prepare_and_fit}

export PYTHONPATH="$CODE/src:$ROOT/sources/critic-selected-a1229239-r1"

if [[ "$MODE" == "prepare" || "$MODE" == "prepare_and_fit" ]]; then
  if [[ ! -f "$RUN/$TASK/preparation/ASSEMBLY.json" ]]; then
    "$PY" -B "$RUN/PREPARE-STREAM.py" \
      --task "$TASK" \
      --code "$CODE" \
      --source-commit "$COMMIT" \
      --output "$RUN/$TASK/preparation" \
      --expected-hostname "$(hostname)" \
      --max-seconds "${TASK_PREPARE_MAX_SECONDS:-9000}" \
      --files-per-batch "${TASK_FILES_PER_BATCH:-4}"
  fi
fi

if [[ "$MODE" == "fit" || "$MODE" == "prepare_and_fit" ]]; then
  ASSEMBLY="$RUN/$TASK/preparation/ASSEMBLY.json"
  ASSEMBLY_SHA=$(sha256sum "$ASSEMBLY" | awk '{print $1}')
  exec "$PY" -B "$RUN/FITS.py" \
    --assembly "$ASSEMBLY" \
    --assembly-sha256 "$ASSEMBLY_SHA" \
    --output "$RUN/$TASK/fits" \
    --source-commit "$COMMIT" \
    --expected-hostname "$(hostname)" \
    --attempt "${TASK_ATTEMPT:-r1}" \
    --max-elapsed-seconds "${TASK_FIT_MAX_SECONDS:-4800}"
fi
