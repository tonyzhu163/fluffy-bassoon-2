#!/usr/bin/env bash
# Prepared for owner admission/deployment/UI launch; never a submission command.
set -euo pipefail
TASK_CODE=${1:?exact clean candidate worktree}
TASK_PIN=${2:?exact candidate commit}
TASK_OUT=${3:?fresh runtime output}
TASK_RESOURCE=${4:?actual allocated-zero-GPU receipt}
TASK_RESOURCE_SHA=${5:?receipt SHA256}
TASK_HOST=${6:?exact allocation hostname}
TASK_PYTHON=/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python
trap 'TASK_RC=$?; if [[ -d "$TASK_OUT" ]]; then printf "%s\n" "$TASK_RC" > "$TASK_OUT/exit-code.txt"; fi' EXIT
# External timer begins before the Python interpreter or any model/test import.
# <=1770s TERM +10s KILL remains below the1800s scheduler request.
timeout --signal=TERM --kill-after=10s 1770 "$TASK_PYTHON" -I -B -S \
  "$TASK_CODE/outputs/robotwin-sac65-cpu-compat-20261031-r1/gate_job.py" \
  --code-root "$TASK_CODE" --code-commit "$TASK_PIN" --output "$TASK_OUT" \
  --resource-receipt "$TASK_RESOURCE" --resource-sha256 "$TASK_RESOURCE_SHA" \
  --expected-hostname "$TASK_HOST"
