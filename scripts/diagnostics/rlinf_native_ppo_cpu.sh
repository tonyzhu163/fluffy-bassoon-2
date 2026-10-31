#!/usr/bin/env bash
# Actual Baige0GPU/8CPU/64GiB,<=20min. Imports/config/fake-worker tests only.
set -Eeuo pipefail
TASK_STARTED=$(date +%s)
TASK_SOURCE=${1:?exact clean combined source}
TASK_RUN=${2:?fresh immutable CPU run ID}
TASK_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_PYTHON=${NATIVE_PYTHON:-/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python}
TASK_OUTPUT=/mowenjie/fluffy-bassoon-2/outputs/$TASK_RUN
export REPO_PATH=/mowenjie/fluffy-bassoon-2/sources/robotwin-baseline-smoke-20261028-r1/RLinf
export ROBOTWIN_PATH=/mowenjie/fluffy-bassoon-2/sources/robotwin-baseline-smoke-20261028-r1/RoboTwin
export EMBODIED_PATH=$REPO_PATH/examples/embodiment
export PYTHONPATH=$TASK_CODE/src:$REPO_PATH:$ROBOTWIN_PATH
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd "$TASK_CODE"
task_cleanup() {
  TASK_RC=$?
  trap - EXIT
  "$TASK_PYTHON" -m fluffy_bassoon.meta_rl.rlinf_native_ppo finish-cpu \
    --output "$TASK_OUTPUT" --exit-code "$TASK_RC" || true
  if [ -d "$TASK_OUTPUT" ]; then printf '%s\n' "$TASK_RC" > "$TASK_OUTPUT/exit-code.txt"; fi
  exit "$TASK_RC"
}
trap task_cleanup EXIT
# No CUDA visibility edit before require_zero_gpu attests the real allocation.
timeout --signal=INT --kill-after=20s 1140 "$TASK_PYTHON" \
  -m fluffy_bassoon.meta_rl.rlinf_native_ppo cpu --code "$TASK_CODE" \
  --source "$TASK_SOURCE" --output "$TASK_OUTPUT" \
  --started-unix "$TASK_STARTED"
