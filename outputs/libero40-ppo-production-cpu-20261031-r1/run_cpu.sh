#!/usr/bin/env bash
# Actual Baige allocation:0GPU,4CPU,16GiB,<=20min. Never submits/schedules.
set -Eeuo pipefail
TASK_STARTED=$(date +%s)
TASK_SOURCE=${1:?reviewed exact source}
TASK_RUN=${2:?fresh immutable CPU test identity}
TASK_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_PYTHON=/mowenjie/fluffy-bassoon/envs/lerobot-v044-8fff0fde/bin/python
TASK_TEST=/mowenjie/fluffy-bassoon-2/envs/pytest-overlay-20261029-r1
TASK_OUT=/mowenjie/fluffy-bassoon-2/outputs/$TASK_RUN
TASK_DRIVER=$TASK_CODE/outputs/libero40-ppo-production-cpu-20261031-r1/run_tests.py
cd "$TASK_CODE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
test ! -e "$TASK_OUT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=$TASK_TEST:$TASK_CODE/src
# Visibility remains untouched until require_zero_gpu observes the actual allocation.
task_cleanup() {
  TASK_RC=$?
  trap - EXIT
  if [ -f "$TASK_OUT/cpu-worker-clock.json" ]; then
    "$TASK_PYTHON" "$TASK_DRIVER" finish --output "$TASK_OUT" --exit-code "$TASK_RC" || true
    printf '%s\n' "$TASK_RC" > "$TASK_OUT/exit-code.txt"
  fi
  exit "$TASK_RC"
}
trap task_cleanup EXIT
TASK_SECONDS=$((1100 + TASK_STARTED - $(date +%s)))
test "$TASK_SECONDS" -gt 60
timeout --signal=INT --kill-after=20s "$TASK_SECONDS" "$TASK_PYTHON" "$TASK_DRIVER" run \
  --source "$TASK_SOURCE" --run "$TASK_RUN" --output "$TASK_OUT" \
  --expected-hostname "$(hostname)" --started-unix "$TASK_STARTED"
