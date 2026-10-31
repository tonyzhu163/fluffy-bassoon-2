#!/usr/bin/env bash
# Existing0GPU/4CPU/16GiB CPU allocation, maximum20min. No scheduler or funding.
set -Eeuo pipefail
TASK_STARTED=$(date +%s)
TASK_SOURCE=${1:?reviewed exact source}
TASK_RUN=${2:?fresh immutable CPU preparation identity}
TASK_PREFIX=${3:?fresh eventual production collection prefix}
TASK_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_PYTHON=/mowenjie/fluffy-bassoon/envs/lerobot-v044-8fff0fde/bin/python
TASK_OUT=/mowenjie/fluffy-bassoon-2/outputs/$TASK_RUN
cd "$TASK_CODE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
test ! -e "$TASK_OUT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export PYTHONPATH=$TASK_CODE/src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
# Keep scheduler visibility untouched until require_zero_gpu attests the actual allocation.
task_cleanup() {
  TASK_RC=$?
  trap - EXIT
  if [ -f "$TASK_OUT/cpu-worker-clock.json" ]; then
    "$TASK_PYTHON" -m fluffy_bassoon.meta_rl.vla_source40_production_plan_cpu finish \
      --output "$TASK_OUT" --exit-code "$TASK_RC" || true
    printf '%s\n' "$TASK_RC" > "$TASK_OUT/exit-code.txt"
  fi
  exit "$TASK_RC"
}
trap task_cleanup EXIT
TASK_SECONDS=$((1140 - $(date +%s) + TASK_STARTED))
test "$TASK_SECONDS" -gt 60
timeout --signal=INT --kill-after=20s "$TASK_SECONDS" "$TASK_PYTHON" \
  -m fluffy_bassoon.meta_rl.vla_source40_production_plan_cpu run \
  --source "$TASK_SOURCE" --run "$TASK_RUN" --production-prefix "$TASK_PREFIX" \
  --output "$TASK_OUT" --expected-hostname "$(hostname)" --started-unix "$TASK_STARTED"
