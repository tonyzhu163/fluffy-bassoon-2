#!/usr/bin/env bash
# Actual allocated0GPU/4CPU/16GiB,20min. Frozen tiny SAC actors only; no GPU fallback.
set -Eeuo pipefail
TASK_STARTED=$(date +%s)
TASK_SOURCE=${1:?reviewed exact clean source}
TASK_RUN=${2:?fresh immutable derived-generation CPU identity}
TASK_ADMISSION=${3:?immutable source snapshot admission}
TASK_ADMISSION_SHA=${4:?exact admission SHA256}
TASK_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_PYTHON=/mowenjie/fluffy-bassoon/envs/lerobot-v044-8fff0fde/bin/python
TASK_OUT=/mowenjie/fluffy-bassoon-2/outputs/$TASK_RUN
cd "$TASK_CODE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
test ! -e "$TASK_OUT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export PYTHONPATH=/mowenjie/fluffy-bassoon-2/envs/pytest-overlay-20261029-r1:$TASK_CODE/src
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
# No pre-mask CUDA edit: basePython3.12 attests actual zero allocated devices first.
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
  -m fluffy_bassoon.meta_rl.vla_source40_dpt_same_teacher run --source "$TASK_SOURCE" \
  --run "$TASK_RUN" --output "$TASK_OUT" --admission "$TASK_ADMISSION" --admission-sha256 "$TASK_ADMISSION_SHA" \
  --expected-hostname "$(hostname)" --started-unix "$TASK_STARTED"
