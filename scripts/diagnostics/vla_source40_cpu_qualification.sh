#!/usr/bin/env bash
# Existing Baige CPU allocation:0GPU,4CPU,16GiB,30min; never submits a job.
set -Eeuo pipefail
TASK_STARTED=$(date +%s)
TASK_SOURCE=${1:?reviewed exact source}
TASK_RUN=${2:?fresh immutable CPU identity}
TASK_PLAN=${3:?hash-bound source plan}
TASK_PLAN_SHA=${4:?explicit plan file SHA256}
TASK_LEDGER=${5:?canonical topup ledger}
shift 5
TASK_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_PYTHON=/mowenjie/fluffy-bassoon/envs/lerobot-v044-8fff0fde/bin/python
TASK_TEST=/mowenjie/fluffy-bassoon-2/envs/pytest-overlay-20261029-r1
TASK_OUT=/mowenjie/fluffy-bassoon-2/outputs/$TASK_RUN
case "$TASK_RUN" in *[!a-zA-Z0-9-]*|'') exit 2;; esac
cd "$TASK_CODE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
test ! -e "$TASK_OUT"
test "$(sha256sum "$TASK_PLAN" | cut -d' ' -f1)" = "$TASK_PLAN_SHA"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONPATH=$TASK_TEST:$TASK_CODE/src
# CUDA_VISIBLE_DEVICES is deliberately untouched until the helper has attested
# zero devices from the real allocation using CUDA-capable cp310 Torch.
task_cleanup() {
  TASK_RC=$?
  trap - EXIT
  if [ -f "$TASK_OUT/cpu-worker-clock.json" ]; then
    "$TASK_PYTHON" -m fluffy_bassoon.meta_rl.vla_source40_cpu_qualification finish \
      --output "$TASK_OUT" --exit-code "$TASK_RC" || true
    printf '%s\n' "$TASK_RC" > "$TASK_OUT/exit-code.txt"
  fi
  exit "$TASK_RC"
}
trap task_cleanup EXIT
TASK_SECONDS=$((1740 - $(date +%s) + TASK_STARTED))
test "$TASK_SECONDS" -gt 60
timeout --signal=INT --kill-after=20s "$TASK_SECONDS" "$TASK_PYTHON" \
  -m fluffy_bassoon.meta_rl.vla_source40_cpu_qualification run \
  --source "$TASK_SOURCE" --run "$TASK_RUN" --output "$TASK_OUT" \
  --plan "$TASK_PLAN" --plan-sha256 "$TASK_PLAN_SHA" --ledger "$TASK_LEDGER" \
  --expected-hostname "$(hostname)" --started-unix "$TASK_STARTED" "$@"
