#!/usr/bin/env bash
# Run only inside root's one allocated0GPU CPU qualification job.
set -euo pipefail
TASK_PIN=${1:?exact clean auxiliary/controller commit required}
shift
TASK_RUN=${1:-robotwin-trained-handoff-control-qual-20261031-r1}
if (( $# )); then shift; fi
TASK_EXTRA_TESTS=("$@")
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_CODE=${TASK_QUAL_CODE_ROOT:-$TASK_ROOT/aux/$TASK_PIN}
TASK_PYTHON=$TASK_ROOT/envs/robotwin-vlrla-20261029-r1/bin/python
TASK_OUT=$TASK_ROOT/outputs/$TASK_RUN
[[ "$TASK_PIN" =~ ^[0-9a-f]{40}$ ]]
[[ "$TASK_RUN" =~ ^robotwin-trained-handoff-control-qual-[a-z0-9_-]+$ ]]
test "$(hostname)" != ""
[[ "$(hostname)" == job-* ]]
cd "$TASK_CODE"
test "$(git rev-parse HEAD)" = "$TASK_PIN"
test -z "$(git status --porcelain)"
test ! -e "$TASK_OUT"
mkdir -p "$TASK_OUT"
exec > >(tee "$TASK_OUT/console.log") 2>&1
trap 'TASK_EXIT=$?; printf "%s\n" "$TASK_EXIT" > "$TASK_OUT/exit-code.txt"' EXIT
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$TASK_CODE/src" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
printf '%s\n' "$TASK_PIN" > "$TASK_OUT/source-commit.txt"
hostname > "$TASK_OUT/hostname.txt"
timeout --signal=TERM --kill-after=10s 240 "$TASK_PYTHON" -m pytest -q -x --tb=short \
  --basetemp "/dev/shm/$TASK_RUN" \
  outputs/robotwin-trained-handoff-control-20261030-r1/tests/test_control.py \
  "${TASK_EXTRA_TESTS[@]}"
