#!/usr/bin/env bash
# Execution only inside the owner's independently capped exclusive allocation.
set -Eeuo pipefail
TASK_LAUNCHER_COMMIT=${1:?full launcher commit}
TASK_RUN=${2:?fresh robotwin-trained-pair run identity}
TASK_PLAN=${3:?absolute compiled plan path}
TASK_PLAN_SHA=${4:?compiled plan SHA256}
[[ "$TASK_LAUNCHER_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$TASK_RUN" =~ ^robotwin-trained-pair-[A-Za-z0-9-]+$ ]]
[[ "$TASK_PLAN" = /* && "$TASK_PLAN_SHA" =~ ^[0-9a-f]{64}$ ]]
[[ "$(hostname)" = job-* && "$(uname -s)" = Linux ]]
TASK_LAUNCHER=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_PYTHON=$TASK_ROOT/envs/robotwin-vlrla-20261029-r1/bin/python
TASK_OUTPUT=$TASK_ROOT/outputs/$TASK_RUN
test ! -e "$TASK_OUTPUT"
test "$(git -C "$TASK_LAUNCHER" rev-parse HEAD)" = "$TASK_LAUNCHER_COMMIT"
test -z "$(git -C "$TASK_LAUNCHER" status --porcelain)"
trap 'TASK_RC=$?; if [[ -d "$TASK_OUTPUT" ]]; then printf "%s\n" "$TASK_RC" > "$TASK_OUTPUT/exit-code.txt"; fi' EXIT
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export FLUFFY_GPU_EXECUTION=baige CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# Runtime also enforces2520s including its CPU gates. This independent outer
# timer (2650+20s) leaves room under the45-minute scheduler limit. No retries.
timeout --signal=TERM --kill-after=20s 2650s "$TASK_PYTHON" \
  "$TASK_LAUNCHER/scripts/diagnostics/robotwin_trained_pair.py" \
  --launcher-commit "$TASK_LAUNCHER_COMMIT" --plan "$TASK_PLAN" --plan-sha256 "$TASK_PLAN_SHA"
