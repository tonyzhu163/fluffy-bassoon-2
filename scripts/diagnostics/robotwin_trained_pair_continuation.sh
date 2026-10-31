#!/usr/bin/env bash
# New exclusive allocation only; this script never schedules or stops a job.
set -Eeuo pipefail
TASK_LAUNCHER_COMMIT=${1:?full qualified continuation launcher commit}
TASK_RUN=${2:?fresh robotwin-trained-continuation identity}
TASK_PLAN=${3:?absolute fresh initial compiled admission plan}
TASK_PLAN_SHA=${4:?initial plan SHA256}
TASK_AUTHORITY=${5:?absolute owner continuation authority}
TASK_AUTHORITY_SHA=${6:?continuation authority SHA256}
[[ "$TASK_LAUNCHER_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$TASK_RUN" =~ ^robotwin-trained-continuation-[A-Za-z0-9-]+$ ]]
[[ "$TASK_PLAN" = /* && "$TASK_AUTHORITY" = /* ]]
[[ "$TASK_PLAN_SHA" =~ ^[0-9a-f]{64}$ && "$TASK_AUTHORITY_SHA" =~ ^[0-9a-f]{64}$ ]]
[[ "$(hostname)" = job-* && "$(uname -s)" = Linux ]]
TASK_LAUNCHER=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_PYTHON=$TASK_ROOT/envs/robotwin-vlrla-20261029-r1/bin/python
TASK_OUTPUT=$TASK_ROOT/outputs/$TASK_RUN
test ! -e "$TASK_OUTPUT"
test "$(git -C "$TASK_LAUNCHER" rev-parse HEAD)" = "$TASK_LAUNCHER_COMMIT"
test -z "$(git -C "$TASK_LAUNCHER" status --porcelain)"
TASK_TIMEOUT=$("$TASK_PYTHON" - "$TASK_AUTHORITY" "$TASK_AUTHORITY_SHA" <<'PY'
import hashlib,json,sys
from pathlib import Path
raw=Path(sys.argv[1]).read_bytes()
assert hashlib.sha256(raw).hexdigest()==sys.argv[2]
value=json.loads(raw);seconds=value['maximum_driver_seconds']
assert type(seconds) is int and 2700<=seconds<=43200
assert seconds+180<=value['resources']['scheduler_cap_seconds']<=46800
print(seconds+120)
PY
)
trap 'TASK_RC=$?; if [[ -d "$TASK_OUTPUT" ]]; then printf "%s\n" "$TASK_RC" > "$TASK_OUTPUT/exit-code.txt"; fi' EXIT
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export FLUFFY_GPU_EXECUTION=baige CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
timeout --signal=TERM --kill-after=30s "${TASK_TIMEOUT}s" "$TASK_PYTHON" \
  "$TASK_LAUNCHER/scripts/diagnostics/robotwin_trained_pair_continuation.py" \
  --launcher-commit "$TASK_LAUNCHER_COMMIT" \
  --initial-plan "$TASK_PLAN" --initial-plan-sha256 "$TASK_PLAN_SHA" \
  --authority "$TASK_AUTHORITY" --authority-sha256 "$TASK_AUTHORITY_SHA" \
  --output "$TASK_OUTPUT"
