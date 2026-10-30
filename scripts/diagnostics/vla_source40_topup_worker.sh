#!/usr/bin/env bash
# One of eight already funded3h allocations. No scheduling/resource grant.
set -Eeuo pipefail
TASK_SOURCE=${1:?reviewed commit}
TASK_PREFIX=${2:?fresh campaign run prefix}
TASK_PLAN=${3:?immutable production plan}
TASK_LEDGER=${4:?canonical cumulative action/compute ledger}
TASK_WORKER=${5:?worker0..7}
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_MODEL=/mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python
TASK_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_STARTED=$(date +%s)
TASK_DEADLINE=$((TASK_STARTED + 10800 - 60))
TASK_RECEIPTS=$TASK_ROOT/outputs/$TASK_PREFIX/worker-$TASK_WORKER
mkdir -p "$TASK_RECEIPTS"
export PYTHONPATH=$TASK_CODE/src PYTHONDONTWRITEBYTECODE=1
"$TASK_MODEL" - "$TASK_PLAN" "$TASK_WORKER" > "$TASK_RECEIPTS/assigned.txt" <<'PY'
import sys
from fluffy_bassoon.meta_rl.vla_source40_topup import load_plan
plan, worker = load_plan(sys.argv[1]), int(sys.argv[2])
assert plan['mode'] == 'production' and 0 <= worker < 8
for row in [r for r in plan['slots'] if r['launchable']][worker::8]:
    print(row['slot'])
PY
while IFS= read -r TASK_SLOT; do
  TASK_LEFT=$((TASK_DEADLINE - $(date +%s)))
  if [ "$TASK_LEFT" -lt 900 ]; then break; fi
  TASK_SUFFIX=${TASK_SLOT/libero_spatial/s}
  TASK_SUFFIX=${TASK_SUFFIX/libero_object/o}
  TASK_SUFFIX=${TASK_SUFFIX/libero_goal/g}
  TASK_SUFFIX=${TASK_SUFFIX/libero_10/l}
  TASK_SUFFIX=${TASK_SUFFIX//:/}
  TASK_SUFFIX=${TASK_SUFFIX//\//-}
  TASK_RUN=$TASK_PREFIX-$TASK_SUFFIX
  TASK_RESUME=()
  if [ -f "$TASK_ROOT/outputs/$TASK_RUN/worker-time.json" ]; then TASK_RESUME=(--resume); fi
  TASK_SLICE=2040
  if [ "$TASK_SLICE" -gt "$((TASK_LEFT - 60))" ]; then TASK_SLICE=$((TASK_LEFT - 60)); fi
  #2100s per stream;40*2100 + a1000s qualification <=24h.
  #The enclosing eight3h allocation ceilings include inter-child setup/idle.
  timeout --signal=TERM --kill-after=10s "${TASK_LEFT}s" bash "$TASK_CODE/scripts/diagnostics/vla_source40_topup.sh" \
    "$TASK_SOURCE" "$TASK_RUN" "$TASK_PLAN" "$TASK_SLOT" "$TASK_LEDGER" "$TASK_SLICE" 2100 "${TASK_RESUME[@]}"
  printf '%s\n' "$TASK_SLOT" >> "$TASK_RECEIPTS/attempted.txt"
done < "$TASK_RECEIPTS/assigned.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$TASK_RECEIPTS/worker-ended.txt"
#Unattempted assignments remain visible. This is not a full-coverage receipt.
