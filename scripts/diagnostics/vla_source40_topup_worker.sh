#!/usr/bin/env bash
# One of eight approved <=2h55m allocations;10560s cumulative including startup.
# No scheduler submission. The root-funded24h macro budget is mandatory.
set -Eeuo pipefail
TASK_STARTED=$(date +%s)
TASK_SOURCE=${1:?reviewed commit}
TASK_PREFIX=${2:?fresh campaign run prefix}
TASK_PLAN=${3:?immutable production plan}
TASK_LEDGER=${4:?canonical cumulative action/child-compute ledger}
TASK_WORKER=${5:?worker0..7}
TASK_BUDGET=${6:?root-funded immutable24h macro budget}
TASK_RESUME=${7:-}
case "$TASK_RESUME" in ''|--resume) ;; *) exit 2;; esac
case "$TASK_PREFIX" in *[!a-zA-Z0-9-]*|'') exit 2;; esac
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_MODEL=/mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python
TASK_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_RECEIPTS=$TASK_ROOT/outputs/$TASK_PREFIX/worker-$TASK_WORKER
cd "$TASK_CODE"
mkdir -p "$TASK_RECEIPTS"
exec 9> "$TASK_RECEIPTS/launcher.lock"
flock -n 9
export PYTHONPATH=$TASK_CODE/src PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
TASK_RESUME_ARGS=()
if [ "$TASK_RESUME" = --resume ]; then TASK_RESUME_ARGS=(--resume); fi
TASK_CLOCK=$("$TASK_MODEL" -m fluffy_bassoon.meta_rl.vla_source40_topup_worker begin \
  --root "$TASK_RECEIPTS" --plan "$TASK_PLAN" --ledger "$TASK_LEDGER" --budget "$TASK_BUDGET" \
  --prefix "$TASK_PREFIX" --worker "$TASK_WORKER" --source-commit "$TASK_SOURCE" \
  --started-unix "$TASK_STARTED" "${TASK_RESUME_ARGS[@]}")
IFS=$'\t' read -r TASK_ATTEMPT TASK_DEADLINE TASK_SLICE <<< "$TASK_CLOCK"
exec > >(tee "$TASK_ATTEMPT/console.log") 2>&1
task_cleanup() {
  TASK_RC=$?
  trap - EXIT
  PYTHONPATH=$TASK_CODE/src "$TASK_MODEL" -m fluffy_bassoon.meta_rl.vla_source40_topup_worker finish \
    --root "$TASK_RECEIPTS" --attempt "$TASK_ATTEMPT" --exit-code "$TASK_RC" \
    > "$TASK_ATTEMPT/macro-finish.log" 2>&1 || true
  printf '%s\n' "$TASK_RC" > "$TASK_ATTEMPT/exit-code.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$TASK_ATTEMPT/worker-ended.txt"
  exit "$TASK_RC"
}
trap task_cleanup EXIT
# Clock precedes deployment checks, so an allocated preflight failure is billed.
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
while IFS= read -r TASK_SLOT; do
  TASK_CHOICE=$("$TASK_MODEL" -m fluffy_bassoon.meta_rl.vla_source40_topup_worker select \
    --root "$TASK_RECEIPTS" --attempt "$TASK_ATTEMPT" --slot "$TASK_SLOT")
  IFS=$'\t' read -r TASK_DECISION TASK_RUN TASK_CHILD_SLICE TASK_CHILD_RESUME TASK_TIMEOUT TASK_DECISION_PATH <<< "$TASK_CHOICE"
  if [ "$TASK_DECISION" = fulfilled ]; then
    printf 'Retain proven terminal fulfilled stream: %s (%s)\n' "$TASK_SLOT" "$TASK_DECISION_PATH"
    continue
  fi
  if [ "$TASK_DECISION" != launch ]; then
    printf 'Unfulfilled stream retained; cumulative time cannot fit a child slice: %s (%s)\n' "$TASK_SLOT" "$TASK_DECISION_PATH"
    exit 3
  fi
  TASK_CHILD_ARGS=()
  if [ "$TASK_CHILD_RESUME" = 1 ]; then TASK_CHILD_ARGS=(--resume); fi
  set +e
  timeout --signal=TERM --kill-after=10s "${TASK_TIMEOUT}s" \
    bash "$TASK_CODE/scripts/diagnostics/vla_source40_topup.sh" \
      "$TASK_SOURCE" "$TASK_RUN" "$TASK_PLAN" "$TASK_SLOT" "$TASK_LEDGER" "$TASK_CHILD_SLICE" 2100 "${TASK_CHILD_ARGS[@]}"
  TASK_CHILD_RC=$?
  set -e
  "$TASK_MODEL" -m fluffy_bassoon.meta_rl.vla_source40_topup_worker outcome \
    --root "$TASK_RECEIPTS" --attempt "$TASK_ATTEMPT" --slot "$TASK_SLOT" --exit-code "$TASK_CHILD_RC"
  # A failed/partial child stops for diagnosis. Explicit same-source resume
  # consumes its remaining original macro and child clocks.
  if [ "$TASK_CHILD_RC" -ne 0 ]; then exit "$TASK_CHILD_RC"; fi
done < "$TASK_RECEIPTS/assigned.txt"
