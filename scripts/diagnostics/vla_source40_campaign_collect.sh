#!/usr/bin/env bash
# One explicit source stream; reservation prevents duplicate campaign spending.
set -Eeuo pipefail
TASK_SOURCE=${1:?exact commit}
TASK_RUN=${2:?unique run}
TASK_SUITE=${3:?source suite}
TASK_NUMBER=${4:?source task index}
TASK_PARTITION=${5:?train or validation}
TASK_RESUME=${6:-}
case "$TASK_RESUME" in ''|--resume) ;; *) exit 2;; esac
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_WORKTREE=$TASK_ROOT/worktrees/$TASK_RUN
TASK_PLAN=$TASK_ROOT/outputs/libero40-source-1m-20261029-r1
cd "$TASK_WORKTREE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
printf -v TASK_KEY '%s:%02d' "$TASK_SUITE" "$TASK_NUMBER"
TASK_RESUME_ARGS=()
if [ "$TASK_RESUME" = --resume ]; then TASK_RESUME_ARGS=(--resume); fi
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$TASK_WORKTREE/src \
  /mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python \
  -m fluffy_bassoon.meta_rl.vla_source40_campaign reserve \
  --campaign "$TASK_PLAN/campaign.json" --schedule "$TASK_PLAN/schedule.json" \
  --ledger "$TASK_PLAN/ledger.json" --task "$TASK_KEY" --partition "$TASK_PARTITION" \
  --run "$TASK_RUN" --commit "$TASK_SOURCE" "${TASK_RESUME_ARGS[@]}"
exec bash scripts/diagnostics/vla_source40_collect.sh "$TASK_SOURCE" "$TASK_RUN" \
  "$TASK_PLAN/schedule.json" "$TASK_SUITE" "$TASK_NUMBER" "$TASK_PARTITION" "${TASK_RESUME_ARGS[@]}"
