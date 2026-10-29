#!/usr/bin/env bash
# Existing allocated GPU only. This does not submit or authorize an allocation.
set -Eeuo pipefail
TASK_SOURCE=${1:?exact reviewed worker commit}
TASK_WORKER=${2:?worker index0..7}
TASK_RECEIPT=${3:?fresh absolute worker receipt directory}
TASK_WALL=${4:-14400}
TASK_HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
test "$(git -C "$TASK_HERE" rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git -C "$TASK_HERE" status --porcelain)"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$TASK_HERE/src
exec /mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python \
  -m fluffy_bassoon.meta_rl.vla_source40_campaign_worker run \
  --source-commit "$TASK_SOURCE" --worker "$TASK_WORKER" --workers 8 \
  --expected-hostname "$(hostname)" --output "$TASK_RECEIPT" \
  --max-elapsed-seconds "$TASK_WALL" --child-timeout-seconds 2400
