#!/usr/bin/env bash
# Initial small test on one 4090 or RTX 6000 GPU; later eight-GPU work is separate.
# Scheduler cap:110 minutes; prior failed qualification used <6 minutes.
set -euo pipefail
TASK_SOURCE=${1:?exact source commit}
TASK_QUAL_RUN=${2:?fresh qualification run ID}
TASK_PILOT_RUN=${3:?fresh pilot run ID}
[[ $TASK_SOURCE =~ ^[0-9a-f]{40}$ ]]
[[ $TASK_QUAL_RUN =~ ^robomimic-can-dsrl-baige-20261029-qual-r[0-9]+$ ]]
[[ $TASK_PILOT_RUN =~ ^robomimic-can-dsrl-baige-20261029-pilot-r[0-9]+$ ]]
TASK_ROOT=/mowenjie/fluffy-bassoon-2/worktrees
TASK_QUAL_START=$SECONDS
bash "$TASK_ROOT/$TASK_QUAL_RUN/scripts/diagnostics/can_dsrl_qualification.sh" "$TASK_SOURCE" "$TASK_QUAL_RUN"
if (( SECONDS - TASK_QUAL_START > 1200 )); then
  echo 'Qualification exceeded 20 minutes; refusing the 90-minute pilot to preserve the total budget.'
  exit 4
fi
bash "$TASK_ROOT/$TASK_PILOT_RUN/scripts/diagnostics/can_dsrl_pilot.sh" "$TASK_SOURCE" "$TASK_PILOT_RUN" "$TASK_QUAL_RUN"
