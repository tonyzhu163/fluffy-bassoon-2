#!/usr/bin/env bash
# Closed software/final100 profiles of the existing CPU gate; no fit/robot work.
set -euo pipefail
TASK_CODE=${1:?exact auxiliary code worktree}
TASK_PIN=${2:?exact auxiliary pin}
TASK_OUT=${3:?fresh software output}
TASK_BOOTSTRAP=${4:?actual UI proof ticket}
TASK_HELPERS=${5:?exact three0c16 ownership helpers}
TASK_PROFILE=${6:?exact software profile}
TASK_PROFILE_SHA=${7:?profile SHA256}
(( $# == 7 || $# == 8 ))
TASK_MODE=${8:-original}
TASK_TERM=570
if [[ "$TASK_MODE" == teacher-residual ]]; then TASK_TERM=450
elif [[ "$TASK_MODE" == ad500-readout ]]; then TASK_TERM=270
elif [[ "$TASK_MODE" == final100-ad ]]; then TASK_TERM=6790
elif [[ "$TASK_MODE" == final100-ad-residual ]]; then TASK_TERM=5770
elif [[ "$TASK_MODE" == final100-ad-proof-recovery ]]; then TASK_TERM=5590
else [[ "$TASK_MODE" == original ]]; fi
TASK_PYTHON=/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python
trap 'TASK_RC=$?; if [[ -d "$TASK_OUT" ]]; then printf "%s\n" "$TASK_RC" > "$TASK_OUT/exit-code.txt"; fi' EXIT
timeout --signal=TERM --kill-after=10s "$TASK_TERM" "$TASK_PYTHON" -I -B -S \
  "$TASK_CODE/outputs/robotwin-sac65-cpu-compat-20261031-r6/gate_job.py" \
  --code-root "$TASK_CODE" --code-commit "$TASK_PIN" --output "$TASK_OUT" \
  --bootstrap-ticket "$TASK_BOOTSTRAP" --owned-helpers "$TASK_HELPERS" \
  --software-profile "$TASK_PROFILE" --software-profile-sha256 "$TASK_PROFILE_SHA"
