#!/usr/bin/env bash
set -euo pipefail
umask 077

# Root reviews the pinned reservation and deploys through Git. This
# script only runs inside the new, explicitly zero-GPU CPU allocation.
TASK_ARTIFACT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_PYTHON=/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python
TASK_OUTPUT=/mowenjie/fluffy-bassoon-2/outputs/robotwin-continuation-software-qual-20261031-r1
TASK_LAUNCHER_PIN=
TASK_EXPECTED_HOSTNAME=

while (($#)); do
  case "$1" in
    --launcher-pin) TASK_LAUNCHER_PIN=${2:?}; shift 2 ;;
    --expected-hostname) TASK_EXPECTED_HOSTNAME=${2:?}; shift 2 ;;
    --output) TASK_OUTPUT=${2:?}; shift 2 ;;
    *) echo "Unknown launcher argument: $1" >&2; exit 2 ;;
  esac
done

[[ $(uname -s) == Linux ]] || { echo 'Allocated Linux CPU execution only' >&2; exit 2; }
[[ $TASK_LAUNCHER_PIN =~ ^[0-9a-f]{40}$ ]] || { echo 'An exact root launcher Git pin is required' >&2; exit 2; }
[[ $TASK_EXPECTED_HOSTNAME == job-* && $(hostname) == "$TASK_EXPECTED_HOSTNAME" ]] || {
  echo 'Actual job hostname differs from the explicit owner binding' >&2; exit 2;
}
[[ $TASK_OUTPUT == /mowenjie/fluffy-bassoon-2/outputs/robotwin-continuation-software-qual-* ]] || {
  echo 'Qualification output must be a fresh project-specific run identity' >&2; exit 2;
}
[[ ! -e $TASK_OUTPUT ]] || { echo 'Refuse existing qualification output' >&2; exit 2; }
mkdir -- "$TASK_OUTPUT"

export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTEST_ADDOPTS=
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset PYTHONPATH
# Keep the allocation's incoming CUDA visibility intact for the actual count
# receipt. The Python coordinator hides devices only after count==0 passes.
ulimit -c 0

set +e
timeout --signal=TERM --kill-after=10s 560s \
  "$TASK_PYTHON" "$TASK_ARTIFACT_DIR/qualify.py" \
  --reservation "$TASK_ARTIFACT_DIR/reservation.json" \
  --launcher-pin "$TASK_LAUNCHER_PIN" \
  --expected-hostname "$TASK_EXPECTED_HOSTNAME" --output "$TASK_OUTPUT" \
  > "$TASK_OUTPUT/console.log" 2>&1
TASK_EXIT=$?
set -e
printf '%s\n' "$TASK_EXIT" > "$TASK_OUTPUT/exit-code.txt"
"$TASK_PYTHON" "$TASK_ARTIFACT_DIR/qualify.py" --finish-only \
  --output "$TASK_OUTPUT" --external-exit-code "$TASK_EXIT"
exit "$TASK_EXIT"
