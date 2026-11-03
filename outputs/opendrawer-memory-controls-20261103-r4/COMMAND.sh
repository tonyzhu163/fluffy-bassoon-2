#!/usr/bin/env bash
set -Eeuo pipefail
: "${TASK_CODE:?}" "${TASK_PIN:?}"
export TASK_MEMORY_OUTPUT=/mowenjie/fluffy-bassoon-2/outputs/opendrawer-memory-controls-20261103-r4
exec bash "$TASK_CODE/outputs/opendrawer-memory-controls-20261103-r3/COMMAND.sh"
