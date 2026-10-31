#!/usr/bin/env bash
# Bounded same-task comparison; semantic task holdouts follow runtime qualification.
set -Eeuo pipefail
TASK_SOURCE=${1:?exact integrated source commit}
TASK_RUN=${2:?fresh comparison output identity}
TASK_SCRIPT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$TASK_SCRIPT/vla_libero_qualification.sh" "$TASK_SOURCE" "$TASK_RUN" comparison
