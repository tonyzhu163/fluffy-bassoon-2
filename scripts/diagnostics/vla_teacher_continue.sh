#!/usr/bin/env bash
# Fresh immutable 20k-interaction ordinary DSRL continuation, then teacher gate.
set -Eeuo pipefail
TASK_SOURCE=${1:?exact integrated source commit}
TASK_RUN=${2:?fresh continuation output identity}
TASK_SCRIPT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TASK_PRIOR=${3:-/mowenjie/fluffy-bassoon-2/outputs/libero-vla-baige-20261029-qual-r3/experiment}
exec bash "$TASK_SCRIPT/vla_cnd_poc.sh" "$TASK_SOURCE" "$TASK_RUN" "$TASK_PRIOR" continuation
