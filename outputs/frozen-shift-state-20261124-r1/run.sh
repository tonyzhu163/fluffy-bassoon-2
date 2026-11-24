#!/usr/bin/env bash
# In a Baige Distributed Training allocation with one RTX 4090.
# Usage: run.sh <pin> <attempt> [panel.py options, e.g. --starts development]
# Runs raw, repeat and the four state adapters through the shared panel. Results land in
# $root/outputs/frozen-vla-proposal-shift-20261124-r1/<attempt>.
set -euo pipefail
[[ $# -ge 2 ]]
here=$(cd "$(dirname "$0")" && pwd -P)
shared="$here/../frozen-vla-proposal-shift-20261124-r1"
# state-dropped must read the frozen checkpoint's own state mean, not the archived CPU fallback.
echo "b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84  /mowenjie/fluffy-bassoon/inputs/pi05-libero-openpi-20261027-r1/assets/physical-intelligence/libero/norm_stats.json" | sha256sum -c -
(
    code=$(git -C "$here" rev-parse --show-toplevel)
    source "$here/../frozen-vla-native-transfer-20261123-r1/env.sh"
    "$python" -B "$here/adapter.py" | grep -F '"dropped_state_from": "checkpoint:'
)
exec "$shared/run.sh" "$1" "$2" --adapters state-mapped,state-dropped,state-shuffled,state-mapped-shuffled "${@:3}"
