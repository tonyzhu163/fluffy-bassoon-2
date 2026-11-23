#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
root=/mowenjie/fluffy-bassoon-2
export REACH_CACHE=$root/outputs/frozen-vla-phase-transfer-20261123-r1/cache
export REACH_CPUS=30,31
exec bash "$root/outputs/frozen-vla-reachability-20261123-r1/final-b1cf4be383c928001d5a8af2eb4ef190d043d632/cpu.sh" "$here/$1" "${@:2}"
