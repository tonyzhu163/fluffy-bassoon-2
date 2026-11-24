#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
root=/mowenjie/fluffy-bassoon-2
out=${OBJECT_OUT:-$root/outputs/frozen-transfer-object-20261124-r1/cpu01}
mkdir -p "$out/cache"
export REACH_CPUS=${OBJECT_CPUS:-32,33}
export REACH_CACHE="$out/cache"
runtime="$root/outputs/frozen-vla-reachability-20261123-r1/final-b1cf4be383c928001d5a8af2eb4ef190d043d632/cpu.sh"
script=$1
shift
exec bash "$runtime" "$here/$script" "$out" "$@"
