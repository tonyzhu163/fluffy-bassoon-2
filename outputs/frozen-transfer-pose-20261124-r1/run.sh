#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
export POSE_CODE=$(git -C "$here" rev-parse --show-toplevel)
test -z "$(git -C "$POSE_CODE" status --porcelain --untracked-files=no)"
root=/mowenjie/fluffy-bassoon-2
prior=$POSE_CODE/outputs/frozen-vla-reachability-20261123-r1
export REACH_CPUS=$1 REACH_CACHE="$3/cache-$1"
[[ $REACH_CPUS == 46 || $REACH_CPUS == 47 ]]
arm=$2
out=$3
mode=${4:-replay}
export POSE_PANEL=$POSE_CODE/outputs/frozen-vla-native-transfer-20261123-r1
mkdir -p "$REACH_CACHE"
# cpu.sh pins the original native reset source, release, simulator, software renderer and CPU runtime.
if [[ $mode == replay ]]; then
  exec bash "$prior/cpu.sh" "$here/entry.py" replay "$arm" "$root/outputs/frozen-vla-native-transfer-20261123-r1/gpu01" "$out"
else
  exec bash "$prior/cpu.sh" "$here/entry.py" "$mode" "$arm" "$out"
fi
