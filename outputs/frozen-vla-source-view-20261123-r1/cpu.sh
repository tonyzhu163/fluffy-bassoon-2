#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
code=$(cd "$here/../.." && pwd -P)
export VIEW_PIN=$(git -C "$code" rev-parse HEAD)
[[ -z $(git -C "$code" status --porcelain --untracked-files=no) ]]
root=/mowenjie/fluffy-bassoon-2
run=$root/outputs/frozen-vla-source-view-20261123-r1
out=$run/${1:?fresh output name}
shift
mkdir "$out"
export REACH_CACHE=$run/cache REACH_CPUS=36,37
export PYTHONPATH
# The original qualifier sets pinned simulator imports. Insert the versioned FK helper via site-independent entry.
exec bash "$code/outputs/frozen-vla-reachability-20261123-r1/cpu.sh" "$here/entry.py" "$here/qualify.py" \
    "$root/outputs/frozen-vla-native-transfer-20261123-r1/gpu01" "$out" "$@" >"$out/console.log" 2>&1
