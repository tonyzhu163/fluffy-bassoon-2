#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
tree=$(git -C "$here" rev-parse --show-toplevel)
pin=$1
[[ $(hostname) == job-*-master-0 && $(git -C "$tree" rev-parse HEAD) == "$pin" ]]
[[ -z $(git -C "$tree" status --porcelain --untracked-files=no) ]]
r=/mowenjie/fluffy-bassoon-2
run=$r/outputs/frozen-vla-cotraining-20261124-r1
out=$run/sampler01
mkdir "$out"
exec > >(tee "$out/console.log") 2>&1
start=$(date +%s)
trap 'rc=$?; echo "$rc $start $(date +%s)" > "$out/exit.txt"' EXIT
echo "pin=$pin host=$(hostname)" > "$out/launch.txt"
code=$tree
source "$tree/outputs/frozen-vla-native-transfer-20261123-r1/env.sh"
export PYTHONPATH="$here/../model:$here/../data:$PYTHONPATH"
export JAX_PLATFORMS=cuda FLUFFY_GPU_EXECUTION=baige CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
"$root/setups/robocasa365-native-20261101-r1/env/bin/python" -Bu "$tree/scripts/[train]gpu_probe.py" --expected-count 1 --purpose evaluation --baige-container --output "$out/gpu.json"
timeout --kill-after=10 600 "$python" -Bu "$here/sampler.py" "$run/data/dataset-v1" "$out/sampler.json"
