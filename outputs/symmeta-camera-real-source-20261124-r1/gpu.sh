#!/usr/bin/env bash
# Parent submits as one exclusive 4090 distributed job after source cache and independent banks are ready.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
source "$here/env.sh"
mode=${1:?fit-score or score-kinova}
pin=${2:?full Git pin}
attempt=${3:?fresh attempt}
input=${4:?source cache or sealed fit attempt}
expected=${5:?SHA256 of ready.json or sealed.json}
[[ $(uname -s) == Linux && $(hostname) == job-* && $pin =~ ^[0-9a-f]{40}$ && $expected =~ ^[0-9a-f]{64}$ ]]
[[ $(git -C "$here" rev-parse HEAD) == "$pin" && -z $(git --no-optional-locks -C "$here" status --porcelain --untracked-files=no) ]]
[[ $attempt =~ ^[A-Za-z0-9_-]+$ && $input =~ ^[A-Za-z0-9_-]+$ ]]
case "$mode" in
    fit-score) receipt=$run/$input/ready.json; [[ ${6:?both or ur5e} == both || $6 == ur5e ]] ;;
    score-kinova) receipt=$run/$input/sealed.json ;;
    *) exit 2 ;;
esac
[[ $(sha256sum "$receipt" | cut -d' ' -f1) == "$expected" ]]
out=$run/$attempt
mkdir "$out"
echo "$pin $(hostname) $mode $input $expected" > "$out/launch.txt"
start=$(date +%s)
trap 'rc=$?; end=$(date +%s); echo "$rc $start $end" > "$out/exit.txt"' EXIT
df -h "$out" > "$out/storage.txt"
code=$(git -C "$here" rev-parse --show-toplevel)
confirm=$root/setups/symmeta-perceiver-followups-20261123-r1/src-6030a74ba800
[[ $(git -C "$confirm" rev-parse HEAD) == 6030a74ba800011b650aaedf7a83fb62a80c7418 ]]
export PYTHONPATH="$PYTHONPATH:$confirm/outputs/symmeta-camera-independent-body-20261124-r1:$confirm/outputs/symmeta-camera-frozen-preprocess-20261124-r1"
export JAX_PLATFORMS=cuda FLUFFY_GPU_EXECUTION=baige CUDA_VISIBLE_DEVICES=0
libs=$root/setups/libero90-full-action-20261119-r1/desktop-libs02
IFS=: read -ra paths <<< "${LD_LIBRARY_PATH:-}"
drivers=""
for path in "${paths[@]}"; do
    if [[ -n $path && ! -f $path/libtorch_python.so ]]; then drivers+=":$path"; fi
done
export LD_LIBRARY_PATH="$libs/prefix/usr/lib/x86_64-linux-gnu$drivers"
"$root/setups/robocasa365-native-20261101-r1/env/bin/python" -Bu "$code/scripts/[train]gpu_probe.py" --expected-count 1 \
    --purpose training --baige-container --baige-exclusive-training --output "$out/gpu.json" > "$out/probe.log" 2>&1
"$py" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"]=="passed" and d["exclusive_training_admission"] and "4090" in d["devices"][0]["cuda_name"]' "$out/gpu.json"
if [[ $mode == fit-score ]]; then
    "$py" -Bu "$here/fit_real.py" "$out" "$run/$input" "$primary" > "$out/fit.log" 2>&1
    "$py" -Bu "$here/score_targets.py" "$out" UR5e "$out" > "$out/UR5e.log" 2>&1
    if [[ $6 == both ]]; then "$py" -Bu "$here/score_targets.py" "$out" Kinova3 "$out" > "$out/Kinova3.log" 2>&1; fi
else
    "$py" -Bu "$here/score_targets.py" "$run/$input" Kinova3 "$out" > "$out/Kinova3.log" 2>&1
fi
