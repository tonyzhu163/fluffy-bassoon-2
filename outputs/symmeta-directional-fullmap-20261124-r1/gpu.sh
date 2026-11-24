#!/usr/bin/env bash
# Parent distributed single RTX4090 forward. PIN NEW_GPU_ATTEMPT READER_GPU_ATTEMPT [BANK]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
code=$(git -C "$here" rev-parse --show-toplevel)
pin=${1:?code pin}; attempt=${2:?new gpu attempt}; reader=${3:?original reader attempt}; bank=${4:-cpu01}
[[ $(hostname) == job-* && $attempt =~ ^gpu[0-9]+$ && $reader =~ ^gpu[0-9]+$ && $bank =~ ^cpu[0-9]+$ ]]
[[ $(git -C "$code" rev-parse HEAD) == "$pin" && -z $(git -C "$code" status --porcelain --untracked-files=no) ]]
root=/mowenjie/fluffy-bassoon-2
run=$root/outputs/$(basename "$here")
up=$root/outputs/symmeta-active-map-reader-20261124-r1
[[ -f "$up/$reader/selected.json" && $(cut -d' ' -f1 "$up/$bank/exit.txt") == 0 ]]
mkdir -p "$run"
out=$run/$attempt
mkdir "$out"
start=$(date +%s)
echo "$pin $(hostname) $start $reader $bank" > "$out/launch.txt"
trap 'rc=$?; echo "$rc $start $(date +%s)" > "$out/exit.txt"' EXIT
df -h "$out" > "$out/storage.txt"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$code/src" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0 FLUFFY_GPU_EXECUTION=baige JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
py=/mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python
libs=$root/setups/libero90-full-action-20261119-r1/desktop-libs02
IFS=: read -ra paths <<< "${LD_LIBRARY_PATH:-}"
drivers=""
for path in "${paths[@]}"; do
    if [[ -n $path && ! -f $path/libtorch_python.so ]]; then drivers+=":$path"; fi
done
export LD_LIBRARY_PATH="$libs/prefix/usr/lib/x86_64-linux-gnu$drivers"
timeout 120 "$root/setups/robocasa365-native-20261101-r1/env/bin/python" -Bu "$code/scripts/[train]gpu_probe.py" \
    --expected-count 1 --purpose training --baige-container --baige-exclusive-training --output "$out/gpu.json" > "$out/probe.log" 2>&1
"$py" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"]=="passed" and d["exclusive_training_admission"] and "4090" in d["devices"][0]["cuda_name"]' "$out/gpu.json"
TIMEFORMAT='wall=%R user=%U sys=%S'
{ time timeout "$((1740 - $(date +%s) + start))" "$py" -Bu "$here/forward.py" "$out" "$up/$reader" "$up/$bank" > "$out/forward.log" 2>&1; } 2> "$out/time.txt"
