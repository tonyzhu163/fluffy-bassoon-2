#!/usr/bin/env bash
# Parent's exclusive 4090 distributed job. Arguments: full pin, fresh attempt, physical fixture.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
code=$(git -C "$here" rev-parse --show-toplevel)
pin=${1:?pin}
attempt=${2:?attempt}
bank=${3:?fixture}
[[ $(hostname) == job-* && $attempt =~ ^gpu[[:alnum:]]+$ && $bank =~ ^cpu[0-9]+$ ]]
[[ $(git -C "$here" rev-parse HEAD) == "$pin" && -z $(git -C "$here" status --porcelain --untracked-files=no) ]]
root=/mowenjie/fluffy-bassoon-2
run="$root/outputs/$(basename "$here")"
out="$run/$attempt"
mkdir "$out"
start=$(date +%s)
echo "$pin $(hostname) $bank $start" > "$out/launch.txt"
trap 'rc=$?; echo "$rc $start $(date +%s)" > "$out/exit.txt"' EXIT
df -h "$out" > "$out/storage.txt"
[[ $(cut -d' ' -f1 "$run/$bank/exit.txt") == 0 ]]
export PYTHONPATH="$code/src" PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export JAX_PLATFORMS=cuda FLUFFY_GPU_EXECUTION=baige CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false
py=/mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python
libs=$root/setups/libero90-full-action-20261119-r1/desktop-libs02
IFS=: read -ra paths <<< "${LD_LIBRARY_PATH:-}"
drivers=""
for path in "${paths[@]}"; do
    if [[ -n $path && ! -f $path/libtorch_python.so ]]; then drivers+=":$path"; fi
done
export LD_LIBRARY_PATH="$libs/prefix/usr/lib/x86_64-linux-gnu$drivers"
"$root/setups/robocasa365-native-20261101-r1/env/bin/python" -Bu "$code/scripts/[train]gpu_probe.py" \
    --expected-count 1 --purpose training --baige-container --baige-exclusive-training \
    --output "$out/gpu.json" > "$out/probe.log" 2>&1
"$py" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"]=="passed" and d["exclusive_training_admission"] and "4090" in d["devices"][0]["cuda_name"]' "$out/gpu.json"
remaining() { echo $((5280 - $(date +%s) + start)); }
timeout "$(remaining)" "$py" -Bu "$here/learn.py" "$out" "$run/$bank" > "$out/learn.log" 2>&1
timeout "$(remaining)" "$py" -Bu "$here/targets.py" "$out" "$run/$bank" \
    "$root/outputs/symmeta-query-direction-probe-20261124-r1/cpu03" > "$out/targets.log" 2>&1
