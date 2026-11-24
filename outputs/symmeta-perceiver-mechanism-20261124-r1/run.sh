#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
code=$(git -C "$here" rev-parse --show-toplevel)
pin=${1:?full deployed revision}
attempt=${2:?fresh attempt name}
mode=${3:-full}
[[ $pin =~ ^[a-f0-9]{40}$ && $(git -C "$code" rev-parse HEAD) == "$pin" ]]
[[ $attempt =~ ^[a-z0-9_-]+$ && ( $mode == full || $mode == qualify ) ]]
root=/mowenjie/fluffy-bassoon-2
out="$root/outputs/$(basename "$here")/$attempt"
mkdir -p "$(dirname "$out")"
mkdir "$out"
start=$(date +%s)
trap 'rc=$?; echo "$rc $start $(date +%s)" > "$out/exit.txt"' EXIT
export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$root/setups/xemb-icml-campaign-20261121-r1/cv2-headless-overlay:$code/src"
py=/mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python
args=()
if [[ $mode == qualify ]]; then args+=(--qualify); fi
printf '%s %s %s %s\n' "$pin" "$(hostname)" "$mode" "$py" > "$out/launch.txt"
df -h "$out" > "$out/storage.txt"
taskset -c 38,39 "$py" -Bu "$here/replay.py" "$out" "${args[@]}" > "$out/run.log" 2>&1
if [[ $mode == full ]]; then
    taskset -c 38,39 "$py" -Bu "$here/summarize.py" "$out" >> "$out/run.log" 2>&1
fi
