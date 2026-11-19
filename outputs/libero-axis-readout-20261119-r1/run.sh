#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
code=$(git -C "$here" rev-parse --show-toplevel)
pin=$1
attempt=$2
[[ "$attempt" =~ ^[a-z0-9]+$ && $(git -C "$code" rev-parse HEAD) == "$pin" ]]
[[ -z $(git -C "$code" status --porcelain) ]]
root=/mowenjie/fluffy-bassoon-2
source_run="$root/outputs/libero-axis-calibration-20261119-r1/cpu01"
[[ $(cut -d' ' -f1 "$source_run/launch.txt") == 00e2e507e35578059074902b371f0acd83f0e6b5 ]]
py="$root/setups/robocasa365-native-20261101-r1/env/bin/python"
out="$root/outputs/$(basename "$here")/$attempt"
mkdir "$out"
start=$(date +%s)
trap 'rc=$?; echo "$rc $(date +%s) $start" > "$out/exit.txt"' EXIT
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
df -h "$out" > "$out/storage.txt"
printf '%s %s %s\n' "$pin" "$(hostname)" "$$" > "$out/launch.txt"
timeout --kill-after=5s 5m "$py" -Bu "$here/probe.py" \
    --episodes "$root/outputs/libero-decision-pretraining-20261119-r1/menu02/episodes" \
    --fits "$source_run/fit" --output "$out/probes" > "$out/run.log" 2>&1
