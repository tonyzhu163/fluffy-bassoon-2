#!/usr/bin/env bash
set -euo pipefail
pin=$1
attempt=$2
run=libero-witness-transfer-20261119-r1
root=/mowenjie/fluffy-bassoon-2
release="$root/sources/libero-pro2-e5500c69"
runtime="$root/setups/libero-meta-learning-20261117-r1"
archive="$root/outputs/libero-interrupted-practice-20261119-r1/closed01"
code=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
[[ $(git -C "$code" rev-parse HEAD) == "$pin" && -z $(git -C "$code" status --porcelain) ]]
[[ $(git -C "$release" rev-parse HEAD) == e5500c690c1498eca1b7050350654964190cdf70 ]]
[[ $(cut -d' ' -f1 "$archive/launch.txt") == 22da285052a4878c4843d80d9241ba599ae7b66d ]]
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$release:$runtime/simulator-source:$code/src"
export LIBERO_PRO_ROOT="$runtime/simulator-source" LIBERO_CONFIG_PATH="$runtime/libero-config"
out="$root/outputs/$run/$attempt"
mkdir "$out"
start=$(date +%s)
trap 'rc=$?; echo "$rc $(date +%s) $start" > "$out/exit.txt"' EXIT
df -h "$out" > "$out/storage.txt"
printf '%s %s %s\n' "$pin" "$(hostname)" "$$" > "$out/launch.txt"
timeout --kill-after=10s 30m "$runtime/env/bin/python" -Bu "$code/outputs/$run/replay.py" "$out" > "$out/run.log" 2>&1
