#!/usr/bin/env bash
set -euo pipefail
pin=$1
attempt=$2
[[ "$attempt" =~ ^cpu[0-9]+$ ]]
here=$(cd "$(dirname "$0")" && pwd)
code=$(git -C "$here" rev-parse --show-toplevel)
[[ "$(git -C "$code" rev-parse HEAD)" == "$pin" ]]
out=/mowenjie/fluffy-bassoon-2/outputs/$(basename "$here")/$attempt
mkdir -p "$(dirname "$out")"
mkdir "$out"
df -h "$out" > "$out/storage.txt"
printf '%s %s %s\n' "$pin" "$(hostname)" "$(date -Is)" > "$out/launch.txt"
trap 'rc=$?; printf "%s %s\n" "$rc" "$(date -Is)" > "$out/exit.txt"' EXIT
export PYTHONPATH="$code/src" CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
py=/mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python
taskset -pc $$ >> "$out/launch.txt"
[[ "$(taskset -pc $$)" == *": 38,39" || "$(taskset -pc $$)" == *": 38-39" ]]
ulimit -t 670
{ time "$py" -Bu "$here/score.py" "$out" > "$out/run.log" 2>&1; } 2>"$out/time.txt"
