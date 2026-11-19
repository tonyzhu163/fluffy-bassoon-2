#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
code=$(git -C "$here" rev-parse --show-toplevel)
pin=$1
attempt=$2
[[ "$attempt" =~ ^[a-z0-9]+$ ]]
[[ $(git -C "$code" rev-parse HEAD) == "$pin" && -z $(git -C "$code" status --porcelain) ]]
root=/mowenjie/fluffy-bassoon-2
archive="$root/outputs/libero-decision-pretraining-20261119-r1/menu02"
[[ $(cut -d' ' -f1 "$archive/launch.txt") == b90ee5a0c049d301529d0bf661d0a682b858ed78 ]]
py="$root/setups/robocasa365-native-20261101-r1/env/bin/python"
out="$root/outputs/$(basename "$here")/$attempt"
mkdir "$out"
start=$(date +%s)
trap 'rc=$?; echo "$rc $(date +%s) $start" > "$out/exit.txt"' EXIT
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH="$code/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
df -h "$out" > "$out/storage.txt"
printf '%s %s %s\n' "$pin" "$(hostname)" "$$" > "$out/launch.txt"
"$py" -Bu -c 'import sys, numpy, torch; print(sys.version, numpy.__version__, torch.__version__, "CUDA", torch.cuda.is_available()); assert not torch.cuda.is_available()' > "$out/runtime.txt"
timeout --kill-after=10s 15m "$py" -Bu "$here/reader.py" --episodes "$archive/episodes" --output "$out/fit" > "$out/run.log" 2>&1
