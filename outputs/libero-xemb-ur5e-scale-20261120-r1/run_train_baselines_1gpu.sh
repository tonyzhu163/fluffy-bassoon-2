#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
code=$(git -C "$here" rev-parse --show-toplevel)
root=/mowenjie/fluffy-bassoon-2
legacy=/mowenjie/fluffy-bassoon
release="$root/sources/libero-pro2-e5500c69"
sim="$root/setups/libero-meta-learning-20261117-r1"
run="$root/outputs/$(basename "$here")"
python="$legacy/envs/openpi-215abfb/bin/python"

pin=$1
attempt=$2
src_attempt=${3:-gpu01}
host=$(hostname)
[[ "$attempt" =~ ^[a-z0-9]+$ ]]
[[ "$host" == job-* ]] || { echo "Require Baige Distributed Training container" >&2; exit 2; }
[[ $(git -C "$code" rev-parse HEAD) == "$pin" && -z $(git -C "$code" status --porcelain) ]]

out="$run/$attempt"
src="$run/$src_attempt"
mkdir -p "$out/train"
printf '%s %s %s train-rma-gru-farneback\n' "$pin" "$host" "$$" > "$out/launch.txt"
start=$(date +%s)
trap 'rc=$?; printf "%s %s %s\n" "$rc" "$(date +%s)" "$start" > "$out/exit.txt"' EXIT

export LIBERO_PRO_ROOT="$sim/simulator-source" LIBERO_CONFIG_PATH="$sim/libero-config"
export CROSSARM_RELEASE="$release"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$code/src:$release:$sim/simulator-source:$code/outputs/libero-crossarm-20261120-r1:$code/outputs/libero-xemb-ur5e-sysid-20261120-r1:$here"
export JAX_PLATFORMS=cuda FLUFFY_GPU_EXECUTION=baige XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=0

date +%s > "$out/gpu-start.txt"
"$root/setups/robocasa365-native-20261101-r1/env/bin/python" -Bu "$code/scripts/[train]gpu_probe.py" \
    --expected-count 1 --purpose training --baige-container --baige-exclusive-training \
    --output "$out/gpu.json" > "$out/probe.log" 2>&1

"$python" -Bu "$here/train_rma_gru_and_farneback.py" \
    --archive "$src/collect/archive.json" \
    --output "$out/train" \
    --steps 3000 --batch-size 64 --cnn-width 96 --mlp-width 192 --seeds 41,42,43 \
    > "$out/train.log" 2>&1
