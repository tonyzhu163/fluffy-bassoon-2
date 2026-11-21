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
reuse_attempt=${4:-gpu09test}
prev_attempt=${5:-gpu10test}
host=$(hostname)
[[ "$attempt" =~ ^[a-z0-9]+$ ]]
[[ "$host" == job-* ]] || { echo "Require Baige Distributed Training container" >&2; exit 2; }
[[ $(git -C "$code" rev-parse HEAD) == "$pin" && -z $(git -C "$code" status --porcelain) ]]

out="$run/$attempt"
src="$run/$src_attempt"
prev="$run/$prev_attempt"
mkdir -p "$out/logs" "$out/test_live/shard-3"
cp -f "$prev/test_live/shard-3/live-progress.json" "$out/test_live/shard-3/live-progress.json"
printf '%s %s %s official-test-finish-shard3\n' "$pin" "$host" "$$" > "$out/launch.txt"
start=$(date +%s)
trap 'rc=$?; printf "%s %s %s\n" "$rc" "$(date +%s)" "$start" > "$out/exit.txt"' EXIT

export LIBERO_PRO_ROOT="$sim/simulator-source" LIBERO_CONFIG_PATH="$sim/libero-config"
export CROSSARM_RELEASE="$release"
export OPENPI_DATA_HOME="$legacy/cache/openpi-215abfb"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$code/src:$release:$sim/simulator-source:$code/outputs/libero-crossarm-20261120-r1:$code/outputs/libero-xemb-ur5e-sysid-20261120-r1:$here"
export JAX_PLATFORMS=cuda FLUFFY_GPU_EXECUTION=baige XLA_PYTHON_CLIENT_PREALLOCATE=false
export NCCL_P2P_DISABLE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1

libs="$root/setups/libero90-full-action-20261119-r1/desktop-libs02"
glvnd="$libs/prefix/usr/lib/x86_64-linux-gnu"
IFS=: read -ra paths <<< "${LD_LIBRARY_PATH:-}"
driver_paths=""
for path in "${paths[@]}"; do
    if [[ -n "$path" && ! -f "$path/libtorch_python.so" ]]; then driver_paths+=":$path"; fi
done
export LD_LIBRARY_PATH="$glvnd:$libs/prefix/usr/lib/x86_64-linux-gnu$driver_paths"
export __EGL_VENDOR_LIBRARY_FILENAMES="$libs/nvidia-vendor.json"

export CUDA_VISIBLE_DEVICES=0
date +%s > "$out/gpu-start.txt"
"$root/setups/robocasa365-native-20261101-r1/env/bin/python" -Bu "$code/scripts/[train]gpu_probe.py" \
    --expected-count 1 --purpose training --baige-container --baige-exclusive-training \
    --output "$out/gpu.json" > "$out/probe.log" 2>&1

CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
    timeout --kill-after=20s 25m "$python" -Bu "$here/eval_test_official.py" run-shard \
    --train-dir "$src/train" \
    --output-dir "$out/test_live" \
    --shard-idx 3 \
    --num-shards 8 \
    --seed-ckpt 41 \
    --reuse-support-dir "$run/$reuse_attempt/test_live" \
    > "$out/logs/test-shard-3.log" 2>&1

# Assemble complete 8-shard merged directory without mutating immutable prev_attempt
for s in 0 1 2 4 5 6 7; do
    ln -sfn "$prev/test_live/shard-$s" "$out/test_live/shard-$s"
done
CUDA_VISIBLE_DEVICES=0 "$python" -Bu "$here/eval_test_official.py" merge \
    --output-dir "$out/test_live" --num-shards 8 > "$out/logs/merge.log" 2>&1
