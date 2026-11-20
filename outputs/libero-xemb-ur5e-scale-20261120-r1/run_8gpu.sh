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
sim_python="$sim/env/bin/python"

pin=$1
stage=$2
attempt=$3
host=$(hostname)
[[ "$attempt" =~ ^[a-z0-9]+$ ]]
[[ "$host" == job-* ]] || { echo "Require Baige Distributed Training container" >&2; exit 2; }
[[ $(git -C "$code" rev-parse HEAD) == "$pin" && -z $(git -C "$code" status --porcelain) ]]

out="$run/$attempt"
mkdir -p "$run"
mkdir "$out"
df -h "$out" > "$out/storage.txt"
printf '%s %s %s %s\n' "$pin" "$host" "$$" "$stage" > "$out/launch.txt"
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

libs="$root/setups/libero90-full-action-20261119-r1/desktop-libs02"
glvnd="$libs/prefix/usr/lib/x86_64-linux-gnu"
IFS=: read -ra paths <<< "${LD_LIBRARY_PATH:-}"
driver_paths=""
for path in "${paths[@]}"; do
    if [[ -n "$path" && ! -f "$path/libtorch_python.so" ]]; then driver_paths+=":$path"; fi
done
export LD_LIBRARY_PATH="$glvnd:$libs/prefix/lib/x86_64-linux-gnu$driver_paths"
export __EGL_VENDOR_LIBRARY_FILENAMES="$libs/nvidia-vendor.json"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
date +%s > "$out/gpu-start.txt"
"$root/setups/robocasa365-native-20261101-r1/env/bin/python" -Bu "$code/scripts/[train]gpu_probe.py" \
    --expected-count 8 --purpose training --baige-container --baige-exclusive-training \
    --output "$out/gpu.json" > "$out/probe.log" 2>&1

if [[ "$stage" == "collect-train" ]]; then
    mkdir -p "$out/archive" "$out/logs"
    pids=()
    for s in 0 1 2 3 4 5 6 7; do
        CUDA_VISIBLE_DEVICES="$s" MUJOCO_EGL_DEVICE_ID="$s" \
            timeout --kill-after=20s 60m "$python" -Bu "$here/collect_scale.py" \
            --output "$out/archive" --num-shards 8 --shard-idx "$s" \
            > "$out/logs/collect-shard-$s.log" 2>&1 &
        pids+=($!)
    done
    fail=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then fail=1; fi
    done
    [[ "$fail" -eq 0 ]] || { echo "One or more collect shards failed" >&2; exit 1; }

    CUDA_VISIBLE_DEVICES=0 "$python" -Bu "$here/collect_scale.py" \
        --output "$out/archive" --num-shards 8 --merge > "$out/logs/merge.log" 2>&1

    CUDA_VISIBLE_DEVICES=0 timeout --kill-after=20s 30m "$python" -Bu "$here/train_scale.py" \
        --archive "$out/archive/archive.json" --output "$out/train" \
        --steps 3000 --batch-size 64 --cnn-width 96 --mlp-width 192 --seeds 41,42,43 \
        > "$out/logs/train.log" 2>&1
    date +%s > "$out/gpu-end.txt"
    exit 0
fi

if [[ "$stage" == "eval-live" ]]; then
    colfit_attempt=${4:?Missing collect-train attempt name}
    mkdir -p "$out/live" "$out/logs"
    pids=()
    for s in 0 1 2 3 4 5 6 7; do
        CUDA_VISIBLE_DEVICES="$s" MUJOCO_EGL_DEVICE_ID="$s" \
            timeout --kill-after=20s 60m "$python" -Bu "$here/eval_live_scale.py" \
            --archive "$run/$colfit_attempt/archive/archive.json" \
            --train-dir "$run/$colfit_attempt/train" \
            --output "$out/live" --num-shards 8 --shard-idx "$s" --seed-ckpt 41 \
            > "$out/logs/eval-shard-$s.log" 2>&1 &
        pids+=($!)
    done
    fail=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then fail=1; fi
    done
    [[ "$fail" -eq 0 ]] || { echo "One or more eval shards failed" >&2; exit 1; }

    CUDA_VISIBLE_DEVICES=0 "$python" -Bu "$here/eval_live_scale.py" \
        --output "$out/live" --num-shards 8 --merge > "$out/logs/merge.log" 2>&1
    date +%s > "$out/gpu-end.txt"
    exit 0
fi

echo "Unsupported stage $stage" >&2
exit 2
