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
cv2_overlay="$root/setups/xemb-icml-campaign-20261121-r1/cv2-headless-overlay"

pin=$1
attempt=$2
ckpt_attempt=${3:-gpu03v2}
homog_attempt=${4:-gpu04homog}
host=$(hostname)
[[ "$attempt" =~ ^[a-z0-9]+$ ]]
[[ "$host" == job-* ]] || { echo "Require Baige Distributed Training container" >&2; exit 2; }
[[ $(git -C "$code" rev-parse HEAD) == "$pin" && -z $(git -C "$code" status --porcelain) ]]

out="$run/$attempt"
mkdir -p "$out/logs" "$out/stage3"
df -h "$out" > "$out/storage.txt"
printf '%s %s %s stage3-scaling-ablations\n' "$pin" "$host" "$$" > "$out/launch.txt"
start=$(date +%s)
trap 'rc=$?; printf "%s %s %s\n" "$rc" "$(date +%s)" "$start" > "$out/exit.txt"' EXIT

export LIBERO_PRO_ROOT="$sim/simulator-source" LIBERO_CONFIG_PATH="$sim/libero-config"
export CROSSARM_RELEASE="$release"
export OPENPI_DATA_HOME="$legacy/cache/openpi-215abfb"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$cv2_overlay:$code/src:$release:$sim/simulator-source:$code/outputs/libero-crossarm-20261120-r1:$code/outputs/libero-xemb-ur5e-sysid-20261120-r1:$code/outputs/libero-xemb-ur5e-scale-20261120-r1:$here"
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
export LD_LIBRARY_PATH="$glvnd$driver_paths"
export __EGL_VENDOR_LIBRARY_FILENAMES="$libs/nvidia-vendor.json"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
date +%s > "$out/gpu-start.txt"
"$root/setups/robocasa365-native-20261101-r1/env/bin/python" -Bu "$code/scripts/[train]gpu_probe.py" \
    --expected-count 8 --purpose training --baige-container --baige-exclusive-training \
    --output "$out/gpu.json" > "$out/probe.log" 2>&1

archive="$root/outputs/libero-xemb-ur5e-scale-20261120-r1/gpu01/archive/archive.json"
ckpts="$run/$ckpt_attempt/stage1/checkpoints"
homog_ckpts="$run/$homog_attempt/stage1/checkpoints"
reuse="$root/outputs/libero-xemb-ur5e-scale-20261120-r1/gpu09test/test_live"

# Step 1: Horizon ablation (K in {1..24} chunks = 5..120 controls) on locked 96/96 UR5e (runs on GPU 0 in ~1 min)
CUDA_VISIBLE_DEVICES=0 "$python" -Bu "$here/eval_stage3_scaling.py" run-horizon \
    --out-dir "$out/stage3" --train-archive "$archive" --ckpt-dir "$ckpts" \
    --homog-ckpt-dir "$homog_ckpts" --reuse-support-dir "$reuse" \
    > "$out/logs/horizon.log" 2>&1 &
h_pid=$!

# Step 2: Collect 1,466 real MuJoCo Panda+Sawyer train trials across 128 continuous embodiments (8 GPUs parallel)
pids=()
for s in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES="$s" MUJOCO_EGL_DEVICE_ID="$s" \
        "$python" -Bu "$here/eval_stage3_scaling.py" collect-diverse \
        --out-dir "$out/stage3" --shard-idx "$s" --num-shards 8 \
        > "$out/logs/collect-$s.log" 2>&1 &
    pids+=($!)
done
wait "$h_pid"
for p in "${pids[@]}"; do wait "$p"; done

# Step 3: Sweep 5 models x 7 M-levels x 3 seeds = 105 models on 8 GPUs parallel, evaluated on locked 96/96 UR5e
pids=()
for w in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES="$w" \
        "$python" -Bu "$here/eval_stage3_scaling.py" run-scaling \
        --out-dir "$out/stage3" --reuse-support-dir "$reuse" \
        --worker-idx "$w" --num-workers 8 --steps 2500 \
        > "$out/logs/scale-$w.log" 2>&1 &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

CUDA_VISIBLE_DEVICES=0 "$python" -Bu "$here/eval_stage3_scaling.py" merge \
    --out-dir "$out/stage3" --num-workers 8 > "$out/logs/merge.log" 2>&1
date +%s > "$out/gpu-end.txt"
