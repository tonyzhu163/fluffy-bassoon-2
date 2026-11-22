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
base_ckpt_attempt=${3:-gpu03v2}
reuse_pro2_attempt=${4:-}
host=$(hostname)
[[ "$attempt" =~ ^[a-z0-9]+$ ]]
[[ "$host" == job-* ]] || { echo "Require Baige Distributed Training container" >&2; exit 2; }
[[ $(git -C "$code" rev-parse HEAD) == "$pin" && -z $(git -C "$code" status --porcelain) ]]

out="$run/$attempt"
mkdir -p "$out/logs" "$out/pro2"
df -h "$out" > "$out/storage.txt"
printf '%s %s %s pro2-visual-stress\n' "$pin" "$host" "$$" > "$out/launch.txt"
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
base_ckpts="$run/$base_ckpt_attempt/stage1/checkpoints"

PRO2_CKPT_ARGS=()
if [[ -n "$reuse_pro2_attempt" ]]; then
    PRO2_CKPT_ARGS=(--pro2-ckpt-dir "$run/$reuse_pro2_attempt/pro2/checkpoints")
else
    # Stage 1: Collect Panda+Sawyer train-split active probes across BASE + S13/S14/S15/S16/S19 (8 GPUs)
    pids=()
    for s in 0 1 2 3 4 5 6 7; do
        CUDA_VISIBLE_DEVICES="$s" MUJOCO_EGL_DEVICE_ID="$s" \
            "$python" -Bu "$here/eval_pro2_visual_stress.py" collect-train \
            --out-dir "$out/pro2" --shard-idx "$s" --num-shards 8 \
            > "$out/logs/collect-train-$s.log" 2>&1 &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done

    # Stage 2: Train Camera-Conditioned EquivFlow-Procrustes-V, Equiv-V2 Pro2, and Perceiver-ICBA Pro2 (1 GPU)
    CUDA_VISIBLE_DEVICES=0 "$python" -Bu "$here/eval_pro2_visual_stress.py" train-readers \
        --out-dir "$out/pro2" --num-shards 8 --steps 3000 > "$out/logs/train-readers.log" 2>&1
fi

# Stage 3: Zero-shot eval on held-out 6-DoF UR5e test tasks across S13/S14/S15/S16/S19 (8 GPUs)
pids=()
for s in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES="$s" MUJOCO_EGL_DEVICE_ID="$s" \
        timeout --kill-after=20s 90m "$python" -Bu "$here/eval_pro2_visual_stress.py" eval-test \
        --out-dir "$out/pro2" --train-archive "$archive" --base-ckpt-dir "$base_ckpts" \
        "${PRO2_CKPT_ARGS[@]}" \
        --shard-idx "$s" --num-shards 8 --seed-ckpt 41 \
        > "$out/logs/eval-test-$s.log" 2>&1 &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

CUDA_VISIBLE_DEVICES=0 "$python" -Bu "$here/eval_pro2_visual_stress.py" merge \
    --out-dir "$out/pro2" --num-shards 8 > "$out/logs/merge.log" 2>&1
date +%s > "$out/gpu-end.txt"
