#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
code=$(git -C "$here" rev-parse --show-toplevel)
root=/mowenjie/fluffy-bassoon-2
legacy=/mowenjie/fluffy-bassoon
release="$root/sources/libero-pro2-e5500c69"
sim="$root/setups/libero-meta-learning-20261117-r1"
campaign_code="$code/outputs/xemb-icml-campaign-20261121-r1"
campaign_run="$root/outputs/xemb-icml-campaign-20261121-r1"
run="$root/outputs/$(basename "$here")"
python="$legacy/envs/openpi-215abfb/bin/python"
cv2_overlay="$root/setups/xemb-icml-campaign-20261121-r1/cv2-headless-overlay"

pin=$1
attempt=$2
stage3_attempt=${3:-}
host=$(hostname)
[[ "$attempt" =~ ^[a-z0-9]+$ ]]
[[ "$host" == job-* ]] || { echo "Require Baige Distributed Training container" >&2; exit 2; }
[[ $(git -C "$code" rev-parse HEAD) == "$pin" && -z $(git -C "$code" status --porcelain) ]]

if [[ -n "$stage3_attempt" ]]; then
    diverse="$campaign_run/$stage3_attempt/stage3/diverse_shards"
else
    mapfile -t found < <(ls -d "$campaign_run"/*/stage3/diverse_shards)
    [[ ${#found[@]} -eq 1 ]] || { echo "Pass the stage-3 attempt: ${found[*]}" >&2; exit 2; }
    diverse=${found[0]}
fi
ckpts="$campaign_run/gpu03v2/stage1/checkpoints"
reuse="$root/outputs/libero-xemb-ur5e-scale-20261120-r1/gpu09test/test_live"
[[ -d "$diverse/shard-0" && -d "$ckpts" && -d "$reuse" ]]

out="$run/$attempt"
[[ ! -e "$out" ]] || { echo "Attempt $attempt already exists; use a fresh name" >&2; exit 2; }
mkdir -p "$out"
df -h "$out" > "$out/storage.txt"
printf '%s %s %s soft-inner-offline\n' "$pin" "$host" "$$" > "$out/launch.txt"
start=$(date +%s)
trap 'rc=$?; printf "%s %s %s\n" "$rc" "$(date +%s)" "$start" > "$out/exit.txt"' EXIT

export LIBERO_PRO_ROOT="$sim/simulator-source" LIBERO_CONFIG_PATH="$sim/libero-config"
export CROSSARM_RELEASE="$release"
export OPENPI_DATA_HOME="$legacy/cache/openpi-215abfb"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$cv2_overlay:$code/src:$release:$sim/simulator-source:$code/outputs/libero-crossarm-20261120-r1:$code/outputs/libero-xemb-ur5e-sysid-20261120-r1:$code/outputs/libero-xemb-ur5e-scale-20261120-r1:$campaign_code:$here"
export JAX_PLATFORMS=cuda FLUFFY_GPU_EXECUTION=baige XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES=0

libs="$root/setups/libero90-full-action-20261119-r1/desktop-libs02"
glvnd="$libs/prefix/usr/lib/x86_64-linux-gnu"
IFS=: read -ra paths <<< "${LD_LIBRARY_PATH:-}"
driver_paths=""
for path in "${paths[@]}"; do
    if [[ -n "$path" && ! -f "$path/libtorch_python.so" ]]; then driver_paths+=":$path"; fi
done
export LD_LIBRARY_PATH="$glvnd$driver_paths"
export __EGL_VENDOR_LIBRARY_FILENAMES="$libs/nvidia-vendor.json"

date +%s > "$out/gpu-start.txt"
"$root/setups/robocasa365-native-20261101-r1/env/bin/python" -Bu "$code/scripts/[train]gpu_probe.py" \
    --expected-count 1 --purpose training --baige-container --baige-exclusive-training \
    --output "$out/gpu.json" > "$out/probe.log" 2>&1

"$python" -Bu "$here/eval_soft_inner.py" \
    --output "$out" \
    --diverse-dir "$diverse" \
    --reuse-support-dir "$reuse" \
    --campaign-dir "$campaign_code" \
    --checkpoint-dir "$ckpts" \
    > "$out/run.log" 2>&1
date +%s > "$out/gpu-end.txt"
