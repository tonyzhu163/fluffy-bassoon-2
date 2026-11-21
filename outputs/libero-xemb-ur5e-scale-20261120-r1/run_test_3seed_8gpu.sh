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
shard_start=$3
src_attempt=${4:-gpu01}
base_attempt=${5:-gpu13baselines}
reuse_attempt=${6:-gpu09test}
out_tag=${7:-gpu14test576}
host=$(hostname)
[[ "$attempt" =~ ^[a-z0-9]+$ ]]
[[ "$host" == job-* ]] || { echo "Require Baige Distributed Training container" >&2; exit 2; }
[[ $(git -C "$code" rev-parse HEAD) == "$pin" && -z $(git -C "$code" status --porcelain) ]]

out="$run/$attempt"
shared="$run/$out_tag/test_live"
mkdir -p "$out/logs" "$shared"
printf '%s %s %s official-test-3seed-576 shards=%s..%s\n' "$pin" "$host" "$$" "$shard_start" "$((shard_start + 7))" > "$out/launch.txt"
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

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
date +%s > "$out/gpu-start.txt"
"$root/setups/robocasa365-native-20261101-r1/env/bin/python" -Bu "$code/scripts/[train]gpu_probe.py" \
    --expected-count 8 --purpose training --baige-container --baige-exclusive-training \
    --output "$out/gpu.json" > "$out/probe.log" 2>&1

pids=()
for g in 0 1 2 3 4 5 6 7; do
    s=$((shard_start + g))
    CUDA_VISIBLE_DEVICES="$g" MUJOCO_EGL_DEVICE_ID="$g" \
        timeout --kill-after=20s 115m "$python" -Bu "$here/eval_test_3seed_576.py" run-shard \
        --train-dir "$run/$src_attempt/train" \
        --baselines-dir "$run/$base_attempt/train" \
        --reuse-support-dir "$run/$reuse_attempt/test_live" \
        --output-dir "$shared" \
        --shard-idx "$s" \
        > "$out/logs/shard-$s.log" 2>&1 &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

# If all 16 shards are present, automatically merge into official-test-3seed-576-summary.json
count=$(ls "$shared"/shard-*/shard-test.json 2>/dev/null | wc -l || echo 0)
if [[ "$count" -eq 16 ]]; then
    CUDA_VISIBLE_DEVICES=0 "$python" -Bu "$here/eval_test_3seed_576.py" merge \
        --output-dir "$shared" --num-shards 16 > "$out/logs/merge.log" 2>&1
fi
