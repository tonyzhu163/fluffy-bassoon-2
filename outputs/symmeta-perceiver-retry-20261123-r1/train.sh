#!/usr/bin/env bash
set -euo pipefail
start=$(date +%s)
pin=${1:?full deployed commit}
attempt=${2:?fresh training attempt name}
[[ $# == 2 && $pin =~ ^[a-f0-9]{40}$ && $attempt =~ ^[[:alnum:]][[:alnum:]_-]*$ ]]
here=$(cd "$(dirname "$0")" && pwd -P)
code=$(git -C "$here" rev-parse --show-toplevel)
root=/mowenjie/fluffy-bassoon-2
legacy=/mowenjie/fluffy-bassoon
host=$(hostname)
[[ $host == job-* ]] || { echo "Requires Baige Distributed Training" >&2; exit 2; }
[[ $(git -C "$code" rev-parse HEAD) == "$pin" && -z $(git -C "$code" status --porcelain) ]]
if git -C "$code" symbolic-ref -q HEAD >/dev/null; then echo "Require pinned detached checkout" >&2; exit 2; fi

run="$root/outputs/$(basename "$here")"
mkdir -p "$run"
out="$run/$attempt"
mkdir "$out"  # Atomically reserve a fresh attempt.
trap 'rc=$?; printf "%s %s %s\n" "$rc" "$(date +%s)" "$start" > "$out/exit.txt"' EXIT
printf '%s %s %s\n' "$pin" "$host" "$$" > "$out/launch.txt"
df -h "$out" > "$out/storage.txt"
printf '%s\n' "tag=perceiver_cont_warm seeds=41,42,43 steps=2500 batch_size=32 gpu_count=1" \
    "code=$code pin=$pin wall_limit_seconds=3600" > "$out/config.txt"

python="$legacy/envs/openpi-215abfb/bin/python"
release="$root/sources/libero-pro2-e5500c69"
sim="$root/setups/libero-meta-learning-20261117-r1"
campaign="$code/outputs/xemb-icml-campaign-20261121-r1"
wider="$code/outputs/symmeta-wider-family-20261123-r1"
overlay="$root/setups/xemb-icml-campaign-20261121-r1/cv2-headless-overlay"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export LIBERO_PRO_ROOT="$sim/simulator-source" LIBERO_CONFIG_PATH="$sim/libero-config"
export CROSSARM_RELEASE="$release" OPENPI_DATA_HOME="$legacy/cache/openpi-215abfb" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONPATH="$overlay:$code/src:$release:$LIBERO_PRO_ROOT:$code/outputs/libero-crossarm-20261120-r1:$code/outputs/libero-xemb-ur5e-sysid-20261120-r1:$code/outputs/libero-xemb-ur5e-scale-20261120-r1:$campaign:$wider"
export FLUFFY_GPU_EXECUTION=baige JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
libs="$root/setups/libero90-full-action-20261119-r1/desktop-libs02"
IFS=: read -ra paths <<< "${LD_LIBRARY_PATH:-}"
driver_paths=""
for path in "${paths[@]}"; do
    if [[ -n $path && ! -f "$path/libtorch_python.so" ]]; then driver_paths+=":$path"; fi
done
export LD_LIBRARY_PATH="$libs/prefix/usr/lib/x86_64-linux-gnu$driver_paths"
export __EGL_VENDOR_LIBRARY_FILENAMES="$libs/nvidia-vendor.json"

# Both steps share one deadline with time for process cleanup.
bounded() {
    local remaining=$(( start + 3600 - 15 - $(date +%s) ))
    (( remaining > 0 )) || return 124
    timeout --kill-after=5s "${remaining}s" "$@"
}
date +%s > "$out/gpu-start.txt"
bounded "$root/setups/robocasa365-native-20261101-r1/env/bin/python" -Bu "$code/scripts/[train]gpu_probe.py" \
    --expected-count 1 --purpose training --baige-container --baige-exclusive-training \
    --output "$out/gpu.json" > "$out/probe.log" 2>&1
bounded "$python" -Bu "$here/export.py" --output "$out" > "$out/run.log" 2>&1
date +%s > "$out/gpu-end.txt"
