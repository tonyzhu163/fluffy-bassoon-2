#!/usr/bin/env bash
# Invoke in the allocated job for GPU mode. The parent owns submission and the aggregate budget.
set -euo pipefail
mode=${1:?cpu or gpu}; pin=${2:?full code revision}; attempt=${3:?fresh attempt name}
if [[ $mode == gpu && ${MATCHED_TIMED:-0} == 0 ]]; then
    exec timeout --kill-after=5s 5300s env MATCHED_TIMED=1 bash "$0" "$@"
fi
here=$(cd "$(dirname "$0")" && pwd -P)
repo=$(git -C "$here" rev-parse --show-toplevel)
root=/mowenjie/fluffy-bassoon-2
run=$root/outputs/frozen-vla-cotraining-20261124-r1
manifest=$run/data/dataset-v1/manifest.json
[[ $(uname -s) == Linux && $pin =~ ^[0-9a-f]{40}$ && $attempt =~ ^[a-zA-Z0-9_-]+$ ]]
[[ $(git -C "$repo" rev-parse HEAD) == "$pin" && -z $(git -C "$repo" status --porcelain) ]]
case "$mode" in
cpu) [[ $# == 3 && $attempt == cpu* ]] ;;
gpu) [[ $# == 5 && $attempt == gpu* && $(hostname) == job-*-master-0 && $5 =~ ^[0-9a-f]{64}$ ]]
     [[ $(sha256sum "$4/fit.json" | cut -d' ' -f1) == "$5" ]] ;;
*) exit 2 ;;
esac
native=a8a997e7d06bb131d3f9710204c58e106f0f0904
code=$root/outputs/frozen-vla-native-transfer-20261123-r1/src-$native
source "$code/outputs/frozen-vla-native-transfer-20261123-r1/env.sh"
[[ $(git -C "$code" rev-parse HEAD) == "$native" && -z $(git -C "$code" status --porcelain --untracked-files=no) ]]
[[ $(git -C "$CROSSARM_RELEASE" rev-parse HEAD) == e5500c690c1498eca1b7050350654964190cdf70 ]]
[[ -z $(git -C "$CROSSARM_RELEASE" status --porcelain --untracked-files=no) ]]
out=$run/matched-task/$attempt
mkdir "$out"
exec > >(tee "$out/console.log") 2>&1
start=$(date +%s)
trap 'rc=$?; echo "$rc $start $(date +%s)" > "$out/exit.txt"' EXIT
mkdir -p "$out/tmp" "$out/cache"
export TMPDIR="$out/tmp" XDG_CACHE_HOME="$out/cache" NUMBA_CACHE_DIR="$out/cache/numba" MPLCONFIGDIR="$out/cache/matplotlib"
export PYTHONPATH="$here/../evaluation:$here/../model:$here/../data:$PYTHONPATH" PYTHONHASHSEED=0
df -h "$out" > "$out/storage.txt"
if [[ $mode == cpu ]]; then
    export CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void JAX_PLATFORMS=cpu LIBGL_ALWAYS_SOFTWARE=1
    export GALLIUM_DRIVER=llvmpipe EGL_PLATFORM=surfaceless MUJOCO_EGL_DEVICE_ID=0
    export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
    taskset -c 42,43 "$python" -Bu "$here/matched.py" "$out" "$pin" "$manifest" --fixture
else
    export FLUFFY_GPU_EXECUTION=baige JAX_PLATFORMS=cuda NCCL_P2P_DISABLE=1
    export __EGL_VENDOR_LIBRARY_FILENAMES="$libs/nvidia-vendor.json" CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-0}
    unset LIBGL_ALWAYS_SOFTWARE GALLIUM_DRIVER EGL_PLATFORM
    probe=$root/setups/robocasa365-native-20261101-r1/env/bin/python
    PYTHONPATH="$code/src" "$probe" -Bu "$code/scripts/[train]gpu_probe.py" \
        --expected-count 1 --baige-container --purpose evaluation --output "$out/gpu.json"
    MUJOCO_EGL_DEVICE_ID=$("$python" -B - "$out/gpu.json" "$code/outputs/reground-rotation-reference-20261123-r1/reference.py" <<'PY'
import json, runpy, sys
p = json.load(open(sys.argv[1]))
device, = p["devices"]
name = device["cuda_name"].upper()
if p["status"] != "passed" or not (name in ("NVIDIA RPBZZZ6", "NVIDIA GEFORCE RTX 4090") or "RTX PRO 6000" in name):
    raise ValueError(name)
print(runpy.run_path(sys.argv[2])["egl_device"](0))
PY
)
    [[ $MUJOCO_EGL_DEVICE_ID =~ ^[0-9]+$ ]]
    export MUJOCO_EGL_DEVICE_ID
    "$python" -Bu "$here/matched.py" "$out" "$pin" "$manifest" --fit "$4" --fit-sha "$5"
fi
