#!/usr/bin/env bash
# Each invocation is one fresh attempt. Root submits gpu mode through Distributed Training.
set -euo pipefail
if [[ ${1:-} == gpu ]]; then
    # Bound the probe, model load, inference and readback within the scheduler wall limit.
    shift
    exec timeout --kill-after=10s 14340s bash "$0" timed-gpu "$@"
fi
here=$(cd "$(dirname "$0")" && pwd -P)
repo=$(cd "$here/../../.." && pwd -P)
root=/mowenjie/fluffy-bassoon-2
shared="$root/outputs/frozen-vla-cotraining-20261124-r1/evaluation"
mode=${1:?cpu or gpu}
pin=${2:?full executable commit}
attempt=${3:?fresh attempt name}
[[ $pin =~ ^[0-9a-f]{40}$ && $attempt =~ ^[a-zA-Z0-9_-]+$ && $(uname -s) == Linux ]]
[[ $(git -C "$repo" rev-parse HEAD) == "$pin" && -z $(git -C "$repo" status --porcelain --untracked-files=no) ]]
if [[ $mode == cpu ]]; then
    [[ $# == 3 && $attempt == cpu* ]]
elif [[ $mode == timed-gpu ]]; then
    [[ $# == 5 && $attempt == gpu* && $(hostname) == job-*-master-0 && $5 =~ ^[0-9a-f]{64}$ ]]
    [[ -f $4/fit.json && -f $4/heads.npz && $(sha256sum "$4/fit.json" | cut -d' ' -f1) == "$5" ]]
else
    exit 2
fi
native_pin=a8a997e7d06bb131d3f9710204c58e106f0f0904
code="$root/outputs/frozen-vla-native-transfer-20261123-r1/src-$native_pin"
[[ $(git -C "$code" rev-parse HEAD) == "$native_pin" && -z $(git -C "$code" status --porcelain --untracked-files=no) ]]
source "$code/outputs/frozen-vla-native-transfer-20261123-r1/env.sh"
[[ $(git -C "$CROSSARM_RELEASE" rev-parse HEAD) == e5500c690c1498eca1b7050350654964190cdf70 ]]
[[ -z $(git -C "$CROSSARM_RELEASE" status --porcelain --untracked-files=no) ]]
out="$shared/$attempt"
mkdir -p "$shared"
mkdir "$out"
exec > >(tee "$out/console.log") 2>&1
start=$(date +%s)
trap 'rc=$?; echo "$rc $start $(date +%s)" > "$out/exit.txt"' EXIT
df -h "$out" > "$out/storage.txt"
echo "host=$(hostname) pin=$pin native=$native_pin mode=$mode" > "$out/launch.txt"
export PYTHONPATH="$here/../model:$here/../data:$PYTHONPATH"
export TMPDIR="$out/tmp" XDG_CACHE_HOME="$out/cache" NUMBA_CACHE_DIR="$out/cache/numba" MPLCONFIGDIR="$out/cache/matplotlib"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"
if [[ $mode == cpu ]]; then
    export CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void JAX_PLATFORMS=cpu LIBGL_ALWAYS_SOFTWARE=1
    export GALLIUM_DRIVER=llvmpipe EGL_PLATFORM=surfaceless MUJOCO_EGL_DEVICE_ID=0
    export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
    taskset -c 40,41 "$python" -Bu "$here/run.py" "$out" "$pin" --fixture
else
    export FLUFFY_GPU_EXECUTION=baige JAX_PLATFORMS=cuda NCCL_P2P_DISABLE=1
    export __EGL_VENDOR_LIBRARY_FILENAMES="$libs/nvidia-vendor.json" CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-0}
    unset LIBGL_ALWAYS_SOFTWARE GALLIUM_DRIVER EGL_PLATFORM
    PYTHONPATH="$code/src" "$root/setups/robocasa365-native-20261101-r1/env/bin/python" -Bu \
        "$code/scripts/[train]gpu_probe.py" --expected-count 1 --baige-container --purpose evaluation --output "$out/gpu.json"
    MUJOCO_EGL_DEVICE_ID=$("$python" -B - "$out/gpu.json" "$code/outputs/reground-rotation-reference-20261123-r1/reference.py" <<'PY'
import json, runpy, sys
probe = json.load(open(sys.argv[1]))
device, = probe["devices"]
name = device["cuda_name"].upper()
if not (probe["status"] == "passed" and (name in ("NVIDIA RPBZZZ6", "NVIDIA GEFORCE RTX 4090") or "RTX PRO 6000" in name)):
    raise ValueError(name)
print(runpy.run_path(sys.argv[2])["egl_device"](0))
PY
)
    [[ $MUJOCO_EGL_DEVICE_ID =~ ^[0-9]+$ ]]
    export MUJOCO_EGL_DEVICE_ID
    "$python" -Bu "$here/run.py" "$out" "$pin" --fit "$4" --fit-sha "$5"
fi
