#!/usr/bin/env bash
set -euo pipefail
attempt=${1:-gpu01}
here=$(cd "$(dirname "$0")" && pwd -P)
code=/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-native-controller-20261123-r1/src-7518022c88711815c8207b4397e4ce84036c6134
source "$code/outputs/frozen-vla-native-transfer-20261123-r1/env.sh"
export PYTHONPATH="$here:$PYTHONPATH:$code/outputs/frozen-vla-native-controller-20261123-r1"
out="$here/$attempt"
mkdir -p "$out"
start=$(date +%s)
trap 'echo "$? $start $(date +%s)" > "$out/exit.txt"' EXIT
exec > >(tee "$out/log.txt") 2>&1
echo "$(hostname) $(date -Is)" > "$out/launch.txt"
cd "$here"
export FLUFFY_GPU_EXECUTION=baige JAX_PLATFORMS=cuda NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-0}
export __EGL_VENDOR_LIBRARY_FILENAMES="$libs/nvidia-vendor.json"
export NUMBA_CACHE_DIR="$out/cache/numba"
unset LIBGL_ALWAYS_SOFTWARE
"$root/setups/robocasa365-native-20261101-r1/env/bin/python" -Bu "$code/scripts/[train]gpu_probe.py" \
    --expected-count 1 --baige-container --purpose evaluation --output "$out/gpu.json"
egl=$("$python" -B - "$out/gpu.json" "$code/outputs/reground-rotation-reference-20261123-r1/reference.py" <<'PY'
import json, runpy, sys
p = json.load(open(sys.argv[1]))
assert p["status"] == "passed"
print(runpy.run_path(sys.argv[2])["egl_device"](0))
PY
)
export MUJOCO_EGL_DEVICE_ID="$egl"
timeout --kill-after=5s 5400s "$python" -Bu "$here/live.py" "$out"
