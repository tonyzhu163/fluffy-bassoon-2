#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/runtime.sh"
pin=$1
out=$2
qual=$3
[[ $(hostname) == job-*-master-0 && -f "$out/launch.txt" ]]
runtime
"$python" -B - "$qual/qualified.json" "$pin" <<'PY'
import json, sys
v=json.load(open(sys.argv[1]))
assert v["pin"] == sys.argv[2] and v["status"] == "passed" and v["fixture"] is True
assert v["episodes"] == 4 and v["paired_starts"] == 1
assert v["rows"][0]["handoffs"] == {"IIWA34": True, "IIWA35": True}
assert v["rows"][0]["controls"] == {"raw": 5, "tuned": 5, "IIWA34": 110, "IIWA35": 120}
PY
export FLUFFY_GPU_EXECUTION=baige JAX_PLATFORMS=cuda
export __EGL_VENDOR_LIBRARY_FILENAMES="$libs/nvidia-vendor.json"
unset LIBGL_ALWAYS_SOFTWARE GALLIUM_DRIVER EGL_PLATFORM
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-0}
PYTHONPATH="$repo/src" "$root/setups/robocasa365-native-20261101-r1/env/bin/python" -Bu \
    "$repo/scripts/[train]gpu_probe.py" --expected-count 1 --purpose evaluation --baige-container --output "$out/gpu.json"
MUJOCO_EGL_DEVICE_ID=$("$python" -B - "$out/gpu.json" "$repo/outputs/reground-rotation-reference-20261123-r1/reference.py" <<'PY'
import json, runpy, sys
probe=json.load(open(sys.argv[1]))
device,=probe["devices"]
assert probe["status"] == "passed" and "RTX 4090" in device["cuda_name"].upper()
print(runpy.run_path(sys.argv[2])["egl_device"](0))
PY
)
[[ $MUJOCO_EGL_DEVICE_ID =~ ^[0-9]+$ ]]
export MUJOCO_EGL_DEVICE_ID
"$python" -Bu "$here/run.py" "$out" "$pin"
