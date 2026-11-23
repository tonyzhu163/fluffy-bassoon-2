#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
arm=$1
out=$2
[[ $(hostname) == job-*-master-0 && ( $arm == IIWA || $arm == UR5e ) ]]
[[ $out == /mowenjie/fluffy-bassoon-2/outputs/$(basename "$here")/gpu01-"$arm" && -f $out/launch.txt ]]
source "$here/runtime.sh"
export FLUFFY_GPU_EXECUTION=baige JAX_PLATFORMS=cuda NCCL_P2P_DISABLE=1
export __EGL_VENDOR_LIBRARY_FILENAMES="$libs/nvidia-vendor.json"
unset LIBGL_ALWAYS_SOFTWARE GALLIUM_DRIVER
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-0}
"$root/setups/robocasa365-native-20261101-r1/env/bin/python" -Bu "$code/scripts/[train]gpu_probe.py" \
    --expected-count 1 --baige-container --purpose evaluation --output "$out/gpu.json"
MUJOCO_EGL_DEVICE_ID=$("$python" -B - "$out/gpu.json" "$code/outputs/reground-rotation-reference-20261123-r1/reference.py" <<'PY'
import json, runpy, sys
probe = json.load(open(sys.argv[1]))
device, = probe["devices"]
name = device["cuda_name"].upper()
if probe["status"] != "passed" or not (name in ("NVIDIA RPBZZZ6", "NVIDIA GEFORCE RTX 4090") or "RTX PRO 6000" in name):
    raise ValueError(f"Single-card evaluation admission failed: {name}")
print(runpy.run_path(sys.argv[2])["egl_device"](0))
PY
)
[[ $MUJOCO_EGL_DEVICE_ID =~ ^[0-9]+$ ]]
export MUJOCO_EGL_DEVICE_ID
printf 'native_git=%s\npolicy_python=%s\nsim_python=%s\ncv2_overlay=%s\nopenpi_checkpoint=%s\nCUDA_VISIBLE_DEVICES=%s\nEGL_device=%s\nEGL_vendor_file=%s\n' \
    "$base_pin" "$python" "$sim/env/bin/python" "$campaign/cv2-headless-overlay" \
    /mowenjie/fluffy-bassoon/inputs/pi05-libero-openpi-20261027-r1 "$CUDA_VISIBLE_DEVICES" \
    "$MUJOCO_EGL_DEVICE_ID" "$__EGL_VENDOR_LIBRARY_FILENAMES" > "$out/runtime.txt"
"$python" -Bu "$here/live.py" "$arm" "$out"
"$python" -Bu "$here/report.py" "$arm" "$out"
