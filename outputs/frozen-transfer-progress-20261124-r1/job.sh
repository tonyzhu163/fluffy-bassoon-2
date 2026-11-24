#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/runtime.sh"
pin=$1
arm=$2
out=$3
qual=$4
[[ $(hostname) == job-*-master-0 && ( $arm == IIWA || $arm == Kinova3 ) && -f "$out/launch.txt" ]]
runtime
"$python" -B - "$qual" "$pin" <<'PY'
import json, sys
from pathlib import Path
for arm in ("IIWA","Kinova3"):
    d = json.loads((Path(sys.argv[1])/arm/"qualified.json").read_bytes())
    assert d["status"] == "passed" and d["complete_cases"] == 8 and d["episodes"] == 32
    assert d["origin"]["pin"] == sys.argv[2] and d["origin"]["arm"] == arm and d["origin"]["fixture"]
PY
export FLUFFY_GPU_EXECUTION=baige JAX_PLATFORMS=cuda NCCL_P2P_DISABLE=1
export __EGL_VENDOR_LIBRARY_FILENAMES="$libs/nvidia-vendor.json"
unset LIBGL_ALWAYS_SOFTWARE GALLIUM_DRIVER EGL_PLATFORM
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-0}
PYTHONPATH="$repo/src" "$root/setups/robocasa365-native-20261101-r1/env/bin/python" -Bu \
    "$repo/scripts/[train]gpu_probe.py" --expected-count 1 --purpose evaluation --baige-container --output "$out/gpu.json"
MUJOCO_EGL_DEVICE_ID=$("$python" -B - "$out/gpu.json" "$repo/outputs/reground-rotation-reference-20261123-r1/reference.py" <<'PY'
import json, runpy, sys
p=json.load(open(sys.argv[1]))
d,=p["devices"]
name=d["cuda_name"].upper()
if p["status"] != "passed" or not (name in ("NVIDIA RPBZZZ6","NVIDIA GEFORCE RTX 4090") or "RTX PRO 6000" in name):
    raise ValueError(f"Unqualified evaluation allocation {name}")
print(runpy.run_path(sys.argv[2])["egl_device"](0))
PY
)
[[ $MUJOCO_EGL_DEVICE_ID =~ ^[0-9]+$ ]]
export MUJOCO_EGL_DEVICE_ID
"$python" -Bu "$here/run.py" "$arm" "$out" --pin "$pin"
"$python" -Bu "$here/readback.py" "--${arm,,}" "$out" --out "$out/result.json"
