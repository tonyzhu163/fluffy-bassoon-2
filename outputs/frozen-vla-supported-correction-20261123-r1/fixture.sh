#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
out=${1:?new absolute output directory}
mkdir "$out"
source "$here/bindings.sh"
export CUDA_VISIBLE_DEVICES='' MUJOCO_EGL_DEVICE_ID=0 LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe
export EGL_PLATFORM=surfaceless __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
exec taskset -c 0,1 "$python" -Bu "$here/native.py" "$out" "$matrix" --fixture "$root/outputs/frozen-vla-native-transfer-20261123-r1/gpu01/libero_goal-3-14-Kinova3.json"
