#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
code=$(cd "$here/../.." && pwd -P)
source "$code/outputs/frozen-vla-native-transfer-20261123-r1/env.sh"
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu MUJOCO_EGL_DEVICE_ID=0 LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe EGL_PLATFORM=surfaceless
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
out=$root/outputs/frozen-vla-source-view-20261123-r1/${1:?fresh name}
mkdir "$out"
export NUMBA_CACHE_DIR=$out/cache/numba MESA_SHADER_CACHE_DIR=$out/cache/mesa
git -C "$code" rev-parse HEAD > "$out/pin.txt"
exec taskset -c 36,37 "$python" -Bu "$here/motor.py" "$out" --cpu-smoke >"$out/console.log" 2>&1
