#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
code=$(cd "$here/../.." && pwd -P)
source "$code/outputs/frozen-vla-native-transfer-20261123-r1/env.sh"
export CUDA_VISIBLE_DEVICES='' MUJOCO_EGL_DEVICE_ID=0 LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe
export EGL_PLATFORM=surfaceless __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
out=$1
shift
mkdir "$out"
export NUMBA_CACHE_DIR="$out/cache/numba" MESA_SHADER_CACHE_DIR="$out/cache/mesa" NUMEXPR_NUM_THREADS=1
echo "$(git -C "$code" rev-parse HEAD) $(hostname)" > "$out/launch.txt"
exec taskset -c 30,31 "$sim/env/bin/python" -Bu "$here/${1:-cpu.py}" "$out" > "$out/log.txt" 2>&1
