#!/usr/bin/env bash
set -euo pipefail
audit=$(cd "$(dirname "$0")" && pwd)
root=$(dirname "$audit")
export REGROUND_RUNTIME_ROOT="$root"
source "$root/runtime/activate.sh"
export CUDA_VISIBLE_DEVICES="" LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=1
export MESA_SHADER_CACHE_DISABLE=true NUMBA_CACHE_DIR="$audit/cpu-cache"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="$audit/cpu-cache"
export XDG_CACHE_HOME="$audit/cpu-cache"
cd "$audit"
exec "$root/env/bin/python" -Bu reset_inspection.py
