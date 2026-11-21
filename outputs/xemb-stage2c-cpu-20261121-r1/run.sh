#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
code=$(cd "$here/../.." && pwd)
storage=${XEMB_STORAGE:-/mowenjie/fluffy-bassoon-2}
setup=${XEMB_SIM_SETUP:-$storage/setups/libero-meta-learning-20261117-r1}
export CROSSARM_RELEASE=${CROSSARM_RELEASE:-$storage/sources/libero-pro2-e5500c69}
export LIBERO_PRO_ROOT="$setup/simulator-source" LIBERO_CONFIG_PATH="$setup/libero-config"
export PYTHONPATH="$code/outputs/libero-crossarm-20261120-r1:$code/src/fluffy_bassoon/xemb:$CROSSARM_RELEASE:$LIBERO_PRO_ROOT"
export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void JAX_PLATFORMS=cpu
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 LIBGL_ALWAYS_SOFTWARE=1
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
output=${1:-$storage/outputs/xemb-stage2c-cpu-20261121-r1/cpu01}
export NUMBA_CACHE_DIR="$output/cache/numba" XDG_CACHE_HOME="$output/cache"
exec "$setup/env/bin/python" -Bu "$here/qualify.py" "$output"
