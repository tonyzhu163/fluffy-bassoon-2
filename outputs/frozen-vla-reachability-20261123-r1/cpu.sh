#!/usr/bin/env bash
# bash cpu.sh <script.py> <arguments> ; execute inside the Baige LIBERO CPU runtime.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
root=${BAIGE_PROJECT:-/mowenjie/fluffy-bassoon-2}
setup=$root/setups/libero-meta-learning-20261117-r1
pin=a8a997e7d06bb131d3f9710204c58e106f0f0904
code=${NATIVE_CODE:-$root/outputs/frozen-vla-native-transfer-20261123-r1/src-$pin}
export CROSSARM_RELEASE=$root/sources/libero-pro2-e5500c69
[[ $(git -C "$CROSSARM_RELEASE" rev-parse HEAD) == e5500c690c1498eca1b7050350654964190cdf70 ]]
[[ $(git -C "$code" rev-parse HEAD) == "$pin" ]]
export LIBERO_PRO_ROOT=$setup/simulator-source LIBERO_CONFIG_PATH=$setup/libero-config
export PYTHONPATH="$root/setups/xemb-icml-campaign-20261121-r1/cv2-headless-overlay:$CROSSARM_RELEASE:$LIBERO_PRO_ROOT:$code/src:$code/outputs/libero-crossarm-20261120-r1:$code/outputs/libero-xemb-ur5e-sysid-20261120-r1"
export NATIVE_PANEL=$code/outputs/frozen-vla-native-transfer-20261123-r1
export LD_LIBRARY_PATH="$root/setups/libero90-full-action-20261119-r1/desktop-libs02/prefix/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES='' MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
export LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe LP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export NUMBA_CACHE_DIR=${REACH_CACHE:-$root/outputs/frozen-vla-reachability-20261123-r1/cache}/numba
export MESA_SHADER_CACHE_DIR=${REACH_CACHE:-$root/outputs/frozen-vla-reachability-20261123-r1/cache}/mesa
cd "$here"
exec taskset -c "${REACH_CPUS:-24,25}" "$setup/env/bin/python" -Bu "$@"
