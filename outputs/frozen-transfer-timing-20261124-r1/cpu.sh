#!/usr/bin/env bash
set -euo pipefail
role=${1:?fit or test}
[[ $role == fit || $role == test ]]
base=${2:?fresh scratch experiment directory}
here=$(cd "$(dirname "$0")" && pwd -P)
out="$base/$role"
mkdir -p "$base"
mkdir "$out"
data=/mowenjie/fluffy-bassoon-2
sim="$data/setups/libero-meta-learning-20261117-r1"
code="$data/outputs/frozen-vla-gripper-20261123-r1/source02"
export CROSSARM_RELEASE="$data/sources/libero-pro2-e5500c69" LIBERO_PRO_ROOT="$sim/simulator-source" LIBERO_CONFIG_PATH="$sim/libero-config"
[[ $(git -C "$code" rev-parse HEAD) == c30e7f76989563a808fc5985b0679d18a5066e4c ]]
[[ $(git -C "$CROSSARM_RELEASE" rev-parse HEAD) == e5500c690c1498eca1b7050350654964190cdf70 ]]
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYOPENGL_PLATFORM=egl MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json LIBGL_ALWAYS_SOFTWARE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$base/cache" NUMBA_CACHE_DIR="$base/cache/numba" MPLCONFIGDIR="$base/cache/matplotlib" TMPDIR="$base/tmp"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$TMPDIR"
lib="$data/setups/libero90-full-action-20261119-r1/desktop-libs02"
export LD_LIBRARY_PATH="$lib/prefix/usr/lib/x86_64-linux-gnu:$lib/prefix/lib/x86_64-linux-gnu"
export PYTHONPATH="$code/outputs/libero-crossarm-20261120-r1:$CROSSARM_RELEASE:$LIBERO_PRO_ROOT"
{
    date -Is
    hostname
    uname -sr
    sha256sum "$here/PLAN.md" "$here/diagnostic.py" "$here/cpu.sh"
    cat "$sim/env/pyvenv.cfg"
    df -h "$out"
} > "$out/launch.txt"
taskset -c 42,43 "$sim/env/bin/python" -Bu "$here/diagnostic.py" "$role" "$out" > "$out/run.log" 2>&1
