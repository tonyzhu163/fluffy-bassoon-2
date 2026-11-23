#!/usr/bin/env bash
set -eu -o pipefail
pin=${1:?full Git revision}
attempt=${2:?fresh attempt name}
[[ $attempt =~ ^[a-zA-Z0-9_-]+$ ]]
here=$(cd "$(dirname "$0")" && pwd -P)
code=$(git -C "$here" rev-parse --show-toplevel)
[[ $(git -C "$code" rev-parse HEAD) == "$pin" ]]
[[ -z $(git -C "$code" status --porcelain --untracked-files=no) ]]
data=/mowenjie/fluffy-bassoon-2
out="$data/outputs/$(basename "$here")/$attempt"
mkdir "$out"
exec > >(tee "$out/run.log") 2>&1
start=$(date +%s)
trap 'echo "$? $start $(date +%s)" > "$out/exit.txt"' EXIT
echo "$pin $(hostname) $(date -Is)" > "$out/launch.txt"
df -h "$out"
sim="$data/setups/libero-meta-learning-20261117-r1"
export CROSSARM_RELEASE="$data/sources/libero-pro2-e5500c69" LIBERO_PRO_ROOT="$sim/simulator-source" LIBERO_CONFIG_PATH="$sim/libero-config"
[[ $(git -C "$CROSSARM_RELEASE" rev-parse HEAD) == e5500c690c1498eca1b7050350654964190cdf70 ]]
export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu PYOPENGL_PLATFORM=egl MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json LIBGL_ALWAYS_SOFTWARE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$out/cache" NUMBA_CACHE_DIR="$out/cache/numba" MPLCONFIGDIR="$out/cache/matplotlib" TMPDIR="$out/tmp"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$TMPDIR"
lib="$data/setups/libero90-full-action-20261119-r1/desktop-libs02"
export LD_LIBRARY_PATH="$lib/prefix/usr/lib/x86_64-linux-gnu:$lib/prefix/lib/x86_64-linux-gnu"
export PYTHONPATH="$code/outputs/libero-crossarm-20261120-r1:$CROSSARM_RELEASE:$LIBERO_PRO_ROOT"
stage=${3:-native}
[[ $stage == native || $stage == grasp ]]
taskset -c "${GRIPPER_CPU_PAIR:-18,19}" "$sim/env/bin/python" -Bu "$here/$stage.py" "$out"
