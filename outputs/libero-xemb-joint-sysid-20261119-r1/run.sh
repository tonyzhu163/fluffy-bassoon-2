#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
code=$(git -C "$here" rev-parse --show-toplevel)
base=/mowenjie/fluffy-bassoon-2
runtime="$base/setups/libero-meta-learning-20261117-r1"
release="$base/sources/libero-pro2-e5500c69"
pilot=libero-xemb-sysid-20261119-r1
archive="$base/outputs/$pilot/gpu02col/archive"
out=${2:?fresh attempt directory}
test "$(git -C "$code" rev-parse HEAD)" = "${1:?registered commit}"
test -z "$(git -C "$code" status --porcelain)"
test "$(git -C "$release" rev-parse HEAD)" = e5500c690c1498eca1b7050350654964190cdf70
mkdir "$out"
mkdir -p "$out/cache/numba" "$out/tmp" "$out/cache/matplotlib"
export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void JAX_PLATFORMS=cpu
export LIBGL_ALWAYS_SOFTWARE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=1
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1
export XDG_CACHE_HOME="$out/cache" MPLCONFIGDIR="$out/cache/matplotlib" NUMBA_CACHE_DIR="$out/cache/numba" TMPDIR="$out/tmp"
export LIBERO_PRO_ROOT="$runtime/simulator-source" LIBERO_CONFIG_PATH="$runtime/libero-config"
export PYTHONPATH="$code/src:$code/outputs/$pilot:$release:$runtime/simulator-source"
start=$(date +%s)
trap 'rc=$?; echo "$rc $start $(date +%s)" > "$out/exit.txt"' EXIT
{
  date -Is
  hostname
  git -C "$code" rev-parse HEAD
  git -C "$release" rev-parse HEAD
  "$runtime/env/bin/python" -V
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  sha256sum "$archive/archive.json" "$here/"{run.sh,experiment.py,joint_reader.py} "$code/outputs/$pilot/sim_worker.py" "$code/src/fluffy_bassoon/xemb/"{embodiment.py,readers.py}
} > "$out/launch.txt"
timeout --kill-after=5s 5m "$runtime/env/bin/python" -Bu "$here/experiment.py" "$archive" "$out" > "$out/run.log" 2>&1
test -s "$out/scores.json"
