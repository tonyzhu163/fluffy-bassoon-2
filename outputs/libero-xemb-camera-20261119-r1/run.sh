#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
code=$(git -C "$here" rev-parse --show-toplevel)
pin=${1:?verified experiment commit}
out=${2:?fresh output path}
root=/mowenjie/fluffy-bassoon-2
runtime="$root/setups/libero-meta-learning-20261117-r1"
release="$root/sources/libero-pro2-e5500c69"
test "$(git -C "$code" rev-parse HEAD)" = "$pin"
test -z "$(git -C "$code" status --porcelain)"
test "$(git -C "$release" rev-parse HEAD)" = e5500c690c1498eca1b7050350654964190cdf70
mkdir "$out"
export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=1 LIBGL_ALWAYS_SOFTWARE=1
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONPATH="$release:$runtime/simulator-source" LIBERO_PRO_ROOT="$runtime/simulator-source"
export LIBERO_CONFIG_PATH="$runtime/libero-config" XDG_CACHE_HOME="$out/cache" TMPDIR="$out/tmp"
export NUMBA_CACHE_DIR="$out/cache/numba" MPLCONFIGDIR="$out/cache/matplotlib"
mkdir -p "$TMPDIR" "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"
start=$(date +%s)
trap 'rc=$?; echo "$rc $start $(date +%s)" > "$out/exit.txt"' EXIT
split="$code/outputs/libero-meta-learning-20261117-r1/SPLITS.json"
{
  echo "$pin $(hostname) $(date -Is)"
  git -C "$release" rev-parse HEAD
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  sha256sum "$split" "$here/gate.py" "$here/run.sh" "$release"/libero_remote/environment/{libero,observations}.py \
    "$runtime/simulator-source/libero/libero/envs/"{env_wrapper,bddl_base_domain}.py
} > "$out/launch.txt"
timeout --kill-after=5s 10m "$runtime/env/bin/python" -Bu "$here/gate.py" "$out" "$release" "$split" > "$out/run.log" 2>&1
test -s "$out/result.json"
