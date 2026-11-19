#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
code=$(git -C "$here" rev-parse --show-toplevel)
root=/mowenjie/fluffy-bassoon-2
release="$root/sources/libero-pro2-e5500c69"
runtime="$root/setups/libero-meta-learning-20261117-r1"
out="$root/outputs/$(basename "$here")/${2:?fresh attempt}"
pin=${1:?registered pin}
test "$(git -C "$code" rev-parse HEAD)" = "$pin"
test -z "$(git -C "$code" status --porcelain)"
test "$(git -C "$release" rev-parse HEAD)" = e5500c690c1498eca1b7050350654964190cdf70
archive="$root/outputs/libero-interrupted-practice-20261119-r1"
test "$(cut -d' ' -f1 "$archive/closed01/launch.txt")" = 22da285052a4878c4843d80d9241ba599ae7b66d
test "$(cut -d' ' -f1 "$archive/release02/launch.txt")" = fccdc7472d7031b41c805a03d473bfdaf46cad6a
mkdir "$out"
export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void JAX_PLATFORMS=cpu
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=1 LIBGL_ALWAYS_SOFTWARE=1
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONPATH="$release:$runtime/simulator-source:$code/src"
export LIBERO_PRO_ROOT="$runtime/simulator-source" LIBERO_CONFIG_PATH="$runtime/libero-config"
export XDG_CACHE_HOME="$out/cache" TMPDIR="$out/tmp" NUMBA_CACHE_DIR="$out/cache/numba"
mkdir -p "$TMPDIR" "$NUMBA_CACHE_DIR"
start=$(date +%s)
trap 'rc=$?; echo "$rc $start $(date +%s)" > "$out/exit.txt"' EXIT
{
  echo "$pin $(hostname) $(date -Is)"
  echo "interpreter $(readlink -f "$runtime/env/bin/python")"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES"
  cat "$archive/closed01/launch.txt" "$archive/release02/launch.txt"
  df -h "$out"
} > "$out/launch.txt"
timeout --kill-after=5s 10m "$runtime/env/bin/python" -Bu "$here/gate.py" "$out" > "$out/run.log" 2>&1
test -s "$out/result.json"
