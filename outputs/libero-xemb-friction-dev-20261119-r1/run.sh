#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
code=$(git -C "$here" rev-parse --show-toplevel)
root=/mowenjie/fluffy-bassoon-2
release="$root/sources/libero-pro2-e5500c69"
runtime="$root/setups/libero-meta-learning-20261117-r1"
out="$root/outputs/$(basename "$here")/cpu01"
archive="$root/outputs/libero-retry-headroom-20261119-r1/paired01"
pin=$1
[[ $(git -C "$code" rev-parse HEAD) == "$pin" && -z $(git -C "$code" status --porcelain) ]]
[[ $(git -C "$release" rev-parse HEAD) == e5500c690c1498eca1b7050350654964190cdf70 ]]
[[ $(cut -d' ' -f1 "$archive/launch.txt") == c45aa965db73ad0644382da578ff03a30b2083d7 ]]
mkdir "$out"
export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void JAX_PLATFORMS=cpu
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=1 LIBGL_ALWAYS_SOFTWARE=1
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONHASHSEED=0
export PYTHONPATH="$release:$runtime/simulator-source:$code/src"
export LIBERO_PRO_ROOT="$runtime/simulator-source" LIBERO_CONFIG_PATH="$runtime/libero-config"
export XDG_CACHE_HOME="$out/cache" TMPDIR="$out/tmp" NUMBA_CACHE_DIR="$out/cache/numba"
mkdir -p "$TMPDIR" "$NUMBA_CACHE_DIR"
start=$(date +%s)
trap 'rc=$?; echo "$rc $start $(date +%s)" > "$out/exit.txt"' EXIT
{
  echo "$pin $(hostname) $(date -Is) pid=$$"
  echo "interpreter=$(readlink -f "$runtime/env/bin/python") CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES"
  cat "$archive/launch.txt"
  sha256sum "$here/replay.py" "$release/libero_remote/environment/libero.py" "$release/assets/task_manifest.json"
} > "$out/launch.txt"
timeout --kill-after=5s 15m "$runtime/env/bin/python" -Bu "$here/replay.py" "$out" > "$out/native.log" 2>&1
test "$(wc -l < "$out/branches.jsonl")" -eq 32
