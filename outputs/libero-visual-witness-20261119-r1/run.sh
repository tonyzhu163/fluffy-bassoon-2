#!/usr/bin/env bash
set -euo pipefail
pin=$1
attempt=$2
run=libero-visual-witness-20261119-r1
root=/mowenjie/fluffy-bassoon-2
release="$root/sources/libero-pro2-e5500c69"
runtime="$root/setups/libero-meta-learning-20261117-r1"
archive="$root/outputs/libero-bowl-decisions-20261118-r1"
code=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
[[ $(git -C "$code" rev-parse HEAD) == "$pin" && -z $(git -C "$code" status --porcelain) ]]
[[ $(git -C "$release" rev-parse HEAD) == e5500c690c1498eca1b7050350654964190cdf70 ]]
[[ $(cut -d' ' -f1 "$archive/paired01/launch.txt") == f29a4573222e44373453ade72f2aa31c264f41ee ]]
[[ $(cut -d' ' -f1 "$archive/pacing01/launch.txt") == 6ddd4381fd9ef9145e1398f3a602de3e47394193 ]]
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1
export LIBGL_ALWAYS_SOFTWARE=1 MUJOCO_EGL_DEVICE_ID=1 __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
export PYTHONPATH="$release:$runtime/simulator-source:$code/src"
export LIBERO_PRO_ROOT="$runtime/simulator-source" LIBERO_CONFIG_PATH="$runtime/libero-config"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
out="$root/outputs/$run/$attempt"
mkdir "$out"
start=$(date +%s)
trap 'rc=$?; echo "$rc $(date +%s) $start" > "$out/exit.txt"' EXIT
df -h "$out" > "$out/storage.txt"
printf '%s %s %s\n' "$pin" "$(hostname)" "$$" > "$out/launch.txt"
timeout --kill-after=10s 30m "$runtime/env/bin/python" -Bu "$code/outputs/$run/replay.py" "$out" > "$out/run.log" 2>&1
