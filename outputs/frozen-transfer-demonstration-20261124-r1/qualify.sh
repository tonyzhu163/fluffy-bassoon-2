#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/runtime.sh"
[[ $# == 2 && $1 =~ ^[0-9a-f]{40}$ && $2 =~ ^cpu[A-Za-z0-9_-]+$ ]]
pin=$1
out="$shared/$2"
pinned_tree "$repo" "$pin"
mkdir -p "$shared"
mkdir "$out"
trap 'rc=$?; echo "$rc" > "$out/exit.txt"' EXIT
runtime
export CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void JAX_PLATFORMS=cpu LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe
export EGL_PLATFORM=surfaceless MUJOCO_EGL_DEVICE_ID=0 __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
df -h "$out" > "$out/storage.txt"
taskset -c 38,39 "$python" -Bu "$here/run.py" "$out" "$pin" --cpu > "$out/run.log" 2>&1
