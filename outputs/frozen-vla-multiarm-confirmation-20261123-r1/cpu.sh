#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 && $1 =~ ^[0-9a-f]{40}$ && $2 =~ ^cpu[A-Za-z0-9_-]+$ ]]
here=$(cd "$(dirname "$0")" && pwd -P)
[[ $(uname -s) == Linux && $(git -C "$here" rev-parse HEAD) == "$1" ]]
out=/mowenjie/fluffy-bassoon-2/outputs/$(basename "$here")/$2
mkdir "$out"
source "$here/runtime.sh"
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe
export EGL_PLATFORM=surfaceless MUJOCO_EGL_DEVICE_ID=0 __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
printf '%s %s %s\n' "$1" "$(hostname)" "$(date -Is)" > "$out/launch.txt"
taskset -c 38 "$python" -Bu "$here/qualify.py" IIWA "$out" > "$out/IIWA.log" 2>&1 &
iiwa=$!
taskset -c 39 "$python" -Bu "$here/qualify.py" UR5e "$out" > "$out/UR5e.log" 2>&1 &
ur5e=$!
rc=0
wait "$iiwa" || rc=$?
wait "$ur5e" || rc=$?
echo "$rc" > "$out/exit.txt"
exit "$rc"
