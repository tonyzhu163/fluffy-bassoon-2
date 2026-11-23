#!/usr/bin/env bash
set -euo pipefail
[[ $# == 3 && $1 =~ ^(frames|native|real-policy|readback|layouts|audit)$ && ! -e $3 ]]
mode=$1
export RETRY_PIN=$2
here=$(cd "$(dirname "$0")" && pwd -P)
mkdir -p "$3"
attempt=$(cd "$3" && pwd -P)
source "$here/env.sh"
export CUDA_VISIBLE_DEVICES='' LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe MUJOCO_EGL_DEVICE_ID=0
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
export MESA_SHADER_CACHE_DISABLE=true XDG_CACHE_HOME="$attempt/cache"
exec bash "$control/$name/transport/with-cache.sh" "$py" -Bu "$here/driver.py" cpu "$mode" "$attempt"
