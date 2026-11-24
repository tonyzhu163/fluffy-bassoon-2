#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
code=$(cd "$here/../.." && pwd -P)
revision=$(git -C "$code" rev-parse HEAD)
test "$revision" = "${1:?expected deployed commit}"
source "$code/outputs/frozen-vla-native-transfer-20261123-r1/env.sh"
out="$root/outputs/$(basename "$here")/${2:-cpu01}"
mkdir "$out"
exec >"$out/run.log" 2>&1
trap 'echo $? > "$out/exit.txt"' EXIT
export PYTHONPATH="$PYTHONPATH:$code/outputs/frozen-vla-native-controller-20261123-r1"
export CUDA_VISIBLE_DEVICES='' NUMEXPR_NUM_THREADS=1 JAX_PLATFORMS=cpu
export NUMBA_CACHE_DIR="$out/numba-cache" EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
export GALLIUM_DRIVER=llvmpipe MESA_SHADER_CACHE_DIR="$out/mesa-cache"
printf '%s\n' "$revision" "$(hostname)" "$(date -Is)" > "$out/launch.txt"
df -h "$out" >> "$out/launch.txt"
cat /sys/fs/cgroup/cpu.max /sys/fs/cgroup/cpuset.cpus.effective >> "$out/launch.txt"
command -v quota >/dev/null && quota -s >> "$out/launch.txt" || true
{ time taskset -c 48,49 "$sim/env/bin/python" -Bu "$here/replay.py" "$out" 2>>"$out/run.log"; } 2>"$out/replay-time.txt"
{ time taskset -c 48,49 "$sim/env/bin/python" -Bu "$here/score.py" "$out" 2>>"$out/run.log"; } 2>"$out/scoring-time.txt"
date -Is >> "$out/launch.txt"
