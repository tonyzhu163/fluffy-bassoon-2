#!/usr/bin/env bash
# CPU only, on parent-deployed pin. PHASE_CPUS names two parent-allocated CPU cores.
set -euo pipefail
source "$(dirname "$0")/runtime.sh"
[[ $# == 2 && $1 =~ ^[0-9a-f]{40}$ && $2 =~ ^cpu[A-Za-z0-9_-]+$ && $(uname -s) == Linux ]]
pin=$1
attempt=$2
[[ ${PHASE_CPUS:-44,45} =~ ^([0-9]+),([0-9]+)$ ]]
cores=("${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}")
[[ ${cores[0]} != "${cores[1]}" ]]
pinned_tree "$repo" "$pin"
mkdir -p "$shared"
out="$shared/$attempt"
mkdir "$out"
trap 'rc=$?; echo "$rc" > "$out/exit.txt"' EXIT
runtime
export CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void JAX_PLATFORMS=cpu LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe
export EGL_PLATFORM=surfaceless MUJOCO_EGL_DEVICE_ID=0 __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
printf '%s\n' "$pin" > "$out/pin.txt"
df -h "$out" > "$out/storage.txt"
arms=(IIWA Kinova3)
pids=()
for i in 0 1; do
    arm=${arms[i]}
    mkdir "$out/$arm"
    taskset -c "${cores[i]}" "$python" -Bu "$here/run.py" "$arm" "$out/$arm" --cpu --pin "$pin" > "$out/$arm/run.log" 2>&1 &
    pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=$?; done
exit "$rc"
