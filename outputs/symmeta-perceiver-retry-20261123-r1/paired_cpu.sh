#!/usr/bin/env bash
# Linux CPU qualification against real archived inputs and software-rendered native resets.
set -euo pipefail
source "$(dirname "$0")/live.sh"
[[ $# == 1 && $(uname -s) == Linux && $1 == /* ]]
out=$1
mkdir "$out"
environment
export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void JAX_PLATFORMS=cpu LIBGL_ALWAYS_SOFTWARE=1
export MUJOCO_EGL_DEVICE_ID=0 __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
export NUMBA_CACHE_DIR="$out/cache/numba" MPLCONFIGDIR="$out/cache/matplotlib"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"
prediction_sha=$("$python" -B "$here/paired_qualify.py" fixture "$out")
echo "Temporary reversed schema fixture $prediction_sha"
pids=()
for panel in "${panels[@]}"; do
    (
        mkdir "$out/$panel"
        invoke "$panel" live.py prepare "$panel" "$out" --predictions-sha "$prediction_sha" --cpu-fixture > "$out/$panel/prepare.log" 2>&1
        invoke "$panel" paired_qualify.py reset "$out" --panel "$panel" > "$out/$panel/reset.log" 2>&1
    ) &
    pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=$?; done
(( rc == 0 )) || exit "$rc"
"$python" -B "$here/paired_records.py" seal "$out" --predictions-sha "$prediction_sha"
"$python" -B "$here/paired_qualify.py" checks "$out"
