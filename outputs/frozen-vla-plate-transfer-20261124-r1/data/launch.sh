#!/usr/bin/env bash
set -euo pipefail
mode=${1:?cpu or gpu}
pin=${2:?full Git revision}
attempt=${3:?fresh attempt}
[[ $# == 3 && $pin =~ ^[a-f0-9]{40}$ && $attempt =~ ^(cpu|gpu)[0-9]+$ && $attempt == "$mode"* ]]
if [[ $mode == gpu && ${PLATE_DATA_TIMED:-0} == 0 ]]; then
    exec timeout --kill-after=10s 7000s env PLATE_DATA_TIMED=1 bash "$0" "$@"
fi
[[ $(uname -s) == Linux ]]
here=$(cd "$(dirname "$0")" && pwd -P)
repo=$(git -C "$here" rev-parse --show-toplevel)
root=/mowenjie/fluffy-bassoon-2
run=$root/outputs/frozen-vla-plate-transfer-20261124-r1/data
out=$run/$attempt
mkdir "$out"
exec > >(tee "$out/console.log") 2>&1
start=$(date +%s)
trap 'rc=$?; echo "$rc $start $(date +%s)" > "$out/exit.txt"' EXIT
printf '%s\n' "$(hostname)" "$pin" "$mode" "$attempt" "$start" > "$out/launch.txt"
df -B1 "$out" > "$out/storage.txt"
quota -s >> "$out/storage.txt" 2>/dev/null || true
[[ $(git -C "$repo" rev-parse HEAD) == "$pin" && -z $(git -C "$repo" status --porcelain --untracked-files=no) ]]
native=a8a997e7d06bb131d3f9710204c58e106f0f0904
code=$root/outputs/frozen-vla-native-transfer-20261123-r1/src-$native
[[ $(git -C "$code" rev-parse HEAD) == "$native" && -z $(git -C "$code" status --porcelain --untracked-files=no) ]]
source "$code/outputs/frozen-vla-native-transfer-20261123-r1/env.sh"
[[ $(git -C "$CROSSARM_RELEASE" rev-parse HEAD) == e5500c690c1498eca1b7050350654964190cdf70 ]]
[[ -z $(git -C "$CROSSARM_RELEASE" status --porcelain --untracked-files=no) ]]
export PYTHONPATH="$repo/outputs/frozen-vla-cotraining-20261124-r1/data:$PYTHONPATH"
export TMPDIR="$out/tmp" XDG_CACHE_HOME="$out/cache" NUMBA_CACHE_DIR="$out/cache/numba" MPLCONFIGDIR="$out/cache/matplotlib"
mkdir -p "$TMPDIR" "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"
export PYTHONHASHSEED=0 NUMEXPR_NUM_THREADS=1
py=$sim/env/bin/python
args=()
case "$mode" in
cpu)
    cores=26,27
    export CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void LIBGL_ALWAYS_SOFTWARE=1
    export GALLIUM_DRIVER=llvmpipe EGL_PLATFORM=surfaceless MUJOCO_EGL_DEVICE_ID=0
    export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
    args+=(--cpu-smoke)
    ;;
gpu)
    [[ $(hostname) == job-*-master-0 && ! -e $run/dataset-v1 ]]
    cores=$(python -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0))[:2])))')
    taskset -pc "$cores" $$ >/dev/null
    export FLUFFY_GPU_EXECUTION=baige CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    export __EGL_VENDOR_LIBRARY_FILENAMES="$libs/nvidia-vendor.json"
    unset LIBGL_ALWAYS_SOFTWARE GALLIUM_DRIVER EGL_PLATFORM
    probe=$root/setups/robocasa365-native-20261101-r1/env/bin/python
    PYTHONPATH="$repo/src" "$probe" -Bu "$repo/scripts/[train]gpu_probe.py" \
        --expected-count 1 --baige-container --baige-exclusive-training --purpose training --output "$out/gpu.json"
    egl=$("$py" -B - "$out/gpu.json" "$code/outputs/reground-rotation-reference-20261123-r1/reference.py" <<'PY'
import json, runpy, sys
probe = json.load(open(sys.argv[1]))
device, = probe["devices"]
assert probe["status"] == "passed" and "4090" in device["cuda_name"]
print(runpy.run_path(sys.argv[2])["egl_device"](0))
PY
)
    [[ $egl =~ ^[0-9]+$ ]]
    export MUJOCO_EGL_DEVICE_ID="$egl"
    ;;
*) exit 2 ;;
esac
taskset -c "$cores" "$py" -Bu "$here/collect.py" "$out" "$pin" "${args[@]}"
