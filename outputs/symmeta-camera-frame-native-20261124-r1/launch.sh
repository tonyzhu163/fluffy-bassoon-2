#!/usr/bin/env bash
# Parent supplies the pinned checkout, prelaunch decision and remaining one-card budget.
set -euo pipefail
study=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
source "$study/../symmeta-perceiver-retry-20261123-r1/live.sh"
here=$study
[[ $# -ge 3 && $(uname -s) == Linux ]]
mode=$1 pin=$2 attempt=$3
[[ $pin =~ ^[0-9a-f]{40}$ && $attempt =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]
check_tree "$code" "$pin"
if git -C "$code" symbolic-ref -q HEAD >/dev/null; then exit 1; fi
environment
check_tree "$CROSSARM_RELEASE" e5500c690c1498eca1b7050350654964190cdf70
check_tree "$legacy/third_party/openpi-215abfb" 215abfb217dbac7d5f1273282331b9b1866c0479
export PAIRED_DEPS="$setup/source-ff2ddf0a"
check_tree "$PAIRED_DEPS" ff2ddf0ab03e6fbe243979556d8d8f2ebb9bc29c
export PYTHONPATH="$setup/cv2-headless-overlay:$PAIRED_DEPS/src:$CROSSARM_RELEASE:$LIBERO_PRO_ROOT:$PAIRED_DEPS/outputs/libero-crossarm-20261120-r1:$PAIRED_DEPS/outputs/libero-xemb-ur5e-sysid-20261120-r1:$PAIRED_DEPS/outputs/libero-xemb-ur5e-scale-20261120-r1:$PAIRED_DEPS/outputs/$campaign"
export TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1 RAYON_NUM_THREADS=1
shared="$root/outputs/${study##*/}"
out="$shared/$attempt"
args=()
case "$mode" in
  cpu|prepare)
    [[ $(hostname) != job-* ]]
    export CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void JAX_PLATFORMS=cpu
    export LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe EGL_PLATFORM=surfaceless MUJOCO_EGL_DEVICE_ID=0
    export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
    cores=${CAMNATIVE_CORES:-40,41}
    [[ $cores =~ ^[0-9]+,[0-9]+$ ]]
    cmd=(taskset -c "$cores" "$python" -Bu)
    if [[ $mode == cpu ]]; then [[ $# == 3 ]]; else [[ $# == 5 ]]; fi ;;
  gpu)
    [[ $# == 8 && $(hostname) =~ ^job-[a-z0-9]+-master-0$ ]]
    [[ $6 =~ ^[1-9][0-9]*$ && $6 -le 10740 && $7 =~ ^[a-zA-Z0-9_-]+$ && $8 =~ ^[a-zA-Z0-9_-]+$ ]]
    [[ $(cut -d' ' -f1 "$shared/$7/exit.txt") == 0 && $(cut -d' ' -f1 "$shared/$8/exit.txt") == 0 ]]
    "$python" -B - "$shared/$7/cpu-readback.json" "$shared/$8/panel.json" "$pin" "$5" <<'PY'
import json,sys
c,p = [json.load(open(f)) for f in sys.argv[1:3]]
assert c["cpu_fixture"] and c["scored_native_retries"] == 0 and c["pairs"] == 16
assert c["pin"] == p["pin"] == sys.argv[3] and not p["cpu_fixture"] and p["release_sha256"] == sys.argv[4]
PY
    export FLUFFY_GPU_EXECUTION=baige JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-0}
    export __EGL_VENDOR_LIBRARY_FILENAMES="$root/setups/libero90-full-action-20261119-r1/desktop-libs02/nvidia-vendor.json"
    unset LIBGL_ALWAYS_SOFTWARE GALLIUM_DRIVER EGL_PLATFORM
    export CAMNATIVE_PREPARED_PANEL="$shared/$8/panel.json"
    cmd=("$python" -Bu) ;;
  *) exit 2 ;;
esac
if [[ $mode != cpu ]]; then
    [[ $4 == /* && $5 =~ ^[0-9a-f]{64}$ ]]
    args=(--release "$4" --sha256 "$5")
fi
mkdir -p "$shared"
mkdir "$out"
start=$(date +%s)
trap 'rc=$?; echo "$rc $start $(date +%s)" > "$out/exit.txt"' EXIT
df -h "$out" > "$out/storage.txt"
printf '%s\n' "pin=$pin" "host=$(hostname)" "mode=$mode" "source=$PAIRED_DEPS" "attempt=$attempt" > "$out/launch.txt"
export XDG_CACHE_HOME="$out/cache" TMPDIR="$out/tmp" MESA_SHADER_CACHE_DIR="$out/cache/mesa"
export NUMBA_CACHE_DIR="$out/cache/numba" MPLCONFIGDIR="$out/cache/matplotlib"
mkdir -p "$TMPDIR" "$MESA_SHADER_CACHE_DIR" "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"
exec > "$out/run.log" 2>&1
if [[ $mode == gpu ]]; then
    PYTHONPATH="$code/src" "$root/setups/robocasa365-native-20261101-r1/env/bin/python" -Bu \
        "$code/scripts/[train]gpu_probe.py" --expected-count 1 --purpose evaluation --baige-container --output "$out/gpu.json"
    MUJOCO_EGL_DEVICE_ID=$("$python" -B - "$out/gpu.json" "$code/outputs/reground-rotation-reference-20261123-r1/reference.py" <<'PY'
import json, runpy, sys
p=json.load(open(sys.argv[1]))
device, = p["devices"]
assert p["status"]=="passed" and "RTX 4090" in device["cuda_name"].upper()
print(runpy.run_path(sys.argv[2])["egl_device"](0))
PY
)
    export MUJOCO_EGL_DEVICE_ID
    [[ $MUJOCO_EGL_DEVICE_ID =~ ^[0-9]+$ ]]
    remaining=$(($6 - $(date +%s) + start))
    (( remaining > 0 ))
    timeout --kill-after=5s "${remaining}s" "${cmd[@]}" "$study/native.py" "$out" "$pin" "$mode" "${args[@]}"
else
    "${cmd[@]}" "$study/native.py" "$out" "$pin" "$mode" "${args[@]}"
fi
if [[ $mode != prepare ]]; then "${cmd[@]}" "$study/score.py" "$out"; fi
