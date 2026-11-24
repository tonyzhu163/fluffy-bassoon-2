#!/usr/bin/env bash
# All modes run only in a parent-deployed checkout, one sim per process.
set -euo pipefail
study=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
native_code=$(cd "$study/../.." && pwd -P)
source "$study/../symmeta-perceiver-retry-20261123-r1/live.sh"
mode=${1:?cpu or gpu}; pin=${2:?native revision}; attempt=${3:?fresh attempt}; shift 3
[[ $(uname -s) == Linux && $pin =~ ^[0-9a-f]{40}$ && $attempt =~ ^[A-Za-z0-9_-]+$ ]]
shared="$root/outputs/${study##*/}"
mkdir -p "$shared"
out="$shared/$attempt"
mkdir "$out"
start=$(date +%s)
trap 'rc=$?; e=$(date +%s); echo "$rc $start $e" > "$out/exit.txt"; if [[ $mode == gpu ]]; then echo "$((e-start))" > "$out/gpu-wall.seconds"; fi' EXIT
check_tree "$native_code" "$pin"
runtime_pin=a8a997e7d06bb131d3f9710204c58e106f0f0904
code="$root/outputs/frozen-vla-native-transfer-20261123-r1/src-$runtime_pin"
check_tree "$code" "$runtime_pin"
source "$code/outputs/frozen-vla-native-transfer-20261123-r1/env.sh"
check_tree "$CROSSARM_RELEASE" e5500c690c1498eca1b7050350654964190cdf70
check_tree "$legacy/third_party/openpi-215abfb" 215abfb217dbac7d5f1273282331b9b1866c0479
export XDG_CACHE_HOME="$out/cache" TMPDIR="$out/tmp" MESA_SHADER_CACHE_DIR="$out/cache/mesa"
export NUMBA_CACHE_DIR="$out/cache/numba" MPLCONFIGDIR="$out/cache/matplotlib"
export TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1
mkdir -p "$TMPDIR" "$MESA_SHADER_CACHE_DIR" "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"
df -h "$out" > "$out/storage.txt"
df -i "$out" >> "$out/storage.txt"
echo "$pin $(hostname) $mode $attempt $* $start" > "$out/launch.txt"
exec > "$out/run.log" 2>&1
case "$mode" in
cpu|prepare)
    [[ $(hostname) != job-* && ${1:?CPU index} =~ ^[0-9]+$ ]]
    core=$1; shift
    (( (core<35 || core>39) && (core<42 || core>45) ))
    taskset -c "$core" true
    for sample in 1 2; do
        awk -v c="cpu$core" '$1==c {print}' /proc/stat >> "$out/idle.txt"
        [[ $sample == 2 ]] || sleep 2
    done
    awk 'NR==1 {i=$5;t=$2+$3+$4+$5+$6+$7+$8+$9} NR==2 {n=$2+$3+$4+$5+$6+$7+$8+$9-t; if(n<=0 || ($5-i)/n<0.90) exit 1} END {if(NR!=2)exit 1}' "$out/idle.txt"
    export CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void JAX_PLATFORMS=cpu
    export LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe EGL_PLATFORM=surfaceless MUJOCO_EGL_DEVICE_ID=0
    export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
    if [[ $mode == cpu ]]; then
        [[ $# == 0 ]]
        taskset -c "$core" "$python" -Bu "$study/rollout.py" cpu "$out"
    else
        [[ $# == 2 ]]
        taskset -c "$core" "$python" -Bu "$study/prepare.py" "$out" "$1" "$2"
    fi ;;
gpu)
    [[ $# == 5 && $(hostname) =~ ^job-[a-z0-9]+-master-0$ ]]
    arm=$1; panel=$2; panel_sha=$3; wall=$4; cpureceipt=$5
    [[ $arm == UR5e || $arm == IIWA ]]
    [[ $panel == "$shared"/*/panel.json && $panel_sha =~ ^[a-f0-9]{64}$ && $wall =~ ^[1-9][0-9]*$ ]]
    [[ $cpureceipt == "$shared"/*/completed.json && $(cut -d' ' -f1 "$(dirname "$cpureceipt")/exit.txt") == 0 ]]
    "$python" -B - "$cpureceipt" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert r["cpu_fixture"] and r["groups"]==8 and r["episodes"]==24 and r["scored_native_retries"]==0
PY
    # Distinct robot jobs may overlap. Failed attempts on the same robot consume its reservation.
    exec 9>"$shared/budget-$arm.lock"; flock -n 9
    used=0
    shopt -s nullglob
    for usedfile in "$shared"/*/gpu-wall.seconds; do
        if [[ $(cat "${usedfile%/*}/budget-arm.txt") == "$arm" ]]; then
            read -r prior < "$usedfile"; used=$((used+prior))
        fi
    done
    echo "$arm" > "$out/budget-arm.txt"
    (( wall<=3540 && used+wall<=3540 ))
    export FLUFFY_GPU_EXECUTION=baige JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
    export __EGL_VENDOR_LIBRARY_FILENAMES="$libs/nvidia-vendor.json"
    unset LIBGL_ALWAYS_SOFTWARE GALLIUM_DRIVER EGL_PLATFORM
    export study native_code python root out arm panel panel_sha
    remaining=$((wall-$(date +%s)+start-6))
    (( remaining>0 ))
    timeout -k 5s "${remaining}s" bash -euo pipefail <<'JOB'
PYTHONPATH="$native_code/src" "$root/setups/robocasa365-native-20261101-r1/env/bin/python" -Bu \
    "$native_code/scripts/[train]gpu_probe.py" --expected-count 1 --purpose evaluation --baige-container --output "$out/gpu.json"
MUJOCO_EGL_DEVICE_ID=$("$python" -B - "$out/gpu.json" "$native_code/outputs/reground-rotation-reference-20261123-r1/reference.py" <<'PY'
import json,runpy,sys
p=json.load(open(sys.argv[1]))
device,=p["devices"]
assert p["status"]=="passed" and "RTX 4090" in device["cuda_name"].upper()
print(runpy.run_path(sys.argv[2])["egl_device"](0))
PY
)
export MUJOCO_EGL_DEVICE_ID
[[ $MUJOCO_EGL_DEVICE_ID =~ ^[0-9]+$ ]]
exec "$python" -Bu "$study/rollout.py" gpu "$out" --arm "$arm" --panel "$panel" --sha256 "$panel_sha"
JOB
    ;;
*) exit 2 ;;
esac
