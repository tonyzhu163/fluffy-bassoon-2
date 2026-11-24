#!/usr/bin/env bash
set -euo pipefail
mode=${1:?cpu or gpu}
pin=${2:?full code revision}
[[ $# == 2 && $(uname -s) == Linux && $pin =~ ^[0-9a-f]{40}$ ]]
if [[ $mode == gpu && ${SHAM_TIMED:-0} == 0 ]]; then
    exec timeout --kill-after=5s 2990s env SHAM_TIMED=1 bash "$0" "$@"
fi
here=$(cd "$(dirname "$0")" && pwd -P)
root=/mowenjie/fluffy-bassoon-2
run=$root/outputs/frozen-vla-matched-history-sham-20261124-r1
parent=$root/outputs/frozen-vla-cotraining-20261124-r1
matched_pin=c51e767770bc48827a1d6d3837cddefe1b7a1d6c
fit_pin=29e9e5e5209db26ce0cbf695387c64cdf79b98fa
native=a8a997e7d06bb131d3f9710204c58e106f0f0904
code=$root/outputs/frozen-vla-native-transfer-20261123-r1/src-$native
matched_repo=$parent/matched-task/code-$matched_pin
fit_repo=$parent/model/code-$fit_pin
for pair in "$here:$pin" "$matched_repo:$matched_pin" "$fit_repo:$fit_pin" "$code:$native"; do
    tree=${pair%:*}
    [[ $(git -C "$tree" rev-parse HEAD) == "${pair##*:}" && -z $(git -C "$tree" status --porcelain --untracked-files=no) ]]
done
source "$code/outputs/frozen-vla-native-transfer-20261123-r1/env.sh"
[[ $(git -C "$CROSSARM_RELEASE" rev-parse HEAD) == e5500c690c1498eca1b7050350654964190cdf70 && -z $(git -C "$CROSSARM_RELEASE" status --porcelain --untracked-files=no) ]]
matched_code=$matched_repo/outputs/frozen-vla-cotraining-20261124-r1
fit_code=$fit_repo/outputs/frozen-vla-cotraining-20261124-r1
export PYTHONPATH="$fit_code/model:$fit_code/data:$matched_code/matched-task:$matched_code/evaluation:$PYTHONPATH"
fit="$parent/model/gpu02"
read -r rc _ < "$fit/exit.txt"
[[ $rc == 0 && $(sha256sum "$fit/fit.json" | cut -d' ' -f1) == 4855546591381f149d7509a6af3d45db32e9e82a732788c8376d504f19417dbc ]]
case "$mode" in
    cpu) out="$run/cpu-$pin" ;;
    gpu) [[ $(hostname) == job-*-master-0 ]]; out="$run/gpu01" ;;
    *) exit 2 ;;
esac
mkdir "$out"
exec > >(tee "$out/console.log") 2>&1
start=$(date +%s)
trap 'rc=$?; echo "$rc $start $(date +%s)" > "$out/exit.txt"' EXIT
mkdir "$out/tmp" "$out/cache"
export TMPDIR="$out/tmp" XDG_CACHE_HOME="$out/cache" NUMBA_CACHE_DIR="$out/cache/numba" MPLCONFIGDIR="$out/cache/matplotlib" PYTHONHASHSEED=0
df -h "$out" > "$out/storage.txt"
if [[ $mode == cpu ]]; then
    export CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void JAX_PLATFORMS=cpu LIBGL_ALWAYS_SOFTWARE=1
    export GALLIUM_DRIVER=llvmpipe EGL_PLATFORM=surfaceless MUJOCO_EGL_DEVICE_ID=0
    export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
    taskset -c 36,37 "$python" -Bu "$here/sham.py" "$out" "$pin" "$fit" "$parent/data/dataset-v1/manifest.json" --fixture
    taskset -c 36,37 "$python" -Bu "$here/audit.py" "$out"
else
    export FLUFFY_GPU_EXECUTION=baige JAX_PLATFORMS=cuda NCCL_P2P_DISABLE=1
    export __EGL_VENDOR_LIBRARY_FILENAMES="$libs/nvidia-vendor.json" CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-0}
    unset LIBGL_ALWAYS_SOFTWARE GALLIUM_DRIVER EGL_PLATFORM
    probe=$root/setups/robocasa365-native-20261101-r1/env/bin/python
    PYTHONPATH="$code/src" "$probe" -Bu "$code/scripts/[train]gpu_probe.py" \
        --expected-count 1 --baige-container --purpose evaluation --output "$out/gpu.json"
    MUJOCO_EGL_DEVICE_ID=$("$python" -B - "$out/gpu.json" "$code/outputs/reground-rotation-reference-20261123-r1/reference.py" <<'PY'
import json, runpy, sys
p = json.load(open(sys.argv[1]))
device, = p["devices"]
if p["status"] != "passed" or device["cuda_name"].upper() not in ("NVIDIA RPBZZZ6", "NVIDIA GEFORCE RTX 4090"):
    raise ValueError(device)
print(runpy.run_path(sys.argv[2])["egl_device"](0))
PY
)
    [[ $MUJOCO_EGL_DEVICE_ID =~ ^[0-9]+$ ]]
    export MUJOCO_EGL_DEVICE_ID
    "$python" -Bu "$here/sham.py" "$out" "$pin" "$fit" "$parent/data/dataset-v1/manifest.json"
fi
