#!/usr/bin/env bash
# Parent-owned single-card distributed evaluation. Run inside the allocated job only.
set -euo pipefail
study=$(cd "$(dirname "$0")" && pwd -P)
source "$study/../symmeta-perceiver-retry-20261123-r1/live.sh"
pin=${1:?deployed scoring revision}
attempt=${2:?fresh native attempt}
release=${3:?parent-released full-head-release.json}
release_sha=${4:?parent-verified sha256}
resets_file=${5:?qualified physical query reset reference}
resets_sha=${6:?parent-verified reset sha256}
wall=${7:?remaining distributed seconds}
[[ $# == 7 && $(hostname) =~ ^job-[a-z0-9]+-master-0$ && $attempt =~ ^[a-z0-9_-]+$ ]]
[[ $wall =~ ^[1-9][0-9]*$ && $wall -le 9000 && $release_sha =~ ^[a-f0-9]{64}$ && $resets_sha =~ ^[a-f0-9]{64}$ ]]
shared="$root/outputs/${study##*/}"
[[ "$release" == "$shared"/*/full-head-release.json && "$resets_file" == "$shared"/*/query-resets.json ]]
exec 9>"$shared/native-budget.lock"
flock -n 9
shopt -s nullglob
prior=("$shared"/*/gpu-wall.seconds)
used=0
for consumed in "${prior[@]}"; do read -r n < "$consumed"; used=$((used+n)); done
(( used+wall<=9000 && wall>10 ))
out="$shared/$attempt"
mkdir "$out"
start=$(date +%s)
trap 'rc=$?; end=$(date +%s); echo $((end-start)) > "$out/gpu-wall.seconds"; echo "$rc $start $end" > "$out/exit.txt"' EXIT
check_tree "$code" "$pin"
environment
check_tree "$CROSSARM_RELEASE" e5500c690c1498eca1b7050350654964190cdf70
check_tree "$legacy/third_party/openpi-215abfb" 215abfb217dbac7d5f1273282331b9b1866c0479
export PAIRED_DEPS="$setup/source-wave2b"
check_tree "$PAIRED_DEPS" 465f33dc657ceab341dbceaf6b98355b94810059
export PYTHONPATH="$setup/cv2-headless-overlay:$PAIRED_DEPS/src:$CROSSARM_RELEASE:$LIBERO_PRO_ROOT:$PAIRED_DEPS/outputs/libero-crossarm-20261120-r1:$PAIRED_DEPS/outputs/libero-xemb-ur5e-sysid-20261120-r1:$PAIRED_DEPS/outputs/libero-xemb-ur5e-scale-20261120-r1:$PAIRED_DEPS/outputs/$campaign"
export JAX_PLATFORMS=cuda FLUFFY_GPU_EXECUTION=baige CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export __EGL_VENDOR_LIBRARY_FILENAMES="$root/setups/libero90-full-action-20261119-r1/desktop-libs02/nvidia-vendor.json"
unset LIBGL_ALWAYS_SOFTWARE GALLIUM_DRIVER EGL_PLATFORM
export XDG_CACHE_HOME="$out/cache" TMPDIR="$out/tmp" MPLCONFIGDIR="$out/cache/mpl" NUMBA_CACHE_DIR="$out/cache/numba"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"
echo "$pin $(hostname) $* $start" > "$out/launch.txt"
df -h "$out" > "$out/storage.txt"
export study out code python root release release_sha resets_file resets_sha
remaining=$((wall-$(date +%s)+start-6))
(( remaining>0 ))
timeout -k 4s "${remaining}s" bash -eu -o pipefail <<'JOB' > "$out/run.log" 2>&1
PYTHONPATH="$code/src" "$root/setups/robocasa365-native-20261101-r1/env/bin/python" -Bu \
    "$code/scripts/[train]gpu_probe.py" --expected-count 1 --purpose evaluation --baige-container --output "$out/gpu.json"
MUJOCO_EGL_DEVICE_ID=$("$python" -B - "$out/gpu.json" "$code/outputs/reground-rotation-reference-20261123-r1/reference.py" <<'PY'
import json,runpy,sys
p=json.load(open(sys.argv[1]))
device,=p["devices"]
assert p["status"]=="passed" and "RTX 4090" in device["cuda_name"].upper()
print(runpy.run_path(sys.argv[2])["egl_device"](0))
PY
)
export MUJOCO_EGL_DEVICE_ID
[[ $MUJOCO_EGL_DEVICE_ID =~ ^[0-9]+$ ]]
exec "$python" -Bu "$study/native_panel.py" gpu "$out" "$release" "$release_sha" "$resets_file" "$resets_sha"
JOB
