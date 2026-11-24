#!/usr/bin/env bash
# Parent calls this from the single-card distributed allocation.
set -euo pipefail
study=$(cd "$(dirname "$0")" && pwd -P)
source "$study/../symmeta-perceiver-retry-20261123-r1/live.sh"
body=${1:?UR5e or Kinova3}
pin=${2:?deployed evaluator commit}
attempt=${3:?fresh attempt}
release_sha=${4:?parent-confirmed release sha256}
wall=${5:?remaining allocation seconds}
[[ $# == 5 && $(hostname) =~ ^job-[a-z0-9]+-master-0$ && $attempt =~ ^[A-Za-z0-9_-]+$ ]]
[[ $release_sha =~ ^[0-9a-f]{64}$ && $pin =~ ^[0-9a-f]{40}$ && $wall =~ ^[1-9][0-9]*$ ]]
case "$body" in
    UR5e) prior=5000e5ee526d0ee3b34162440327cfc3e4831482; dep=ff2ddf0ab03e6fbe243979556d8d8f2ebb9bc29c; src=source-ff2ddf0a; cap=5400 ;;
    Kinova3) prior=d2b307bc4ca8c0cf723b3aaba31755d14a2f8303; dep=465f33dc657ceab341dbceaf6b98355b94810059; src=source-wave2b; cap=4200 ;;
    *) exit 2 ;;
esac
(( wall <= cap ))
shared="$root/outputs/${study##*/}"
release="$shared/release01/$body.json"
out="$shared/$attempt"
mkdir "$out"
start=$(date +%s)
trap 'rc=$?; end=$(date +%s); echo "$rc $start $end" > "$out/exit.txt"' EXIT
printf '%s\n' "pin=$pin host=$(hostname) body=$body attempt=$attempt allocated_seconds=$wall start=$start" > "$out/launch.txt"
df -h "$out" > "$out/storage.txt"
environment
export REALNATIVE_ORIGINAL="$root/setups/symmeta-perceiver-followups-20261123-r1/src-${prior:0:12}"
export PAIRED_DEPS="$setup/$src"
# Restrict the cleanliness check to tracked code. Outputs and weight arrays already have independent seals.
for lock in "$code $pin" "$REALNATIVE_ORIGINAL $prior" "$PAIRED_DEPS $dep" \
    "$CROSSARM_RELEASE e5500c690c1498eca1b7050350654964190cdf70" \
    "$legacy/third_party/openpi-215abfb 215abfb217dbac7d5f1273282331b9b1866c0479"; do
    read -r path ref <<< "$lock"
    [[ $(git -C "$path" rev-parse HEAD) == "$ref" ]]
done
! git -C "$code" symbolic-ref -q HEAD >/dev/null || exit 1
git --no-optional-locks -C "$code" diff --quiet HEAD -- "outputs/${study##*/}" "outputs/symmeta-perceiver-retry-20261123-r1/live.sh" 'scripts/[[]train]gpu_probe.py'
git --no-optional-locks -C "$REALNATIVE_ORIGINAL" diff --quiet HEAD -- outputs/symmeta-camera-frame-native-20261124-r1 outputs/symmeta-camera-independent-body-20261124-r1 outputs/symmeta-perceiver-retry-20261123-r1
git --no-optional-locks -C "$PAIRED_DEPS" diff --quiet HEAD -- src outputs/libero-crossarm-20261120-r1 outputs/libero-xemb-ur5e-sysid-20261120-r1 outputs/libero-xemb-ur5e-scale-20261120-r1 "outputs/$campaign"
export PYTHONPATH="$setup/cv2-headless-overlay:$PAIRED_DEPS/src:$CROSSARM_RELEASE:$LIBERO_PRO_ROOT:$PAIRED_DEPS/outputs/libero-crossarm-20261120-r1:$PAIRED_DEPS/outputs/libero-xemb-ur5e-sysid-20261120-r1:$PAIRED_DEPS/outputs/libero-xemb-ur5e-scale-20261120-r1:$PAIRED_DEPS/outputs/$campaign"
export FLUFFY_GPU_EXECUTION=baige JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1 RAYON_NUM_THREADS=1
export __EGL_VENDOR_LIBRARY_FILENAMES="$root/setups/libero90-full-action-20261119-r1/desktop-libs02/nvidia-vendor.json"
unset LIBGL_ALWAYS_SOFTWARE GALLIUM_DRIVER EGL_PLATFORM
export XDG_CACHE_HOME="$out/cache" TMPDIR="$out/tmp" MPLCONFIGDIR="$out/cache/mpl" NUMBA_CACHE_DIR="$out/cache/numba"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"
export code root out study body pin release release_sha python
remaining=$((wall-$(date +%s)+start-6))
(( remaining>0 ))
timeout -k 4s "${remaining}s" bash -eu -o pipefail <<'JOB' > "$out/run.log" 2>&1
PYTHONPATH="$code/src" "$root/setups/robocasa365-native-20261101-r1/env/bin/python" -Bu \
    "$code/scripts/[train]gpu_probe.py" --expected-count 1 --purpose evaluation --baige-container --output "$out/gpu.json"
MUJOCO_EGL_DEVICE_ID=$("$python" -B - "$out/gpu.json" "$code/outputs/reground-rotation-reference-20261123-r1/reference.py" <<'PY'
import json,runpy,sys
p=json.load(open(sys.argv[1]))
d,=p["devices"]
assert p["status"]=="passed" and "RTX 4090" in d["cuda_name"].upper()
print(runpy.run_path(sys.argv[2])["egl_device"](0))
PY
)
export MUJOCO_EGL_DEVICE_ID
[[ $MUJOCO_EGL_DEVICE_ID =~ ^[0-9]+$ ]]
exec "$python" -Bu "$study/rollout.py" "$body" "$out" "$release" "$release_sha" "$pin"
JOB
