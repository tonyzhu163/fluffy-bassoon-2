#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 && $1 =~ ^[0-9a-f]{40}$ && $2 =~ ^[A-Za-z0-9_-]+$ ]]
pin=$1
attempt=$2
here=$(cd "$(dirname "$0")" && pwd -P)
code=$(cd "$here/../.." && pwd -P)
[[ $(git -C "$code" rev-parse HEAD) == "$pin" && $(uname -s) == Linux && $(hostname) == job-*-master-0 ]]
[[ -z $(git -C "$code" status --porcelain --untracked-files=no) ]]
source "$here/env.sh"
[[ $(git -C "$CROSSARM_RELEASE" rev-parse HEAD) == e5500c690c1498eca1b7050350654964190cdf70 ]]
[[ $(git -C /mowenjie/fluffy-bassoon/third_party/openpi-215abfb rev-parse HEAD) == 215abfb217dbac7d5f1273282331b9b1866c0479 ]]
export FLUFFY_GPU_EXECUTION=baige JAX_PLATFORMS=cuda NCCL_P2P_DISABLE=1
export __EGL_VENDOR_LIBRARY_FILENAMES="$libs/nvidia-vendor.json"
unset LIBGL_ALWAYS_SOFTWARE
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-0}
out="$root/outputs/$(basename "$here")/$attempt"
mkdir -p "$(dirname "$out")"
mkdir "$out"
exec > >(tee "$out/console.log") 2>&1
start=$(date +%s)
trap 'rc=$?; echo "$rc $start $(date +%s)" > "$out/exit.txt"' EXIT
echo "host=$(hostname) pin=$pin" > "$out/launch.txt"
df -h "$out" > "$out/storage.txt"
"$root/setups/robocasa365-native-20261101-r1/env/bin/python" -Bu "$code/scripts/[train]gpu_probe.py" \
    --expected-count 1 --baige-container --purpose evaluation --output "$out/gpu.json"
MUJOCO_EGL_DEVICE_ID=$("$python" -B - "$out/gpu.json" "$code/outputs/reground-rotation-reference-20261123-r1/reference.py" <<'PY'
import json, runpy, sys
probe = json.load(open(sys.argv[1]))
d, = probe["devices"]
name = d["cuda_name"].upper()
assert probe["status"] == "passed" and (name in ("NVIDIA RPBZZZ6", "NVIDIA GEFORCE RTX 4090") or "RTX PRO 6000" in name)
print(runpy.run_path(sys.argv[2])["egl_device"](0))
PY
)
export MUJOCO_EGL_DEVICE_ID
matrix="$root/outputs/frozen-vla-motion-models-20261123-r1/cpu01/matrices.json"
echo "20a9d6800e10404ee23a542c70a3ba1fb077daeccd56cf7fea93555c7add64cf  $matrix" | sha256sum -c -
remaining=$((3600 - $(date +%s) + start))
((remaining > 0))
timeout --kill-after=5s "${remaining}s" "$python" -Bu "$here/native_rollout.py" "$out" --matrices "$matrix"
"$python" -Bu "$here/summarize.py" "$out"
