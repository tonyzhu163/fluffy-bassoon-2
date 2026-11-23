#!/usr/bin/env bash
set -euo pipefail
# Parent sets the allocation wall limit to 60 minutes from allocation, one GPU, zero retries.
# Bound the entire command including probes and verification. GNU timeout also signals descendants.
if [[ "${1:-}" != --timed ]]; then
    exec timeout --kill-after=5s 3500s bash "$0" --timed "$@"
fi
shift
[[ $# == 2 && $1 =~ ^[0-9a-f]{40}$ && $2 =~ ^[A-Za-z0-9_-]+$ ]]
here=$(cd "$(dirname "$0")" && pwd -P)
code=$(cd "$here/../.." && pwd -P)
[[ $(hostname) == job-*-master-0 && $(uname -s) == Linux ]]
[[ $(git -C "$code" rev-parse HEAD) == "$1" && -z $(git -C "$code" status --porcelain --untracked-files=no) ]]
source "$code/outputs/frozen-vla-native-transfer-20261123-r1/env.sh"
[[ $(git -C "$CROSSARM_RELEASE" rev-parse HEAD) == e5500c690c1498eca1b7050350654964190cdf70 ]]
[[ $(git -C /mowenjie/fluffy-bassoon/third_party/openpi-215abfb rev-parse HEAD) == 215abfb217dbac7d5f1273282331b9b1866c0479 ]]
out="$root/outputs/frozen-vla-source-view-20261123-r1/$2"
mkdir "$out"
exec > >(tee "$out/console.log") 2>&1
start=$(date +%s)
trap 'rc=$?; echo "$rc $start $(date +%s)" > "$out/exit.txt"' EXIT
echo "host=$(hostname) pin=$1 openpi=215abfb217dbac7d5f1273282331b9b1866c0479" > "$out/launch.txt"
export FLUFFY_GPU_EXECUTION=baige JAX_PLATFORMS=cuda NCCL_P2P_DISABLE=1
export __EGL_VENDOR_LIBRARY_FILENAMES="$libs/nvidia-vendor.json"
unset LIBGL_ALWAYS_SOFTWARE
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-0}
"$root/setups/robocasa365-native-20261101-r1/env/bin/python" -Bu "$code/scripts/[train]gpu_probe.py" \
    --expected-count 1 --baige-container --purpose evaluation --output "$out/gpu.json"
MUJOCO_EGL_DEVICE_ID=$("$python" -B - "$out/gpu.json" "$code/outputs/reground-rotation-reference-20261123-r1/reference.py" <<'PY'
import json, runpy, sys
probe=json.load(open(sys.argv[1]))
device,=probe["devices"]
name=device["cuda_name"].upper()
if probe["status"]!="passed" or not (name in ("NVIDIA RPBZZZ6","NVIDIA GEFORCE RTX 4090") or "RTX PRO 6000" in name):
    raise ValueError(probe)
print(runpy.run_path(sys.argv[2])["egl_device"](0))
PY
)
export MUJOCO_EGL_DEVICE_ID
(cd "$here/evidence/cpu01" && sha256sum --check SHA256SUMS)
"$python" -Bu "$here/motor.py" "$out"
