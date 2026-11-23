#!/usr/bin/env bash
# One parent-submitted Baige Distributed Training job, one allocated accelerator, one-hour scheduler wall.
set -euo pipefail
[[ $# == 2 && $1 =~ ^[0-9a-f]{40}$ && $2 =~ ^[A-Za-z0-9_-]+$ ]]
here=$(cd "$(dirname "$0")" && pwd -P)
pin=$1
attempt=$2
[[ $(git -C "$here" rev-parse HEAD) == "$pin" && -z $(git -C "$here" status --porcelain) ]]
[[ $(uname -s) == Linux && $(hostname) == job-*-master-0 ]]
out="/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-supported-correction-20261123-r1/$attempt"
mkdir "$out"
exec > >(tee "$out/console.log") 2>&1
start=$(date +%s)
trap 'rc=$?; echo "$rc $start $(date +%s)" > "$out/exit.txt"' EXIT
source "$here/bindings.sh"
echo "host=$(hostname) adapter_pin=$pin native_bindings_pin=$native_pin start=$start" > "$out/launch.txt"
df -h "$out" > "$out/storage.txt"
export FLUFFY_GPU_EXECUTION=baige JAX_PLATFORMS=cuda NCCL_P2P_DISABLE=1
export __EGL_VENDOR_LIBRARY_FILENAMES="$libs/nvidia-vendor.json"
unset LIBGL_ALWAYS_SOFTWARE
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-0}
"$root/setups/robocasa365-native-20261101-r1/env/bin/python" -Bu "$code/scripts/[train]gpu_probe.py" \
  --expected-count 1 --baige-container --purpose evaluation --output "$out/gpu.json"
MUJOCO_EGL_DEVICE_ID=$("$python" -B - "$out/gpu.json" "$code/outputs/reground-rotation-reference-20261123-r1/reference.py" <<'PY'
import json,runpy,sys
probe=json.load(open(sys.argv[1]))
device,=probe['devices']
name=device['cuda_name'].upper()
assert probe['status']=='passed' and (name in ('NVIDIA RPBZZZ6','NVIDIA GEFORCE RTX 4090') or 'RTX PRO 6000' in name)
print(runpy.run_path(sys.argv[2])['egl_device'](0))
PY
)
export MUJOCO_EGL_DEVICE_ID
# Leave scheduler teardown time under the hard one-hour allocation.
remaining=$((3500 - $(date +%s) + start))
(( remaining > 0 ))
timeout --kill-after=5s "${remaining}s" "$python" -Bu "$here/native.py" "$out" "$matrix"
