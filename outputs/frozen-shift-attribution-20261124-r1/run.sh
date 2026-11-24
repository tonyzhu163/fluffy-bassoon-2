#!/usr/bin/env bash
# Forward-only attribution job. In a Baige Distributed Training allocation with one RTX 4090.
# Usage: run.sh <pin> <attempt> [teacher.py options, e.g. --views none --adapters a,b]
# 146 boundaries, 1,464 frozen policy calls, up to 723 replayed controls, 146 Panda renders.
# Estimate 0.25 GPU-hours (model load about 5 min, inference about 5 min); wall limit one hour.
# No gradient, optimizer or checkpoint write: the policy is loaded once and only infer() is called.
set -euo pipefail
if [[ "${1:-}" != --timed ]]; then
    exec timeout --kill-after=5s 3500s bash "$0" --timed "$@"
fi
shift
[[ $# -ge 2 && $1 =~ ^[0-9a-f]{40}$ && $2 =~ ^[A-Za-z0-9_-]+$ ]]
pin=$1
attempt=$2
shift 2
here=$(cd "$(dirname "$0")" && pwd -P)
code=$(git -C "$here" rev-parse --show-toplevel)
[[ $(hostname) == job-*-master-0 && $(git -C "$code" rev-parse HEAD) == "$pin" ]]
git -C "$code" diff --quiet HEAD --
source "$here/../frozen-vla-native-transfer-20261123-r1/env.sh"
[[ $(git -C "$CROSSARM_RELEASE" rev-parse HEAD) == e5500c690c1498eca1b7050350654964190cdf70 ]]
[[ $(git -C /mowenjie/fluffy-bassoon/third_party/openpi-215abfb rev-parse HEAD) == 215abfb217dbac7d5f1273282331b9b1866c0479 ]]
export FLUFFY_GPU_EXECUTION=baige JAX_PLATFORMS=cuda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-0} NCCL_P2P_DISABLE=1
export __EGL_VENDOR_LIBRARY_FILENAMES="$libs/nvidia-vendor.json"
unset LIBGL_ALWAYS_SOFTWARE
export PYTHONPATH="$PYTHONPATH:$here/../frozen-vla-source-view-20261123-r1:$here/../frozen-vla-reachability-20261123-r1"
out="$root/outputs/$(basename "$here")/$attempt"
mkdir -p "$(dirname "$out")"
mkdir "$out"
exec > >(tee "$out/console.log") 2>&1
start=$(date +%s)
trap 'rc=$?; echo "$rc $start $(date +%s)" > "$out/exit.txt"' EXIT
echo "host=$(hostname) pin=$pin args=$*" > "$out/launch.txt"
df -h "$out" > "$out/storage.txt"
# Verify the donor archive and print the query count before the GPU is touched.
"$python" -Bu "$here/teacher.py" list "$out" "$@" > "$out/plan.json"
"$root/setups/robocasa365-native-20261101-r1/env/bin/python" -Bu "$code/scripts/[train]gpu_probe.py" \
    --expected-count 1 --baige-container --purpose evaluation --output "$out/gpu.json"
device=$("$python" -B - "$out/gpu.json" "$here/../reground-rotation-reference-20261123-r1/reference.py" <<'PY'
import json, runpy, sys
probe = json.load(open(sys.argv[1]))
d, = probe["devices"]
name = d["cuda_name"].upper()
if probe["status"] != "passed" or name != "NVIDIA GEFORCE RTX 4090":
    raise RuntimeError(f"Unsupported allocation: {name}")
print(runpy.run_path(sys.argv[2])["egl_device"](0))
PY
)
export MUJOCO_EGL_DEVICE_ID="$device"
"$python" -Bu "$here/teacher.py" run "$out" "$@"
"$python" -Bu "$here/teacher.py" score "$out"
