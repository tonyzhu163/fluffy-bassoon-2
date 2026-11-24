#!/usr/bin/env bash
# In a Baige Distributed Training allocation with one RTX 4090.
#   run.sh <pin> <attempt>                       collect, fit, then development and Panda panels
#   run.sh <pin> <attempt> <fit dir> <panel.py options>   panel only on an existing fit,
#       e.g. run.sh $pin locked01 $fit --starts locked --adapters prefix-shift
set -euo pipefail
[[ $# -ge 2 && $1 =~ ^[0-9a-f]{40}$ && $2 =~ ^[A-Za-z0-9_-]+$ ]]
pin=$1
attempt=$2
shift 2
here=$(cd "$(dirname "$0")" && pwd -P)
shared=$here/../frozen-vla-proposal-shift-20261124-r1
code=$(git -C "$here" rev-parse --show-toplevel)
[[ $(hostname) == job-*-master-0 && $(git -C "$code" rev-parse HEAD) == "$pin" ]]
git -C "$code" diff --quiet HEAD --
source "$here/../frozen-vla-native-transfer-20261123-r1/env.sh"
[[ $(git -C "$CROSSARM_RELEASE" rev-parse HEAD) == e5500c690c1498eca1b7050350654964190cdf70 ]]
[[ $(git -C /mowenjie/fluffy-bassoon/third_party/openpi-215abfb rev-parse HEAD) == 215abfb217dbac7d5f1273282331b9b1866c0479 ]]
export FLUFFY_GPU_EXECUTION=baige JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-0} NCCL_P2P_DISABLE=1
export __EGL_VENDOR_LIBRARY_FILENAMES="$libs/nvidia-vendor.json"
unset LIBGL_ALWAYS_SOFTWARE
out="$root/outputs/$(basename "$here")/$attempt"
mkdir -p "$(dirname "$out")"
mkdir "$out"
exec > >(tee "$out/console.log") 2>&1
start=$(date +%s)
trap 'rc=$?; echo "$rc $start $(date +%s)" > "$out/exit.txt"' EXIT
echo "host=$(hostname) pin=$pin args=$*" > "$out/launch.txt"
df -h "$out" > "$out/storage.txt"
"$root/setups/robocasa365-native-20261101-r1/env/bin/python" -Bu "$code/scripts/[train]gpu_probe.py" \
    --expected-count 1 --baige-container --purpose training --output "$out/gpu.json"
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
left() { local s=$((${PANEL_SECONDS:-14400} - $(date +%s) + start)); (( s > 0 )); echo "${s}s"; }
panel() {  # <subdirectory> <panel.py options>
    local dir=$out/$1; shift
    "$python" -Bu "$shared/panel.py" "$dir" --list "$@" > "$dir.plan.json"
    timeout --kill-after=5s "$(left)" "$python" -Bu "$shared/panel.py" "$dir" "$@"
}
arms=prefix-shift,prefix-tokens,prefix-random
if [[ $# -gt 0 ]]; then
    export FROZEN_SHIFT_PREFIX_FIT=$1
    shift
    [[ -f $FROZEN_SHIFT_PREFIX_FIT/fit.json ]]
    panel panel "$@"
else
    timeout --kill-after=5s "$(left)" "$python" -Bu "$here/train.py" "$out" --pin "$pin"
    export FROZEN_SHIFT_PREFIX_FIT=$out
    panel development --starts development --adapters "$arms"
    panel panda --starts panda --adapters "$arms"
fi
