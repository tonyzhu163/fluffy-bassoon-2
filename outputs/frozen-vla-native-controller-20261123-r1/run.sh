#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 && $1 =~ ^[0-9a-f]{40}$ && $2 =~ ^[a-zA-Z0-9_-]+$ ]]
pin=$1
here=$(cd "$(dirname "$0")" && pwd -P)
code=$(cd "$here/../.." && pwd -P)
[[ $(git -C "$code" rev-parse HEAD) == "$pin" && $(hostname) == job-*-master-0 && $(uname -s) == Linux ]]
[[ -z $(git -C "$code" status --porcelain --untracked-files=no) ]]
source "$code/outputs/frozen-vla-native-transfer-20261123-r1/env.sh"
[[ $(git -C "$CROSSARM_RELEASE" rev-parse HEAD) == e5500c690c1498eca1b7050350654964190cdf70 ]]
[[ $(git -C /mowenjie/fluffy-bassoon/third_party/openpi-215abfb rev-parse HEAD) == 215abfb217dbac7d5f1273282331b9b1866c0479 ]]
out="$root/outputs/$(basename "$here")/$2"
mkdir "$out"
start=$(date +%s)
trap 'echo "$? $start $(date +%s)" > "$out/exit.txt"' EXIT
exec > >(tee "$out/log.txt") 2>&1
echo "$pin $(hostname)" > "$out/launch.txt"
cd "$here"
export FLUFFY_GPU_EXECUTION=baige JAX_PLATFORMS=cuda NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-0}
export __EGL_VENDOR_LIBRARY_FILENAMES="$libs/nvidia-vendor.json"
export NUMBA_CACHE_DIR="$out/cache/numba"
unset LIBGL_ALWAYS_SOFTWARE
echo "ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170  /mowenjie/fluffy-bassoon/inputs/pi05-libero-openpi-20261027-r1/READY.json" | sha256sum -c -
echo "b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84  /mowenjie/fluffy-bassoon/inputs/pi05-libero-openpi-20261027-r1/assets/physical-intelligence/libero/norm_stats.json" | sha256sum -c -
"$root/setups/robocasa365-native-20261101-r1/env/bin/python" -Bu "$code/scripts/[train]gpu_probe.py" \
    --expected-count 1 --baige-container --purpose evaluation --output "$out/gpu.json"
egl=$("$python" -B - "$out/gpu.json" "$code/outputs/reground-rotation-reference-20261123-r1/reference.py" <<'PY'
import json, runpy, sys
p = json.load(open(sys.argv[1]))
d, = p["devices"]
name = d["cuda_name"].upper()
assert p["status"] == "passed" and (name in ("NVIDIA GEFORCE RTX 4090", "NVIDIA RPBZZZ6") or "RTX PRO 6000" in name)
print(runpy.run_path(sys.argv[2])["egl_device"](0))
PY
)
export MUJOCO_EGL_DEVICE_ID="$egl"
remaining=$((5400 - $(date +%s) + start - 6))
((remaining > 0))
timeout --kill-after=5s "${remaining}s" "$python" -Bu "$here/live.py" "$out"
