#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
code=$(git -C "$here" rev-parse --show-toplevel)
root=/mowenjie/fluffy-bassoon-2
legacy=/mowenjie/fluffy-bassoon
pin=$1
attempt=$2
[[ "$attempt" =~ ^[a-z0-9]+$ && $(git -C "$code" rev-parse HEAD) == "$pin" && -z $(git -C "$code" status --porcelain) ]]
[[ $CGPU_COUNT == 1 && $NVIDIA_VISIBLE_DEVICES == "$NVIDIA_VISIBLE_GPUS_UUID" && $NVIDIA_VISIBLE_DEVICES != *,* ]]
export CUDA_VISIBLE_DEVICES="$NVIDIA_VISIBLE_DEVICES"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda OPENPI_DATA_HOME="$legacy/cache/openpi-215abfb"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
out="$root/outputs/$(basename "$here")/$attempt"
mkdir "$out"
start=$(date +%s)
trap 'rc=$?; echo "$rc $(date +%s) $start" > "$out/exit.txt"' EXIT
echo "$pin $(hostname) $$" > "$out/launch.txt"
df -h "$out" > "$out/storage.txt"
exec 9>"$root/outputs/libero-bowl-decisions-20261118-r1/device.lock"
flock -w 7200 9
for i in {1..30}; do
    if [[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) && $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits) -le 5 ]]; then break; fi
    sleep 2
done
date +%s > "$out/gpu-start.txt"
"$root/setups/robocasa365-native-20261101-r1/env/bin/python" -Bu "$code/scripts/[train]gpu_probe.py" --expected-count 1 --purpose training --direct-hostname "$(hostname)" --output "$out/gpu.json" > "$out/probe.log" 2>&1
timeout --kill-after=15s 55m "$legacy/envs/openpi-215abfb/bin/python" -Bu "$here/collect.py" "$out/episodes" > "$out/run.log" 2>&1
date +%s > "$out/gpu-end.txt"
flock -u 9
