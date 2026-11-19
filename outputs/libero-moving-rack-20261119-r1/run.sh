#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
code=$(git -C "$here" rev-parse --show-toplevel)
root=/mowenjie/fluffy-bassoon-2
legacy=/mowenjie/fluffy-bassoon
pin=$1
attempt=$2
[[ "$attempt" =~ ^[a-z0-9]+$ ]]
[[ $(git -C "$code" rev-parse HEAD) == "$pin" && -z $(git -C "$code" status --porcelain) ]]
[[ $CGPU_COUNT == 1 && $NVIDIA_VISIBLE_DEVICES == "$NVIDIA_VISIBLE_GPUS_UUID" && $NVIDIA_VISIBLE_DEVICES != *,* ]]
export CUDA_VISIBLE_DEVICES="$NVIDIA_VISIBLE_DEVICES"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda OPENPI_DATA_HOME="$legacy/cache/openpi-215abfb"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
out="$root/outputs/$(basename "$here")/$attempt"
mkdir -p "$(dirname "$out")"
mkdir "$out"
exec 9>"$root/outputs/libero-bowl-decisions-20261118-r1/device.lock"
flock -w 7200 9
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]
# Let the last job-step utilization sample clear after the device lock transfers.
for sample in {1..15}; do
    if [[ $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits) -eq 0 ]]; then break; fi
    sleep 2
done
df -h "$out" > "$out/storage.txt"
start=$(date +%s)
trap 'rc=$?; echo "$rc $(date +%s) $start" > "$out/exit.txt"' EXIT
echo "$pin $(hostname) $$" > "$out/launch.txt"
"$root/setups/robocasa365-native-20261101-r1/env/bin/python" -Bu "$code/scripts/[train]gpu_probe.py" --expected-count 1 --purpose training --direct-hostname "$(hostname)" --output "$out/gpu.json" > "$out/probe.log" 2>&1
timeout --kill-after=20s 45m "$legacy/envs/openpi-215abfb/bin/python" -Bu "$here/fresh_openings.py" "$out/episodes" > "$out/run.log" 2>&1
date +%s > "$out/openings-complete.txt"
timeout --kill-after=20s 45m "$legacy/envs/openpi-215abfb/bin/python" -Bu "$here/live_motor.py" "$out/live" "$out/episodes" > "$out/live.log" 2>&1
