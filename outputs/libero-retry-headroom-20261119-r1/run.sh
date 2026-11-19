#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
code=$(git -C "$here" rev-parse --show-toplevel)
root=/mowenjie/fluffy-bassoon-2
legacy=/mowenjie/fluffy-bassoon
pro="$root/sources/libero-pro2-e5500c69"
setup="$root/setups/libero-meta-learning-20261117-r1"
pin=$1
attempt=$2
[[ "$attempt" =~ ^[a-z0-9]+$ ]]
[[ $(git -C "$code" rev-parse HEAD) == "$pin" && -z $(git -C "$code" status --porcelain) ]]
[[ $(git -C "$pro" rev-parse HEAD) == e5500c690c1498eca1b7050350654964190cdf70 ]]
[[ $(git -C "$legacy/third_party/openpi-215abfb" rev-parse HEAD) == 215abfb217dbac7d5f1273282331b9b1866c0479 ]]
[[ $CGPU_COUNT == 1 && $NVIDIA_VISIBLE_DEVICES == "$NVIDIA_VISIBLE_GPUS_UUID" && $NVIDIA_VISIBLE_DEVICES != *,* ]]
export CUDA_VISIBLE_DEVICES="$NVIDIA_VISIBLE_DEVICES"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda OPENPI_DATA_HOME="$legacy/cache/openpi-215abfb"
export PYTHONPATH="$pro:$setup/simulator-source:$code/src" LIBERO_PRO_ROOT="$setup/simulator-source"
export LIBERO_CONFIG_PATH="$setup/libero-config" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
out="$root/outputs/$(basename "$here")/$attempt"
mkdir -p "$(dirname "$out")"
mkdir "$out"
exec 9>"$root/outputs/libero-bowl-decisions-20261118-r1/device.lock"
flock -w 600 9
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]
df -h "$out" > "$out/storage.txt"
printf probe > "$out/write-probe"
rm "$out/write-probe"
start=$(date +%s)
trap 'rc=$?; echo "$rc $(date +%s) $start" > "$out/exit.txt"' EXIT
echo "$pin $(hostname) $$" > "$out/launch.txt"
"$root/setups/robocasa365-native-20261101-r1/env/bin/python" -Bu "$code/scripts/[train]gpu_probe.py" \
    --expected-count 1 --purpose training --direct-hostname "$(hostname)" --output "$out/gpu.json" > "$out/probe.log" 2>&1
timeout --kill-after=20s 85m "$legacy/envs/openpi-215abfb/bin/python" -Bu "$here/pilot.py" "$out/episodes" > "$out/run.log" 2>&1
