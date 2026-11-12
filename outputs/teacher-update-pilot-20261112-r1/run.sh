#!/usr/bin/env bash
set -Eeuo pipefail
root=/mowenjie/fluffy-bassoon-2
code=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
pin=$1
out=$2
test "$(git -C "$code" rev-parse HEAD)" = "$pin"
git -C "$code" diff --quiet HEAD
native=$root/setups/robocasa365-native-20261101-r1
openpi=$root/outputs/robocasa-sft-reproduction-20261109-r1/sources/openpi
test "$(git -C "$openpi" rev-parse HEAD)" = ca4c6d710db75e276bc7c866a57bd7e4aee5b6e8
test "$(git -C "$native/sources/RoboCasa" rev-parse HEAD)" = 3443cdf3dd20f9ae8ed41adba442059b506891d0
mkdir "$out"
exec > >(tee "$out/run.log") 2>&1
started=$(date +%s)
trap 'rc=$?; echo "{\"exit_code\":$rc,\"started\":$started,\"ended\":$(date +%s)}" > "$out/exit.json"' EXIT
test "${CGPU_COUNT:?}" = 1 && test "${CGPU0_COMPUTE_LIMIT:?}" = 100
test "$(nvidia-smi --query-gpu=uuid --format=csv,noheader)" = "${NVIDIA_VISIBLE_DEVICES:?}"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
export CUDA_VISIBLE_DEVICES="$NVIDIA_VISIBLE_DEVICES"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=73
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.72
export LD_LIBRARY_PATH="$root/setups/astra-robocasa-demo-20261112-r1/egl-libs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export OPENPI_DATA_HOME="$root/outputs/robocasa-sft-reproduction-20261109-r1/openpi-cache"
export JAX_COMPILATION_CACHE_DIR="$out/jax-cache"
export PYTHONPATH="$openpi/src:$openpi/packages/openpi-client/src:$native/sources/RoboCasa:$code/src:$code/outputs/robocasa-sft-reproduction-20261109-r1"
py=$native/env/bin/python
echo "$pin $(hostname)" > "$out/identity.txt"
touch "$out/write-probe"
df -h "$out" > "$out/storage.txt"
"$py" -B "$code/scripts/[train]gpu_probe.py" --expected-count 1 --purpose measurement \
  --direct-hostname "$(hostname)" --output "$out/gpu.json" > "$out/probe.log"
timeout --signal=TERM --kill-after=30 13500 "$py" -Bu "$(dirname "$0")/collect.py" \
  --checkpoint "$root/inputs/robocasa365-public-pi05-c484-75000-20261101-r1" --output "$out/data"
