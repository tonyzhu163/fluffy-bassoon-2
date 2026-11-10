#!/usr/bin/env bash
set -euo pipefail
root=/mowenjie/fluffy-bassoon-2
code=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
pin=${1:?commit} id=${2:?fresh output id} check=${3:?persistence, rescue or stiff}; shift 3
[[ $id =~ ^[a-z0-9-]+$ && $(hostname) == job-*-master-0 ]]
test "$(git -C "$code" rev-parse HEAD)" = "$pin"
git -C "$code" diff --quiet HEAD
base=$root/outputs/robocasa-sft-reproduction-20261109-r1
py=$root/setups/robocasa365-native-20261101-r1/env/bin/python
test "$(git -C "$base/sources/openpi" rev-parse HEAD)" = ca4c6d710db75e276bc7c866a57bd7e4aee5b6e8
test "$(git -C "$base/sources/robocasa" rev-parse HEAD)" = 29f7ce8814c1547f5af762a0997fbd4b64848dd7
folder=$root/outputs/icrl-idea-checks-20261110-r1
out=$folder/$id
mkdir -p "$folder"
mkdir "$out"
exec > >(tee "$out/run.log") 2>&1
started=$(date +%s)
trap 'rc=$?; echo "{\"exit_code\":$rc,\"started\":$started,\"ended\":$(date +%s)}" > "$out/exit.json"' EXIT
echo "$pin $(hostname)" > "$out/identity.txt"
df -h "$out"
export FLUFFY_GPU_EXECUTION=baige MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export NCCL_P2P_DISABLE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.70
export JAX_COMPILATION_CACHE_DIR="$base/jax-cache" OPENPI_DATA_HOME="$base/openpi-cache"
export PYTHONPATH="$base/sources/openpi/src:$base/sources/openpi/packages/openpi-client/src:$base/sources/robocasa:$code/src"
if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
  uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
  [[ $uuid == GPU-* && $uuid != *$'\n'* && $uuid == "${NVIDIA_VISIBLE_DEVICES:?}" ]]
  export CUDA_VISIBLE_DEVICES=0
fi
"$py" -B "$code/scripts/[train]gpu_probe.py" --expected-count 1 --purpose evaluation --baige-container --output "$out/gpu.json"
timeout --kill-after=15 ${CHECK_SECONDS:-10500} "$py" -Bu "$(dirname "$0")/checks.py" "$out" \
 "$root/inputs/robocasa365-public-pi05-c484-75000-20261101-r1" "$check" "$@"
