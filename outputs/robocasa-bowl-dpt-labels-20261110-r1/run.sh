#!/usr/bin/env bash
set -Eeuo pipefail
root=/mowenjie/fluffy-bassoon-2
code=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
base=$root/outputs/robocasa-sft-reproduction-20261109-r1
folder=$root/outputs/robocasa-bowl-dpt-labels-20261110-r1
pin=$1 id=$2; shift 2
[[ $id =~ ^[a-z0-9-]+$ ]] && test "$(git -C "$code" rev-parse HEAD)" = "$pin"
git -C "$code" diff --quiet HEAD
test "$(git -C "$base/sources/openpi" rev-parse HEAD)" = ca4c6d710db75e276bc7c866a57bd7e4aee5b6e8
test "$(git -C "$base/sources/robocasa" rev-parse HEAD)" = 29f7ce8814c1547f5af762a0997fbd4b64848dd7
out=$folder/$id
mkdir "$out"
exec > >(tee "$out/run.log") 2>&1
started=$(date +%s)
trap 'rc=$?; echo "{\"exit_code\":$rc,\"started\":$started,\"ended\":$(date +%s)}" > "$out/exit.json"' EXIT
echo "$pin $(hostname) $*" > "$out/identity.txt"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 PYTHONHASHSEED=73 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.72
export JAX_COMPILATION_CACHE_DIR="$base/jax-cache" OPENPI_DATA_HOME="$base/openpi-cache"
export PYTHONPATH="$base/sources/openpi/src:$base/sources/openpi/packages/openpi-client/src:$base/sources/robocasa:$code/src:$code/outputs/robocasa-sft-reproduction-20261109-r1:$code/outputs/robocasa-meta-feedback-20261109-r1:$code/outputs/robocasa-compact-bowls-20261109-r1"
py=$root/setups/robocasa365-native-20261101-r1/env/bin/python
if [[ $(hostname) == aibox-* ]]; then
    test "${CGPU_COUNT:?}" = 1 && test "${CGPU0_COMPUTE_LIMIT:?}" = 100
    test "$(nvidia-smi --query-gpu=uuid --format=csv,noheader)" = "${NVIDIA_VISIBLE_DEVICES:?}"
    export CUDA_VISIBLE_DEVICES="$NVIDIA_VISIBLE_DEVICES"
    export LD_LIBRARY_PATH="$folder/dev-graphics-01/root/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
    probe=(--purpose measurement --direct-hostname "$(hostname)")
else
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    probe=(--purpose evaluation --baige-container)
fi
"$py" -B "$code/scripts/[train]gpu_probe.py" --expected-count 1 --output "$out/gpu.json" "${probe[@]}" > "$out/probe.log"
for mode in pickup left right; do
    "$py" -Bu "$(dirname "$0")/collect.py" "$mode" "$out" "$root/inputs/robocasa365-public-pi05-c484-75000-20261101-r1" "$@"
done
