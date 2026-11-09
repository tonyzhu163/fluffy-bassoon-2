#!/usr/bin/env bash
set -euo pipefail
base=/mowenjie/fluffy-bassoon-2
here=$(cd "$(dirname "$0")/../.." && pwd)
name=robocasa-drawer-calibration-20261109-r1
pin=${1:?commit}
[[ $HOSTNAME == job-*-master-0 ]]
[[ $(git -C "$here" rev-parse HEAD) == "$pin" ]]
git -C "$here" diff --quiet
shared=$base/outputs/robocasa-sft-reproduction-20261109-r1
[[ $(git -C "$shared/sources/openpi" rev-parse HEAD) == ca4c6d710db75e276bc7c866a57bd7e4aee5b6e8 ]]
[[ $(git -C "$shared/sources/robocasa" rev-parse HEAD) == 29f7ce8814c1547f5af762a0997fbd4b64848dd7 ]]
[[ $(df -Pk "$base" | awk 'END{print $4}') -gt 100663296 ]]
out=$base/outputs/$name/$HOSTNAME
mkdir "$out"
exec > >(tee "$out/console.log") 2>&1
started=$(date +%s)
trap 'rc=$?; printf "%s %s %s\n" "$rc" "$started" "$(date +%s)" > "$out/exit.txt"' EXIT
printf '%s\n' "$pin" >"$out/commit"
export FLUFFY_GPU_EXECUTION=baige MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export JAX_COMPILATION_CACHE_DIR="$shared/jax-cache" OPENPI_DATA_HOME="$shared/openpi-cache" XLA_PYTHON_CLIENT_MEM_FRACTION=0.70
export PYTHONPATH="$shared/sources/openpi/src:$shared/sources/openpi/packages/openpi-client/src:$shared/sources/robocasa:$here/src"
if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
    uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
    [[ $uuid == GPU-* && $uuid != *$'\n'* && "$uuid" == "${NVIDIA_VISIBLE_DEVICES:?}" ]]
    export CUDA_VISIBLE_DEVICES=0
fi
py=$base/setups/robocasa365-native-20261101-r1/env/bin/python
"$py" -B "$here/scripts/[train]gpu_probe.py" --expected-count 1 --purpose evaluation --baige-container --output "$out/GPU-PROBE.json"
timeout -s TERM -k 15 5300 "$py" -Bu "$here/outputs/$name/pilot.py" "$out" "$base/inputs/robocasa365-public-pi05-c484-75000-20261101-r1"
