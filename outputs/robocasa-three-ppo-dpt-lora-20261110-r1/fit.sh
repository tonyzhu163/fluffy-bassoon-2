#!/usr/bin/env bash
set -Eeuo pipefail
root=/mowenjie/fluffy-bassoon-2
run=outputs/robocasa-three-ppo-dpt-lora-20261110-r1
code=$1
export PILOT_COMMIT=$2
test "$(git -C "$code" rev-parse HEAD)" = "$PILOT_COMMIT"
test -z "$(git -C "$code" status --porcelain --untracked-files=no)"
export RLINF_PATH=$root/sources/critic-selected-a1229239-r1
test "$(git -C "$RLINF_PATH" rev-parse HEAD)" = 8f9bfa7661928ef8c17494b04f4d7bf3125634e1
export PYTHONPATH=$code/src:$RLINF_PATH
export ROBOCASA_PI05_TORCH_MODEL=$root/inputs/robocasa365-public-pi05-c484-torch-20261101-r2
export ROBOCASA_PI05_NATIVE_NORM=$root/inputs/robocasa365-public-pi05-c484-75000-20261101-r1/assets/norm_stats.json
export OPENPI_DATA_HOME=$root/setups/robocasa365-native-20261101-r1/cache/.cache/openpi
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${NVIDIA_VISIBLE_DEVICES:?}}
export FLUFFY_GPU_EXECUTION=baige MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NCCL_P2P_DISABLE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export ROBOCASA_NATIVE_OUTPUT=$root/$run/$3
mkdir "$ROBOCASA_NATIVE_OUTPUT"
exec > >(tee "$ROBOCASA_NATIVE_OUTPUT/fit.log") 2>&1
py=$root/setups/robocasa365-native-20261101-r1/env/bin/python
"$py" -B "$code/scripts/[train]gpu_probe.py" --expected-count 1 --output "$ROBOCASA_NATIVE_OUTPUT/gpu.json" --purpose training --baige-container --baige-exclusive-training
exec "$py" -Bu "$code/$run/fit.py" "$code" "$root/$run/data-01" "$ROBOCASA_NATIVE_OUTPUT" "${@:4}"
