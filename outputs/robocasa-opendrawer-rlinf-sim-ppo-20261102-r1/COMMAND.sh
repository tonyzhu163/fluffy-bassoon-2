#!/usr/bin/env bash
set -Eeuo pipefail
: "${TASK_META:?}" "${TASK_CODE_PIN:?}" "${TASK_ATTEMPT:?}" "${TASK_OUTPUT:?}"
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_PY=$TASK_ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
export TASK_STARTED=$(date +%s)
export TASK_DEADLINE=$((TASK_STARTED+28080))
export PYTHONPATH=$TASK_META/src:$TASK_ROOT/sources/robocasa365-public-pi05-nnx-leaves-20261101-r1/RLinf
export RLINF_PATH=$TASK_ROOT/sources/robocasa365-public-pi05-nnx-leaves-20261101-r1/RLinf
export ROBOCASA_ASSETS_PATH=$TASK_ROOT/setups/robocasa365-native-20261101-r1/assets
export OPENPI_DATA_HOME=$TASK_ROOT/setups/robocasa365-native-20261101-r1/cache/.cache/openpi
export ROBOCASA_PI05_TORCH_MODEL=$TASK_ROOT/inputs/robocasa365-public-pi05-c484-torch-20261101-r2
export ROBOCASA_PI05_NATIVE_NORM=$TASK_ROOT/inputs/robocasa365-public-pi05-c484-75000-20261101-r1/assets/norm_stats.json
export ROBOCASA_SIM_PPO_OUTPUT=$TASK_OUTPUT
export ROBOCASA_SIM_PPO_ACCOUNTING=$TASK_ROOT/outputs/robocasa-opendrawer-rlinf-sim-ppo-20261102-r1/accounting
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl NCCL_P2P_DISABLE=1 FLUFFY_GPU_EXECUTION=baige
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 WANDB_MODE=disabled
mkdir -p "$TASK_OUTPUT"
exec > >(tee "$TASK_OUTPUT/console.log") 2>&1
finish() {
 TASK_RC=$?
 trap - EXIT
 TASK_FINISHED=$(date +%s)
 printf '{"started_unix":%s,"finished_unix":%s,"allocated_gpus":2,"observed_gpu_seconds":%s,"exit_code":%s}\n' \
  "$TASK_STARTED" "$TASK_FINISHED" "$((2*(TASK_FINISHED-TASK_STARTED)))" "$TASK_RC" > "$TASK_OUTPUT/clock.json"
 exit "$TASK_RC"
}
trap finish EXIT
test "$(git -C "$TASK_META" rev-parse HEAD)" = "$TASK_CODE_PIN"
test -z "$(git -C "$TASK_META" status --porcelain)"
test "$(git -C "$RLINF_PATH" rev-parse HEAD)" = 8a1233c3fae6f5905adfda7684c455842b0ecfb8
"$TASK_PY" -B "$TASK_META/scripts/[train]gpu_probe.py" --expected-count 2 --purpose training \
 --baige-container --baige-exclusive-training --output "$TASK_OUTPUT/gpu-probe.json"
timeout --signal=TERM --kill-after=60s 28200 "$TASK_PY" -B "$TASK_META/scripts/[train]robocasa_flow_ppo.py"
