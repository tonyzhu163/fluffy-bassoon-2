#!/usr/bin/env bash
set -Eeuo pipefail
: "${TASK_CODE:?}" "${TASK_PIN:?}" "${TASK_ATTEMPT:?}"
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_RUN=$TASK_ROOT/outputs/opendrawer-ppo-ad-dpt-20261102-r1
TASK_PY=$TASK_ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
TASK_STARTED=$(date +%s)
TASK_DEADLINE=$((TASK_STARTED+27900))
export TASK_ATTEMPT_OUT=$TASK_RUN/attempts/$TASK_ATTEMPT
export PYTHONPATH=$TASK_CODE/src:$TASK_ROOT/sources/robocasa365-public-pi05-nnx-leaves-20261101-r1/RLinf
export RLINF_PATH=$TASK_ROOT/sources/robocasa365-public-pi05-nnx-leaves-20261101-r1/RLinf
export ROBOCASA_ASSETS_PATH=$TASK_ROOT/setups/robocasa365-native-20261101-r1/assets
export OPENPI_DATA_HOME=$TASK_ROOT/setups/robocasa365-native-20261101-r1/cache/.cache/openpi
export ROBOCASA_NATIVE_OUTPUT=$TASK_ATTEMPT_OUT/native
export ROBOCASA_PI05_TORCH_MODEL=$TASK_ROOT/inputs/robocasa365-public-pi05-c484-torch-20261101-r2
export ROBOCASA_PI05_NATIVE_NORM=$TASK_ROOT/inputs/robocasa365-public-pi05-c484-75000-20261101-r1/assets/norm_stats.json
export FLUFFY_GPU_EXECUTION=baige NCCL_P2P_DISABLE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 WANDB_MODE=disabled
mkdir -p "$TASK_RUN/attempts"
exec > >(tee "$TASK_RUN/attempts/$TASK_ATTEMPT-console.log") 2>&1
TASK_RC=0
finish() {
    trap - EXIT
    mkdir -p "$TASK_ATTEMPT_OUT"
    printf '%s\n' "$TASK_RC" > "$TASK_ATTEMPT_OUT/exit-code.txt"
    TASK_FINISHED=$(date +%s)
    printf '{"started_unix":%s,"finished_unix":%s,"allocated_gpus":2,"charged_gpu_seconds":%s,"exit_code":%s}\n' \
        "$TASK_STARTED" "$TASK_FINISHED" "$(((TASK_FINISHED-TASK_STARTED)*2))" "$TASK_RC" > "$TASK_ATTEMPT_OUT/clock.json"
    exit "$TASK_RC"
}
trap 'TASK_RC=$?; finish' EXIT
timeout --signal=TERM --kill-after=60s 28020 "$TASK_PY" -B \
    "$TASK_CODE/outputs/opendrawer-ppo-ad-dpt-20261102-r1/RUN.py" \
    --source-commit "$TASK_PIN" --deadline "$TASK_DEADLINE"
