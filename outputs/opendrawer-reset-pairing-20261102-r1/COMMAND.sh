#!/usr/bin/env bash
set -Eeuo pipefail
: "${TASK_PROBE_CODE:?}" "${TASK_PROBE_OUTPUT:?}" "${TASK_SOURCE_SHA:?}"
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_STARTED=$(date +%s)
mkdir -p "$TASK_PROBE_OUTPUT"
exec > >(tee "$TASK_PROBE_OUTPUT/console.log") 2>&1
finish() {
 TASK_RC=$?
 TASK_END=$(date +%s)
 printf '%s %s %s %s\n' "$TASK_STARTED" "$TASK_END" "$((TASK_END-TASK_STARTED))" "$TASK_RC" > "$TASK_PROBE_OUTPUT/clock.txt"
}
trap finish EXIT
test "$(git -C "$TASK_PROBE_CODE" rev-parse HEAD)" = "$TASK_SOURCE_SHA"
export PYTHONPATH=$TASK_ROOT/worktrees/opendrawer-ad-dpt-20261102-r1/src:$TASK_ROOT/sources/robocasa365-public-pi05-nnx-leaves-20261101-r1/RLinf
export RLINF_PATH=$TASK_ROOT/sources/robocasa365-public-pi05-nnx-leaves-20261101-r1/RLinf
export ROBOCASA_ASSETS_PATH=$TASK_ROOT/setups/robocasa365-native-20261101-r1/assets
export ROBOCASA_NATIVE_OUTPUT=$TASK_PROBE_OUTPUT/native
export ROBOCASA_PI05_TORCH_MODEL=$TASK_ROOT/inputs/robocasa365-public-pi05-c484-torch-20261101-r2
export ROBOCASA_PI05_NATIVE_NORM=$TASK_ROOT/inputs/robocasa365-public-pi05-c484-75000-20261101-r1/assets/norm_stats.json
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 FLUFFY_GPU_EXECUTION=baige
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
timeout --signal=TERM --kill-after=30s 1080 "$TASK_ROOT/setups/robocasa365-native-20261101-r1/env/bin/python" -B \
 "$TASK_PROBE_CODE/${TASK_PROBE_SCRIPT:-outputs/opendrawer-reset-pairing-20261102-r1/PROBE.py}"
