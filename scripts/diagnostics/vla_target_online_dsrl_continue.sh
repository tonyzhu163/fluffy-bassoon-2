#!/usr/bin/env bash
# Existing approved allocation only; exactly one development case, no reservation/job submission.
set -Eeuo pipefail
TASK_SOURCE=${1:?exact reviewed continuation commit}
TASK_RUN=${2:?fresh immutable case worker identity}
TASK_PARENT=${3:?terminal original six-case output root}
TASK_NUMBER=${4:?explicit development task0 or40}
TASK_RESUME=${5:-}
case "$TASK_NUMBER" in 0|40) ;; *) exit 2;; esac
case "$TASK_RESUME" in ''|--resume) ;; *) exit 2;; esac
TASK_STARTED=$(date +%s)
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_OLD=/mowenjie/fluffy-bassoon
TASK_WT=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_MODEL=$TASK_OLD/envs/openpi-215abfb/bin/python
TASK_PROBE=$TASK_OLD/envs/lerobot-v044-8fff0fde/bin/python
TASK_OPENPI=$TASK_OLD/third_party/openpi-215abfb
TASK_DSRL=$TASK_ROOT/deps/dsrl-pi0-7f48937-20261029-r1/repo
TASK_DSRL_OVERLAY=$TASK_ROOT/envs/dsrl-pi0-overlay-20261029-r1
TASK_TEST=$TASK_ROOT/envs/pytest-overlay-20261029-r1
TASK_LIBERO=$TASK_ROOT/deps/libero-dsrl-87edbd1-20261029-r1/repo
TASK_SIM=/mowenjie/envs/wam-libero-plus-py310/bin/python
TASK_RESUME_ARGS=()
if [ "$TASK_RESUME" = --resume ]; then
  test -f "$TASK_OUT/worker-time.json"
  TASK_RESUME_ARGS=(--resume)
else
  test ! -e "$TASK_OUT"
  mkdir -p "$TASK_OUT"
fi
exec 9> "$TASK_OUT/launcher.lock"
flock -n 9
cd "$TASK_WT"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH=$TASK_WT/src
TASK_CLOCK=$("$TASK_MODEL" -m fluffy_bassoon.meta_rl.vla_target_online_dsrl_continue clock-begin \
  --parent-root "$TASK_PARENT" --output "$TASK_OUT" --source-commit "$TASK_SOURCE" \
  --task "$TASK_NUMBER" --worker-seconds 6000 --run-id "$TASK_RUN" --started-unix "$TASK_STARTED" \
  "${TASK_RESUME_ARGS[@]}")
IFS=$'\t' read -r TASK_ATTEMPT TASK_STOP_UNIX TASK_SLICE <<< "$TASK_CLOCK"
exec > >(tee "$TASK_ATTEMPT/console.log") 2>&1
task_cleanup() {
  TASK_RC=$?
  trap - EXIT
  PYTHONPATH=$TASK_WT/src "$TASK_MODEL" -m fluffy_bassoon.meta_rl.vla_source40_ppo_worker finish \
    --root "$TASK_OUT" --attempt "$TASK_ATTEMPT" --exit-code "$TASK_RC" || true
  printf '%s\n' "$TASK_RC" > "$TASK_ATTEMPT/exit-code.txt"
  printf '%s\n' "$TASK_RC" > "$TASK_OUT/exit-code.txt"
  exit "$TASK_RC"
}
trap task_cleanup EXIT
TASK_DRIVER_RESUME=()
if [ -f "$TASK_OUT/experiment/contract.json" ]; then
  test "$TASK_RESUME" = --resume
  TASK_DRIVER_RESUME=(--resume)
else
  test ! -e "$TASK_OUT/experiment"
fi
export FLUFFY_GPU_EXECUTION=baige CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled
# Probe Python3.10 must not inherit a Python3.11 model/torch overlay.
PYTHONPATH=$TASK_WT/src "$TASK_PROBE" 'scripts/[train]gpu_probe.py' --expected-count 1 --purpose training \
  --baige-container --baige-exclusive-training --output "$TASK_ATTEMPT/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP=$TASK_ATTEMPT/gpu-probe.json
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$TASK_TEST:$TASK_WT/src \
  "$TASK_PROBE" -m pytest -q -p no:cacheprovider tests/test_vla_target_online_dsrl_continue.py \
  tests/test_vla_target_online_dsrl.py tests/test_vla_target_online_dsrl_six.py \
  > "$TASK_ATTEMPT/focused-tests.log" 2>&1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export OPENPI_DATA_HOME=$TASK_OLD/cache/openpi-215abfb
export LIBERO_CONFIG_PATH=$TASK_ROOT/cache/libero-dsrl-87edbd1-20261029-r1
export MAGICK_HOME=/mowenjie/libs/libero-plus-imagemagick
export MAGICK_CODER_MODULE_PATH=$MAGICK_HOME/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.11/modules-Q16/coders
export LD_LIBRARY_PATH=$MAGICK_HOME/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH=$TASK_DSRL_OVERLAY:$TASK_WT/src:$TASK_OPENPI/src:$TASK_OPENPI/packages/openpi-client/src
TASK_REMAINING=$((TASK_STOP_UNIX - $(date +%s) - 60))
test "$TASK_REMAINING" -gt 420
timeout --signal=INT --kill-after=30s "$TASK_REMAINING" "$TASK_MODEL" \
  -m fluffy_bassoon.meta_rl.vla_target_online_dsrl_continue run \
  --parent-root "$TASK_PARENT" --output "$TASK_OUT/experiment" --source-commit "$TASK_SOURCE" \
  --task "$TASK_NUMBER" --worker-seconds 6000 --stop-unix "$((TASK_STOP_UNIX - 120))" \
  --expected-hostname "$(hostname)" \
  --checkpoint "$TASK_OLD/inputs/pi05-libero-openpi-20261027-r1" --openpi-root "$TASK_OPENPI" \
  --dsrl-root "$TASK_DSRL" --libero-root "$TASK_LIBERO" --simulator-python "$TASK_SIM" \
  --simulator-pythonpath "$TASK_WT/src:$TASK_LIBERO:$TASK_OPENPI/packages/openpi-client/src:$TASK_OLD/envs/pi05-plus-client-deps" \
  "${TASK_DRIVER_RESUME[@]}"
