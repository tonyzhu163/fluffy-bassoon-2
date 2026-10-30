#!/usr/bin/env bash
# Run only inside an already approved one-GPU allocation; reserves no budget.
set -Eeuo pipefail
TASK_SOURCE=${1:?exact reviewed commit}
TASK_RUN=${2:?fresh immutable target baseline identity}
TASK_SECONDS=7200
shift 2
if [ "$#" -gt 0 ] && [ "$1" != --resume ]; then TASK_SECONDS=$1; shift; fi
TASK_RESUME=()
if [ "$#" -gt 0 ]; then test "$1" = --resume; TASK_RESUME=(--resume); shift; fi
test "$#" -eq 0
test "$TASK_SECONDS" -ge 600
test "$TASK_SECONDS" -le 7200
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
TASK_STARTED=$(date +%s)
cd "$TASK_WT"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
if [ "${#TASK_RESUME[@]}" -eq 0 ]; then test ! -e "$TASK_OUT"; mkdir -p "$TASK_OUT";
else test -f "$TASK_OUT/experiment/contract.json"; fi
TASK_ATTEMPT=$TASK_OUT/attempts/$(date +%s%N)
mkdir -p "$TASK_ATTEMPT"
exec 9> "$TASK_OUT/launcher.lock"
flock -n 9
exec > >(tee "$TASK_ATTEMPT/console.log") 2>&1
trap 'TASK_RC=$?; printf "%s\n" "$TASK_RC" > "$TASK_ATTEMPT/exit-code.txt"; printf "%s\n" "$TASK_RC" > "$TASK_OUT/exit-code.txt"' EXIT
export FLUFFY_GPU_EXECUTION=baige CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled
export PYTHONPATH=$TASK_WT/src
"$TASK_PROBE" 'scripts/[train]gpu_probe.py' --expected-count 1 --purpose training \
  --baige-container --baige-exclusive-training --output "$TASK_ATTEMPT/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP=$TASK_ATTEMPT/gpu-probe.json
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$TASK_TEST:$TASK_WT/src \
  "$TASK_PROBE" -m pytest -q -p no:cacheprovider tests/test_vla*.py tests/test_gpu_probe.py \
  > "$TASK_ATTEMPT/focused-tests.log" 2>&1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export OPENPI_DATA_HOME=$TASK_OLD/cache/openpi-215abfb
export LIBERO_CONFIG_PATH=$TASK_ROOT/cache/libero-dsrl-87edbd1-20261029-r1
export MAGICK_HOME=/mowenjie/libs/libero-plus-imagemagick
export MAGICK_CODER_MODULE_PATH=$MAGICK_HOME/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.11/modules-Q16/coders
export LD_LIBRARY_PATH=$MAGICK_HOME/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH=$TASK_DSRL_OVERLAY:$TASK_WT/src:$TASK_OPENPI/src:$TASK_OPENPI/packages/openpi-client/src
TASK_REMAINING=$(( TASK_SECONDS - ( $(date +%s) - TASK_STARTED ) - 60 ))
test "$TASK_REMAINING" -gt 180
timeout --signal=INT --kill-after=30s "$TASK_REMAINING" "$TASK_MODEL" \
  -m fluffy_bassoon.meta_rl.vla_target_online_dsrl_six \
  --output "$TASK_OUT/experiment" --source-commit "$TASK_SOURCE" --expected-hostname "$(hostname)" \
  --checkpoint "$TASK_OLD/inputs/pi05-libero-openpi-20261027-r1" \
  --openpi-root "$TASK_OPENPI" --dsrl-root "$TASK_DSRL" --libero-root "$TASK_LIBERO" \
  --simulator-python "$TASK_SIM" \
  --simulator-pythonpath "$TASK_WT/src:$TASK_LIBERO:$TASK_OPENPI/packages/openpi-client/src:$TASK_OLD/envs/pi05-plus-client-deps" \
  --max-total-actions 64000 --max-elapsed-seconds "$(( TASK_SECONDS - 300 ))" \
  --learner-seed 0 "${TASK_RESUME[@]}"
