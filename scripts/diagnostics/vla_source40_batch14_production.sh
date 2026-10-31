#!/usr/bin/env bash
# Preparation only until an exact44h funding record exists. Never schedules jobs.
set -Eeuo pipefail
TASK_STARTED=$(date +%s)
TASK_SOURCE=${1:?exact reviewed production commit}
TASK_RUN=${2:?fresh immutable production identity}
TASK_PLAN=${3:?immutable per-learner production plan}
TASK_PLAN_SHA=${4:?plan file SHA256}
TASK_FUNDING=${5:?exact approved funding record; proposal alone is rejected}
TASK_SLICE=${6:-21540}
TASK_RESUME=${7:-}
case "$TASK_RESUME" in ''|--resume) ;; *) exit 2;; esac
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_OLD=/mowenjie/fluffy-bassoon
TASK_WT=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_MODEL=$TASK_OLD/envs/openpi-215abfb/bin/python
TASK_PROBE=$TASK_OLD/envs/lerobot-v044-8fff0fde/bin/python
TASK_OPENPI=$TASK_OLD/third_party/openpi-215abfb
TASK_CUDA=$TASK_ROOT/outputs/libero-pi05-student-cuda-20261029-r1/cu128-overlay
TASK_LIBERO=$TASK_ROOT/deps/libero-dsrl-87edbd1-20261029-r1/repo
TASK_SCHEDULE=$TASK_ROOT/outputs/libero40-source-1m-20261029-r1/schedule.json
cd "$TASK_WT"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
TASK_RESUME_ARGS=()
if [ "$TASK_RESUME" = --resume ]; then
  test -f "$TASK_OUT/worker-time.json"
  TASK_RESUME_ARGS=(--resume)
else
  test ! -e "$TASK_OUT"
fi
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=$TASK_WT/src
TASK_CLOCK=$("$TASK_MODEL" -m fluffy_bassoon.meta_rl.vla_source40_batch14_production begin \
  --root "$TASK_OUT" --plan "$TASK_PLAN" --plan-sha256 "$TASK_PLAN_SHA" --funding "$TASK_FUNDING" \
  --schedule "$TASK_SCHEDULE" --source "$TASK_SOURCE" --run "$TASK_RUN" \
  --slice-seconds "$TASK_SLICE" --started-unix "$TASK_STARTED" "${TASK_RESUME_ARGS[@]}")
IFS=$'\t' read -r TASK_ATTEMPT TASK_STOP TASK_ACTUAL_SLICE <<< "$TASK_CLOCK"
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
exec 9> "$TASK_OUT/launcher.lock"
flock -n 9
# Plan values are emitted only after exact plan and ancestor receipt validation.
TASK_VALUES=$("$TASK_MODEL" - "$TASK_PLAN" "$TASK_PLAN_SHA" <<'PYPLAN'
import sys
from fluffy_bassoon.meta_rl.vla_source40_batch14_production import load_plan
p=load_plan(*sys.argv[1:])
print('\t'.join(str(p[k]) for k in ('mode','maximum_native_stream_actions','legacy_history_path','legacy_history_sha256')))
PYPLAN
)
IFS=$'\t' read -r TASK_MODE TASK_ACTION_CAP TASK_HISTORY TASK_HISTORY_SHA <<< "$TASK_VALUES"
export FLUFFY_GPU_EXECUTION=baige CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
"$TASK_PROBE" 'scripts/[train]gpu_probe.py' --expected-count 1 --purpose training \
  --baige-container --baige-exclusive-training --output "$TASK_ATTEMPT/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP=$TASK_ATTEMPT/gpu-probe.json
# Existing batch14/warmstart/JAX qualifications remain evidence; only new glue tests here.
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  PYTHONPATH=$TASK_ROOT/envs/pytest-overlay-20261029-r1:$TASK_WT/src \
  "$TASK_PROBE" -m pytest -q -p no:cacheprovider tests/test_vla_source40_batch14_production.py \
  > "$TASK_ATTEMPT/production-tests.log" 2>&1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export LIBERO_CONFIG_PATH=$TASK_ROOT/cache/libero-dsrl-87edbd1-20261029-r1
export OPENPI_DATA_HOME=$TASK_OLD/cache/openpi-215abfb
export MAGICK_HOME=/mowenjie/libs/libero-plus-imagemagick
export MAGICK_CODER_MODULE_PATH=$MAGICK_HOME/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.11/modules-Q16/coders
export LD_LIBRARY_PATH=$MAGICK_HOME/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH=$TASK_CUDA:$TASK_WT/src:$TASK_OPENPI/src:$TASK_OPENPI/packages/openpi-client/src
"$TASK_MODEL" -c 'import sys,torch; assert sys.version_info[:2] == (3,11); assert torch.__version__ == "2.7.1+cu128" and torch.version.cuda == "12.8"'
TASK_DRIVER_RESUME=()
if [ -e "$TASK_OUT/experiment" ]; then TASK_DRIVER_RESUME=(--resume); fi
TASK_TIMEOUT=$((TASK_STOP - $(date +%s) - 60))
test "$TASK_TIMEOUT" -gt 420
# Soft stop between complete K4 trials, with300s for the longest native trial.
timeout --signal=INT --kill-after=30s "$TASK_TIMEOUT" "$TASK_MODEL" \
  -m fluffy_bassoon.meta_rl.vla_source40_batch14 \
  --schedule "$TASK_SCHEDULE" --output "$TASK_OUT/experiment" \
  --checkpoint "$TASK_OLD/inputs/pi05-libero-openpi-20261027-r1" --openpi-root "$TASK_OPENPI" \
  --simulator-python /mowenjie/envs/wam-libero-plus-py310/bin/python \
  --simulator-pythonpath "$TASK_WT/src:$TASK_LIBERO:$TASK_OPENPI/packages/openpi-client/src:$TASK_OLD/envs/pi05-plus-client-deps" \
  --source-commit "$TASK_SOURCE" --expected-hostname "$(hostname)" \
  --max-total-actions "$TASK_ACTION_CAP" --stop-unix "$((TASK_STOP - 360))" --updates 100 \
  --seed 2 --memory-mode "$TASK_MODE" --width 128 --layers 2 --heads 4 --context-length 300 \
  --warm-start-history "$TASK_HISTORY" --warm-start-history-sha256 "$TASK_HISTORY_SHA" \
  --production-plan "$TASK_PLAN" --production-plan-sha256 "$TASK_PLAN_SHA" "${TASK_DRIVER_RESUME[@]}"
