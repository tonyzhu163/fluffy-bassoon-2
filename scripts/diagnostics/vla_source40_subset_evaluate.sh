#!/usr/bin/env bash
# Targeted robot comparison inside an allocated GPU; no production collection.
set -Eeuo pipefail
TASK_SOURCE=${1:?exact reviewed evaluation commit}
TASK_RUN=${2:?fresh evaluation identity}
TASK_TRAIN_RUN=${3:?completed immutable subset training identity}
TASK_RESUME=${4:-}
case "$TASK_RESUME" in ''|--resume) ;; *) exit 2;; esac
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_OLD=/mowenjie/fluffy-bassoon
TASK_WT=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_SCALES=$TASK_ROOT/outputs/$TASK_TRAIN_RUN/experiment/scales.json
TASK_MODEL=$TASK_OLD/envs/openpi-215abfb/bin/python
TASK_PROBE=$TASK_OLD/envs/lerobot-v044-8fff0fde/bin/python
TASK_OPENPI=$TASK_OLD/third_party/openpi-215abfb
TASK_DSRL=$TASK_ROOT/deps/dsrl-pi0-7f48937-20261029-r1/repo
TASK_OVERLAY=$TASK_ROOT/envs/dsrl-pi0-overlay-20261029-r1
TASK_CUDA=$TASK_ROOT/outputs/libero-pi05-student-cuda-20261029-r1/cu128-overlay
TASK_LIBERO=$TASK_ROOT/deps/libero-dsrl-87edbd1-20261029-r1/repo
TASK_PLAN=$TASK_ROOT/outputs/libero40-source-1m-20261029-r1
TASK_RESUME_ARGS=()
if [ "$TASK_RESUME" = --resume ]; then
  test -f "$TASK_OUT/experiment/contract.json"
  TASK_RESUME_ARGS=(--resume)
else
  test ! -e "$TASK_OUT"
  mkdir -p "$TASK_OUT"
fi
test -f "$TASK_SCALES"
cd "$TASK_WT"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
exec 9> "$TASK_OUT/launcher.lock"
flock -n 9
TASK_ATTEMPT=$TASK_OUT/attempts/$(date +%s%N)
mkdir -p "$TASK_ATTEMPT"
exec > >(tee "$TASK_ATTEMPT/console.log") 2>&1
task_cleanup() {
  TASK_RC=$?
  trap - EXIT
  printf '%s\n' "$TASK_RC" > "$TASK_ATTEMPT/exit-code.txt"
  printf '%s\n' "$TASK_RC" > "$TASK_OUT/exit-code.txt"
  exit "$TASK_RC"
}
trap task_cleanup EXIT
export FLUFFY_GPU_EXECUTION=baige CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONPATH=$TASK_WT/src
"$TASK_PROBE" 'scripts/[train]gpu_probe.py' --expected-count 1 --purpose training \
  --baige-container --baige-exclusive-training --output "$TASK_ATTEMPT/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP=$TASK_ATTEMPT/gpu-probe.json
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  PYTHONPATH=$TASK_ROOT/envs/pytest-overlay-20261029-r1:$TASK_WT/src \
  "$TASK_PROBE" -m pytest -q -p no:cacheprovider tests/test_vla*.py tests/test_gpu_probe.py \
  > "$TASK_ATTEMPT/focused-tests.log" 2>&1
"$TASK_MODEL" -m fluffy_bassoon.meta_rl.vla_source40_campaign reserve-diagnostic \
  --campaign "$TASK_PLAN/campaign.json" --ledger "$TASK_PLAN/general-ledger.json" \
  --run "$TASK_RUN" --commit "$TASK_SOURCE" --action-cap 21046 --prior-action-charge 10 \
  "${TASK_RESUME_ARGS[@]}" > "$TASK_ATTEMPT/reservation.json"
"$TASK_MODEL" -m fluffy_bassoon.meta_rl.vla_source40_campaign reserve-ppo \
  --campaign "$TASK_PLAN/campaign.json" --ledger "$TASK_PLAN/ppo-ledger.json" \
  --run "$TASK_RUN" --commit "$TASK_SOURCE" --action-cap 12000 --purpose target_evaluation \
  "${TASK_RESUME_ARGS[@]}" > "$TASK_ATTEMPT/ppo-pool-evaluation-escrow.json"
TASK_SCALES_SHA=$(sha256sum "$TASK_SCALES" | cut -d' ' -f1)
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export LIBERO_CONFIG_PATH=$TASK_ROOT/cache/libero-dsrl-87edbd1-20261029-r1
export OPENPI_DATA_HOME=$TASK_OLD/cache/openpi-215abfb
export MAGICK_HOME=/mowenjie/libs/libero-plus-imagemagick
export MAGICK_CODER_MODULE_PATH=$MAGICK_HOME/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.11/modules-Q16/coders
export LD_LIBRARY_PATH=$MAGICK_HOME/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH=$TASK_CUDA:$TASK_OVERLAY:$TASK_WT/src:$TASK_DSRL:$TASK_OPENPI/src:$TASK_OPENPI/packages/openpi-client/src
timeout --signal=INT --kill-after=30s 2700 "$TASK_MODEL" \
  -m fluffy_bassoon.meta_rl.vla_source40_subset_evaluate \
  --scales "$TASK_SCALES" --scales-sha256 "$TASK_SCALES_SHA" --output "$TASK_OUT/experiment" \
  --checkpoint "$TASK_OLD/inputs/pi05-libero-openpi-20261027-r1" --openpi-root "$TASK_OPENPI" \
  --dsrl-root "$TASK_DSRL" --libero-root "$TASK_LIBERO" \
  --simulator-python /mowenjie/envs/wam-libero-plus-py310/bin/python \
  --simulator-pythonpath "$TASK_WT/src:$TASK_LIBERO:$TASK_OPENPI/packages/openpi-client/src:$TASK_OLD/envs/pi05-plus-client-deps" \
  --source-commit "$TASK_SOURCE" --expected-hostname "$(hostname)" \
  --max-total-actions 33046 --max-elapsed-seconds 2400 "${TASK_RESUME_ARGS[@]}"
