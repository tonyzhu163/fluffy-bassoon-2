#!/usr/bin/env bash
# One shared all40 PPO worker in an existing authorized allocation. No submission.
set -Eeuo pipefail
TASK_SOURCE=${1:?exact reviewed commit}
TASK_RUN=${2:?immutable run identity}
TASK_ACTIONS=${3:?joint physical action reservation}
TASK_UPDATES=${4:?requested PPO updates}
TASK_MODE=${5:?full or episode}
TASK_SLICE=${6:?allocated slice seconds, including setup and shutdown}
shift 6
TASK_RESUME=
TASK_WIDTH=128
TASK_LAYERS=2
TASK_HEADS=4
TASK_CONTEXT=300
while [ "$#" -gt 0 ]; do
  case "$1" in
    --resume) TASK_RESUME=--resume; shift ;;
    --width) TASK_WIDTH=${2:?width}; shift 2 ;;
    --layers) TASK_LAYERS=${2:?layers}; shift 2 ;;
    --heads) TASK_HEADS=${2:?heads}; shift 2 ;;
    --context-length) TASK_CONTEXT=${2:?context}; shift 2 ;;
    *) exit 2 ;;
  esac
done
case "$TASK_MODE" in full|episode) ;; *) exit 2;; esac
case "$TASK_RESUME" in ''|--resume) ;; *) exit 2;; esac
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_OLD=/mowenjie/fluffy-bassoon
TASK_WORKTREE=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUTPUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_MODEL=$TASK_OLD/envs/openpi-215abfb/bin/python
TASK_PROBE=$TASK_OLD/envs/lerobot-v044-8fff0fde/bin/python
TASK_DSRL_OVERLAY=$TASK_ROOT/envs/dsrl-pi0-overlay-20261029-r1
TASK_OPENPI=$TASK_OLD/third_party/openpi-215abfb
TASK_CUDA=$TASK_ROOT/outputs/libero-pi05-student-cuda-20261029-r1/cu128-overlay
TASK_TEST=$TASK_ROOT/envs/pytest-overlay-20261029-r1
TASK_SIM=/mowenjie/envs/wam-libero-plus-py310/bin/python
TASK_LIBERO=$TASK_ROOT/deps/libero-dsrl-87edbd1-20261029-r1/repo
TASK_CHECKPOINT=$TASK_OLD/inputs/pi05-libero-openpi-20261027-r1
TASK_PLAN=$TASK_ROOT/outputs/libero40-source-1m-20261029-r1
TASK_STARTED=$(date +%s)
TASK_RESUME_ARGS=()
if [ "$TASK_RESUME" = --resume ]; then
  TASK_RESUME_ARGS=(--resume)
  test -f "$TASK_OUTPUT/experiment/learner/latest.pt"
  test -f "$TASK_OUTPUT/experiment/source-contract.json"
else
  test ! -e "$TASK_OUTPUT"
  mkdir -p "$TASK_OUTPUT"
fi
cd "$TASK_WORKTREE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
exec 9> "$TASK_OUTPUT/launcher.lock"
flock -n 9
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH=$TASK_WORKTREE/src
TASK_ATTEMPT=$("$TASK_MODEL" -m fluffy_bassoon.meta_rl.vla_source40_ppo_worker begin \
  --root "$TASK_OUTPUT" --campaign "$TASK_PLAN/campaign.json" --schedule "$TASK_PLAN/schedule.json" \
  --source "$TASK_SOURCE" --run "$TASK_RUN" --action-cap "$TASK_ACTIONS" --updates "$TASK_UPDATES" \
  --memory-mode "$TASK_MODE" --slice-seconds "$TASK_SLICE" --worker-seconds 21600 \
  --width "$TASK_WIDTH" --layers "$TASK_LAYERS" --heads "$TASK_HEADS" --context-length "$TASK_CONTEXT" \
  "${TASK_RESUME_ARGS[@]}")
exec > >(tee "$TASK_ATTEMPT/console.log") 2>&1
task_cleanup() {
  TASK_RC=$?
  trap - EXIT
  PYTHONPATH=$TASK_WORKTREE/src "$TASK_MODEL" -m fluffy_bassoon.meta_rl.vla_source40_ppo_worker finish \
    --root "$TASK_OUTPUT" --attempt "$TASK_ATTEMPT" --exit-code "$TASK_RC" || true
  printf '%s\n' "$TASK_RC" > "$TASK_ATTEMPT/exit-code.txt"
  printf '%s\n' "$TASK_RC" > "$TASK_OUTPUT/exit-code.txt"
  exit "$TASK_RC"
}
trap task_cleanup EXIT
"$TASK_MODEL" -m fluffy_bassoon.meta_rl.vla_source40_campaign reserve-ppo \
  --campaign "$TASK_PLAN/campaign.json" --ledger "$TASK_PLAN/ppo-ledger.json" \
  --run "$TASK_RUN" --commit "$TASK_SOURCE" --action-cap "$TASK_ACTIONS" "${TASK_RESUME_ARGS[@]}"
export FLUFFY_GPU_EXECUTION=baige CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
"$TASK_PROBE" 'scripts/[train]gpu_probe.py' --expected-count 1 --purpose training \
  --baige-container --baige-exclusive-training --output "$TASK_ATTEMPT/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP=$TASK_ATTEMPT/gpu-probe.json
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$TASK_TEST:$TASK_WORKTREE/src \
  "$TASK_PROBE" -m pytest -q -p no:cacheprovider tests/test_vla*.py tests/test_gpu_probe.py \
  > "$TASK_ATTEMPT/focused-tests.log" 2>&1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export LIBERO_CONFIG_PATH=$TASK_ROOT/cache/libero-dsrl-87edbd1-20261029-r1
export OPENPI_DATA_HOME=$TASK_OLD/cache/openpi-215abfb
export MAGICK_HOME=/mowenjie/libs/libero-plus-imagemagick
export MAGICK_CODER_MODULE_PATH=$MAGICK_HOME/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.11/modules-Q16/coders
export LD_LIBRARY_PATH=$MAGICK_HOME/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH=$TASK_CUDA:$TASK_DSRL_OVERLAY:$TASK_WORKTREE/src:$TASK_OPENPI/src:$TASK_OPENPI/packages/openpi-client/src
# Slice duration is per attempt. The scientific driver ceiling and action cap stay fixed.
TASK_DRIVER_SECONDS=21600
TASK_TIMEOUT_SECONDS=$(( TASK_SLICE - ( $(date +%s) - TASK_STARTED ) - 60 ))
test "$TASK_TIMEOUT_SECONDS" -gt 60
timeout --signal=INT --kill-after=30s "$TASK_TIMEOUT_SECONDS" "$TASK_MODEL" \
  -m fluffy_bassoon.meta_rl.vla_source40_multitask_ppo \
  --schedule "$TASK_PLAN/schedule.json" --output "$TASK_OUTPUT/experiment" \
  --checkpoint "$TASK_CHECKPOINT" --openpi-root "$TASK_OPENPI" \
  --simulator-python "$TASK_SIM" \
  --simulator-pythonpath "$TASK_WORKTREE/src:$TASK_LIBERO:$TASK_OPENPI/packages/openpi-client/src:$TASK_OLD/envs/pi05-plus-client-deps" \
  --source-commit "$TASK_SOURCE" --expected-hostname "$(hostname)" \
  --max-total-actions "$TASK_ACTIONS" --max-elapsed-seconds "$TASK_DRIVER_SECONDS" \
  --updates "$TASK_UPDATES" --attempts 4 --memory-mode "$TASK_MODE" \
  --seed 2 --width "$TASK_WIDTH" --layers "$TASK_LAYERS" --heads "$TASK_HEADS" \
  --context-length "$TASK_CONTEXT" --learning-rate 1e-4 --epochs 2 \
  "${TASK_RESUME_ARGS[@]}"
