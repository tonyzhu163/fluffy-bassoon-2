#!/usr/bin/env bash
# Existing allocated GPU; source subset qualification, no new robot interaction.
set -Eeuo pipefail
TASK_SOURCE=${1:?exact reviewed integrated commit}
TASK_RUN=${2:?fresh output identity}
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_OLD=/mowenjie/fluffy-bassoon
TASK_WORKTREE=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUTPUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_MODEL=$TASK_OLD/envs/openpi-215abfb/bin/python
TASK_PROBE=$TASK_OLD/envs/lerobot-v044-8fff0fde/bin/python
TASK_CUDA=$TASK_ROOT/outputs/libero-pi05-student-cuda-20261029-r1/cu128-overlay
TASK_TEST=$TASK_ROOT/envs/pytest-overlay-20261029-r1
test ! -e "$TASK_OUTPUT"
mkdir -p "$TASK_OUTPUT"
cd "$TASK_WORKTREE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
exec > >(tee "$TASK_OUTPUT/console.log") 2>&1
task_cleanup() { TASK_RC=$?; trap - EXIT; printf '%s\n' "$TASK_RC" > "$TASK_OUTPUT/exit-code.txt"; exit "$TASK_RC"; }
trap task_cleanup EXIT
export FLUFFY_GPU_EXECUTION=baige CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONPATH=$TASK_WORKTREE/src
"$TASK_PROBE" 'scripts/[train]gpu_probe.py' --expected-count 1 --purpose training \
  --baige-container --baige-exclusive-training --output "$TASK_OUTPUT/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP=$TASK_OUTPUT/gpu-probe.json
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$TASK_TEST:$TASK_WORKTREE/src \
  "$TASK_PROBE" -m pytest -q -p no:cacheprovider tests/test_vla_source40_subset_training.py \
  tests/test_vla_source40_training.py tests/test_vla_history.py tests/test_gpu_probe.py \
  > "$TASK_OUTPUT/focused-tests.log" 2>&1
export PYTHONPATH=$TASK_CUDA:$TASK_WORKTREE/src
"$TASK_MODEL" -m fluffy_bassoon.meta_rl.vla_source40_subset_training \
  --qualification-subset \
  --train-index "$TASK_ROOT/outputs/libero40-1m-sac-20261029-s00-train-r1/experiment/index.json" \
  --validation-index "$TASK_ROOT/outputs/libero40-pi05-source-smoke-20261029-s00-validation-r1/experiment/index.json" \
  --output "$TASK_OUTPUT/experiment" --scales 25 100 500 --seed 0 \
  --source-commit "$TASK_SOURCE" --expected-hostname "$(hostname)" \
  --max-controller-seconds 300 --max-total-seconds 1500
