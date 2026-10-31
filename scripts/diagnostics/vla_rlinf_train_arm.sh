#!/usr/bin/env bash
# Four matched supervised students on one allocated GPU, <=30 minutes.
set -Eeuo pipefail
TASK_SOURCE=${1:?exact reviewed source commit}
TASK_RUN=${2:?fresh training output identity}
TASK_ARM=${3:?clean or augmented}
TASK_CLEAN=${4:?completed original-LIBERO corpus index}
TASK_PLUS=${5:--}
TASK_PHASE=${6:-supervised}
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_OLD=/mowenjie/fluffy-bassoon
TASK_WORKTREE=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUTPUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_MODEL=$TASK_OLD/envs/maniskill-pi05-rlinf-e1a82ebf-20261028-r1/bin/python
TASK_PROBE=$TASK_OLD/envs/lerobot-v044-8fff0fde/bin/python
TASK_DSRL=$TASK_ROOT/deps/dsrl-pi0-7f48937-20261029-r1/repo
TASK_OVERLAY=$TASK_ROOT/envs/dsrl-pi0-overlay-20261029-r1
cd "$TASK_WORKTREE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
test -f "$TASK_CLEAN"
test ! -e "$TASK_OUTPUT"
TASK_ARGS=(--phase "$TASK_PHASE")
if [ "$TASK_PHASE" = source-sac ] || [ "$TASK_PHASE" = all ]; then
  TASK_ARGS+=(--dsrl-root "$TASK_DSRL")
elif [ "$TASK_PHASE" != supervised ]; then
  printf 'Unknown training phase: %s\n' "$TASK_PHASE" >&2
  exit 2
fi
if [ "$TASK_ARM" = augmented ]; then
  test "$TASK_PLUS" != -
  test -f "$TASK_PLUS"
  TASK_ARGS+=(--plus-index "$TASK_PLUS")
elif [ "$TASK_ARM" != clean ] || [ "$TASK_PLUS" != - ]; then
  printf 'Specify clean alone, or augmented with its Plus source index\n' >&2
  exit 2
fi
mkdir -p "$TASK_OUTPUT"
exec > >(tee "$TASK_OUTPUT/console.log") 2>&1
trap 'TASK_RC=$?; printf "%s\n" "$TASK_RC" > "$TASK_OUTPUT/exit-code.txt"' EXIT
export FLUFFY_GPU_EXECUTION=baige CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 WANDB_MODE=disabled
export PYTHONPATH=$TASK_WORKTREE/src
"$TASK_PROBE" 'scripts/[train]gpu_probe.py' --expected-count 1 --purpose training \
  --baige-container --baige-exclusive-training --output "$TASK_OUTPUT/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP=$TASK_OUTPUT/gpu-probe.json
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  PYTHONPATH="$TASK_ROOT/envs/pytest-overlay-20261029-r1:$TASK_WORKTREE/src" \
  "$TASK_PROBE" -m pytest -q -p no:cacheprovider tests/test_vla*.py tests/test_gpu_probe.py \
  > "$TASK_OUTPUT/focused-tests.log" 2>&1
export PYTHONPATH=$TASK_WORKTREE/src:$TASK_DSRL:$TASK_OVERLAY
"$TASK_MODEL" -m fluffy_bassoon.meta_rl.vla_rlinf_train_arm \
  --output "$TASK_OUTPUT/experiment" --source-commit "$TASK_SOURCE" --arm "$TASK_ARM" \
  --clean-index "$TASK_CLEAN" "${TASK_ARGS[@]}"
