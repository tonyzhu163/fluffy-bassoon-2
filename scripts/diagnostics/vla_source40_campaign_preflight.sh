#!/usr/bin/env bash
# Allocated Linux verification only; this script executes no simulator actions.
set -Eeuo pipefail
TASK_SOURCE=${1:?exact reviewed commit}
TASK_RUN=${2:?fresh preflight identity}
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_OLD=/mowenjie/fluffy-bassoon
TASK_WT=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_PY=$TASK_OLD/envs/lerobot-v044-8fff0fde/bin/python
test ! -e "$TASK_OUT"
mkdir -p "$TASK_OUT"
cd "$TASK_WT"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
exec > >(tee "$TASK_OUT/console.log") 2>&1
trap 'TASK_RC=$?; printf "%s\n" "$TASK_RC" > "$TASK_OUT/exit-code.txt"' EXIT
export FLUFFY_GPU_EXECUTION=baige CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONPATH=$TASK_WT/src
"$TASK_PY" 'scripts/[train]gpu_probe.py' --expected-count 1 --purpose training \
  --baige-container --baige-exclusive-training --output "$TASK_OUT/gpu-probe.json"
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  PYTHONPATH=$TASK_ROOT/envs/pytest-overlay-20261029-r1:$TASK_WT/src \
  "$TASK_PY" -m pytest -q -p no:cacheprovider tests/test_vla*.py tests/test_gpu_probe.py \
  | tee "$TASK_OUT/focused-tests.log"
printf '%s\n' "$TASK_SOURCE" > "$TASK_OUT/source-commit.txt"
printf '0\n' > "$TASK_OUT/simulator-actions.txt"
