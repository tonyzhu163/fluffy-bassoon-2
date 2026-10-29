#!/usr/bin/env bash
# Separate batch-three diagnostic; one minute, no new simulator/VLA calls.
set -euo pipefail
TASK_COMMIT=${1:?40-hex source commit required}
TASK_RUN=${2:?fresh diagnostic output identity required}
[[ "$TASK_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$TASK_RUN" =~ ^robotwin-saved-ppo-[a-zA-Z0-9-]+$ ]]
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_WORKTREE=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUTPUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_PYTHON=$TASK_ROOT/envs/robotwin-vlrla-20261029-r1/bin/python
test ! -e "$TASK_OUTPUT"
mkdir -p "$TASK_OUTPUT"
exec > >(tee "$TASK_OUTPUT/console.log") 2>&1
trap 'TASK_EXIT=$?; printf "%s\n" "$TASK_EXIT" > "$TASK_OUTPUT/exit-code.txt"' EXIT
export PYTHONPATH="$TASK_WORKTREE/src" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export FLUFFY_GPU_EXECUTION=baige CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$TASK_WORKTREE"
test "$(git rev-parse HEAD)" = "$TASK_COMMIT"
test -z "$(git status --porcelain)"
timeout --signal=TERM --kill-after=5s 50 "$TASK_PYTHON" \
    scripts/diagnostics/robotwin_saved_ppo_update.py --source-commit "$TASK_COMMIT" \
    --input "$TASK_ROOT/outputs/robotwin-noise-matched-20261029-r4/experiment/ppo-recovery.pt" \
    --input-sha256 722618543005ffcac5591593f67f2f8eaa3bd9fcc1be5cebf66995e9487306a9 \
    --output "$TASK_OUTPUT/diagnostic" --diagnostic-batch-trials 3 --wall-seconds 45
