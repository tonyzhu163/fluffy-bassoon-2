#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_RUN=robotwin-native-history-20261030-r1
TASK_CODE=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_PYTHON=$TASK_ROOT/envs/robotwin-vlrla-20261029-r1/bin/python
TASK_ARTIFACTS=$TASK_ROOT/exports/$TASK_RUN
mkdir "$TASK_OUT"
exec > >(tee "$TASK_OUT/console.log") 2>&1
trap 'TASK_EXIT=$?; printf "%s\n" "$TASK_EXIT" > "$TASK_OUT/exit-code.txt"' EXIT
cd "$TASK_CODE"
export PYTHONPATH="$TASK_CODE/src"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
"$TASK_PYTHON" -m pytest -q tests/test_robotwin_history_corpus.py tests/test_robotwin_history.py tests/test_vla_noise_ppo.py tests/test_vla_noise_sac.py > "$TASK_OUT/tests.log" 2>&1
cat "$TASK_OUT/tests.log"
"$TASK_PYTHON" "$TASK_ARTIFACTS/smoke.py" > "$TASK_OUT/smoke.log" 2>&1
cat "$TASK_OUT/smoke.log"
