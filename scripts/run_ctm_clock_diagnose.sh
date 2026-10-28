#!/bin/bash
set -euo pipefail
TASK_SOURCE=${1:?exact source}
TASK_RUN=${2:?fresh clock diagnostic identity}
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_PARENT=$TASK_ROOT/outputs/metaworld-ml10-matched-baige-20261028-r1/ctm/train
TASK_DIAG=$TASK_ROOT/outputs/metaworld-ppo-debug-baige-20261029-r3
TASK_CONTROL=$TASK_ROOT/outputs/metaworld-ppo-controls-baige-20261029-r1
TASK_WORKTREE=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUTPUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_PYTHON=$TASK_ROOT/envs/metaworld-baselines-20261028-r1/bin/python
[[ $TASK_SOURCE =~ ^[0-9a-f]{40}$ && $TASK_RUN =~ ^metaworld-clock-debug-baige-20261029-r[0-9]+$ ]]
[[ $(hostname) =~ ^job-[a-z0-9]+-master-0$ ]]
test "$(cat "$(dirname "$TASK_PARENT")/exit-code.txt")" = 0
test "$(cat "$TASK_DIAG/exit-code.txt")" = 0
test "$(cat "$TASK_CONTROL/exit-code.txt")" = 0
TASK_PARENT_SECONDS=$(cat "$(dirname "$TASK_PARENT")/allocation-active-seconds.txt")
TASK_DIAG_SECONDS=$(cat "$TASK_DIAG/allocation-active-seconds.txt")
TASK_CONTROL_SECONDS=$(cat "$TASK_CONTROL/allocation-active-seconds.txt")
# Failed diagnostic r1 consumed131scheduler-pod seconds, retained in its audit.
(( TASK_PARENT_SECONDS + 131 + TASK_DIAG_SECONDS + TASK_CONTROL_SECONDS + 900 <= 21600 ))
cd "$TASK_WORKTREE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
git diff --quiet cd0128f03a060aff2d538a91f35058aa822df845 "$TASK_SOURCE" -- \
 src/fluffy_bassoon/meta_rl/{ppo,metaworld_policy,metaworld_env,metaworld_workers,checkpoint,cores,cmm,metaworld_train}.py src/lerobot_policy_ctm
mkdir "$TASK_OUTPUT"
exec > >(tee "$TASK_OUTPUT/console.log") 2>&1
TASK_STARTED=$(date +%s)
trap 'TASK_EXIT=$?; printf "%s\n" "$TASK_EXIT" > "$TASK_OUTPUT/exit-code.txt"; printf "%s\n" "$(($(date +%s)-TASK_STARTED))" > "$TASK_OUTPUT/allocation-active-seconds.txt"' EXIT
export PYTHONPATH="$TASK_WORKTREE/src" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export FLUFFY_GPU_EXECUTION=baige CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MUJOCO_GL=egl WANDB_MODE=disabled
export FLUFFY_METAWORLD_REAL_QUALIFICATION=1
# Qualified frozen runtime: verify the allocated card and one real simulator trial.
"$TASK_PYTHON" 'scripts/[train]gpu_probe.py' --expected-count 1 --purpose evaluation --baige-container --output "$TASK_OUTPUT/gpu-probe.json"
"$TASK_PYTHON" -m pytest -q -p no:cacheprovider tests/test_metaworld_clock_diagnose.py \
 'tests/test_metaworld_pilot.py::test_real_simulator_trial_and_disjoint_pools[ml10]' > "$TASK_OUTPUT/tests.log" 2>&1
timeout --signal=TERM --kill-after=20s 820 "$TASK_PYTHON" -m fluffy_bassoon.meta_rl.metaworld_clock_diagnose \
 --source-commit "$TASK_SOURCE" --parent-dir "$TASK_PARENT" --output-dir "$TASK_OUTPUT/diagnosis"
