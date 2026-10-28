#!/bin/bash
set -euo pipefail
TASK_SOURCE=${1:?exact source}
TASK_RUN=${2:?fresh control identity}
TASK_COEFFICIENT=${3:?measured candidate value coefficient}
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_PARENT=$TASK_ROOT/outputs/cmm-ml10-matched-baige-20261029-r1/train
TASK_REFERENCE=$TASK_ROOT/outputs/metaworld-ppo-debug-baige-20261029-r2/diagnosis
TASK_WORKTREE=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUTPUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_PYTHON=$TASK_ROOT/envs/metaworld-baselines-20261028-r1/bin/python
[[ $TASK_SOURCE =~ ^[0-9a-f]{40}$ && $TASK_RUN =~ ^metaworld-ppo-controls-baige-20261029-r[0-9]+$ ]]
[[ $TASK_COEFFICIENT =~ ^0\.[0-9]+$ ]]
[[ $(hostname) =~ ^job-[a-z0-9]+-master-0$ ]]
test "$(cat "$(dirname "$TASK_PARENT")/exit-code.txt")" = 0
test "$(cat "$(dirname "$TASK_REFERENCE")/exit-code.txt")" = 0
TASK_PARENT_SECONDS=$(cat "$(dirname "$TASK_PARENT")/allocation-active-seconds.txt")
TASK_DIAG_SECONDS=$(cat "$(dirname "$TASK_REFERENCE")/allocation-active-seconds.txt")
(( TASK_PARENT_SECONDS + TASK_DIAG_SECONDS + 2700 <= 10800 ))
cd "$TASK_WORKTREE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
git diff --quiet bcf3d98242b309e52391cee4c1f783adebcf2d1b "$TASK_SOURCE" -- \
 src/fluffy_bassoon/meta_rl/{ppo,metaworld_policy,metaworld_env,metaworld_workers,checkpoint,cores,cmm,metaworld_train}.py src/lerobot_policy_ctm
mkdir "$TASK_OUTPUT"
exec > >(tee "$TASK_OUTPUT/console.log") 2>&1
TASK_STARTED=$(date +%s)
trap 'TASK_EXIT=$?; printf "%s\n" "$TASK_EXIT" > "$TASK_OUTPUT/exit-code.txt"; printf "%s\n" "$(($(date +%s)-TASK_STARTED))" > "$TASK_OUTPUT/allocation-active-seconds.txt"' EXIT
export PYTHONPATH="$TASK_WORKTREE/src" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export FLUFFY_GPU_EXECUTION=baige CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MUJOCO_GL=egl WANDB_MODE=disabled
export FLUFFY_META_RL_CUDA_QUALIFICATION=1 FLUFFY_METAWORLD_REAL_QUALIFICATION=1
"$TASK_PYTHON" 'scripts/[train]gpu_probe.py' --expected-count 1 --purpose training --baige-container --output "$TASK_OUTPUT/gpu-probe.json"
"$TASK_PYTHON" -m pytest -q -p no:cacheprovider tests/test_metaworld_ppo_diagnose.py \
 tests/test_meta_rl_replay_windows.py tests/test_metaworld_pilot.py tests/test_meta_rl_cmm.py \
 tests/test_continuous_memory_machine.py > "$TASK_OUTPUT/tests.log" 2>&1
timeout --signal=TERM --kill-after=30s 2520 "$TASK_PYTHON" -m fluffy_bassoon.meta_rl.metaworld_value_control \
 --source-commit "$TASK_SOURCE" --parent-dir "$TASK_PARENT" --reference-diagnostic "$TASK_REFERENCE" \
 --output-dir "$TASK_OUTPUT/control" --updates 10 --candidate-value-coef "$TASK_COEFFICIENT" --max-elapsed-seconds 2400
