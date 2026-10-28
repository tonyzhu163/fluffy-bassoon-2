#!/bin/bash
set -euo pipefail
TASK_SOURCE=${1:?exact source commit}
TASK_RUN=${2:?fresh output identity}
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_WORKTREE=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUTPUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_PYTHON=$TASK_ROOT/envs/metaworld-3.1.1-r1/bin/python
[[ $TASK_SOURCE =~ ^[0-9a-f]{40}$ ]]
[[ $TASK_RUN =~ ^metaworld-ppo-opt-baige-20261028-r[0-9]+$ ]]
[[ $(hostname) =~ ^job-[a-z0-9]+-master-0$ ]]
cd "$TASK_WORKTREE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
mkdir "$TASK_OUTPUT"
exec > >(tee "$TASK_OUTPUT/console.log") 2>&1
trap 'TASK_EXIT=$?; printf "%s\n" "$TASK_EXIT" > "$TASK_OUTPUT/exit-code.txt"' EXIT
export PYTHONPATH="$TASK_WORKTREE/src" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export FLUFFY_GPU_EXECUTION=baige
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MUJOCO_GL=egl
TASK_STARTED=$(date +%s)
"$TASK_PYTHON" 'scripts/[train]gpu_probe.py' --expected-count 1 --purpose training \
  --baige-container --output "$TASK_OUTPUT/gpu-probe.json"
FLUFFY_METAWORLD_REAL_QUALIFICATION=1 FLUFFY_META_RL_CUDA_QUALIFICATION=1 \
  "$TASK_PYTHON" -m pytest -q -p no:cacheprovider tests/test_meta_rl_ppo.py \
  tests/test_meta_rl_checkpoint.py tests/test_meta_rl_replay_windows.py tests/test_metaworld_pilot.py \
  tests/test_meta_rl_cuda.py tests/test_meta_rl_structure.py > "$TASK_OUTPUT/qualification.log" 2>&1
"$TASK_PYTHON" -m fluffy_bassoon.meta_rl.metaworld_ppo_timing \
  --source-commit "$TASK_SOURCE" --output-dir "$TASK_OUTPUT/timing-t1" --ctm-ticks 1
printf '%s\n' "$(($(date +%s)-TASK_STARTED))" > "$TASK_OUTPUT/allocation-active-seconds.txt"
