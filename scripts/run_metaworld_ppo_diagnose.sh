#!/bin/bash
set -euo pipefail
TASK_SOURCE=${1:?exact diagnostic source}
TASK_RUN=${2:?fresh diagnostic identity}
TASK_PARENT=${3:?completed parent train directory}
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_WORKTREE=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUTPUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_PYTHON=$TASK_ROOT/envs/metaworld-baselines-20261028-r1/bin/python
[[ $TASK_SOURCE =~ ^[0-9a-f]{40}$ ]]
[[ $TASK_RUN =~ ^metaworld-ppo-debug-baige-20261029-r[0-9]+$ ]]
[[ $(hostname) =~ ^job-[a-z0-9]+-master-0$ ]]
[[ $TASK_PARENT = "$TASK_ROOT"/outputs/*/train ]]
# One card per method: the original method must have released its job first.
test "$(cat "$(dirname "$TASK_PARENT")/exit-code.txt")" = 0
TASK_PARENT_SECONDS=$(cat "$(dirname "$TASK_PARENT")/allocation-active-seconds.txt")
[[ $TASK_PARENT_SECONDS =~ ^[0-9]+$ ]]
TASK_PARENT_CAP=$("$TASK_PYTHON" -c 'import json,sys; c=json.load(open(sys.argv[1]))["contract"]["core"]; assert c in ("persistent_ctm","cmm"); print(21600 if c=="persistent_ctm" else 10800)' "$TASK_PARENT/config.json")
TASK_PARENT_SOURCE=$("$TASK_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["contract"]["source_commit"])' "$TASK_PARENT/config.json")
# The30min diagnostic must fit the original method's total cost cap.
(( TASK_PARENT_SECONDS + 1800 <= TASK_PARENT_CAP ))
cd "$TASK_WORKTREE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
git diff --quiet "$TASK_PARENT_SOURCE" "$TASK_SOURCE" -- \
  src/fluffy_bassoon/meta_rl/{ppo,metaworld_policy,metaworld_env,metaworld_workers,checkpoint,cores,cmm}.py \
  src/lerobot_policy_ctm
mkdir "$TASK_OUTPUT"
exec > >(tee "$TASK_OUTPUT/console.log") 2>&1
TASK_STARTED=$(date +%s)
trap 'TASK_EXIT=$?; printf "%s\n" "$TASK_EXIT" > "$TASK_OUTPUT/exit-code.txt"; printf "%s\n" "$(($(date +%s)-TASK_STARTED))" > "$TASK_OUTPUT/allocation-active-seconds.txt"' EXIT
export PYTHONPATH="$TASK_WORKTREE/src" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export FLUFFY_GPU_EXECUTION=baige CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MUJOCO_GL=egl WANDB_MODE=disabled
export FLUFFY_META_RL_CUDA_QUALIFICATION=1
export FLUFFY_METAWORLD_REAL_QUALIFICATION=1
"$TASK_PYTHON" 'scripts/[train]gpu_probe.py' --expected-count 1 --purpose training \
  --baige-container --output "$TASK_OUTPUT/gpu-probe.json"
"$TASK_PYTHON" -m pytest -q -p no:cacheprovider tests/test_metaworld_ppo_diagnose.py \
  tests/test_meta_rl_replay_windows.py tests/test_metaworld_pilot.py > "$TASK_OUTPUT/tests.log" 2>&1
timeout --signal=TERM --kill-after=30s 1500 "$TASK_PYTHON" -m fluffy_bassoon.meta_rl.metaworld_ppo_diagnose \
  --source-commit "$TASK_SOURCE" --parent-dir "$TASK_PARENT" --output-dir "$TASK_OUTPUT/diagnosis" \
  --compare-minibatch-trials 2
