#!/bin/bash
# CMM on the existing ML10 task/data/schedule, qualified before fresh training.
set -euo pipefail
TASK_SOURCE=${1:?exact source commit}
TASK_RUN=${2:?fresh run identity}
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_WORKTREE=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUTPUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_PYTHON=$TASK_ROOT/envs/metaworld-baselines-20261028-r1/bin/python
[[ $TASK_SOURCE =~ ^[0-9a-f]{40}$ ]]
[[ $TASK_RUN =~ ^cmm-ml10-matched-baige-20261029-r[0-9]+$ ]]
[[ $(hostname) =~ ^job-[a-z0-9]+-master-0$ ]]
cd "$TASK_WORKTREE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
mkdir "$TASK_OUTPUT"
exec > >(tee "$TASK_OUTPUT/console.log") 2>&1
trap 'TASK_EXIT=$?; printf "%s\n" "$TASK_EXIT" > "$TASK_OUTPUT/exit-code.txt"' EXIT
export PYTHONPATH="$TASK_WORKTREE/src" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export FLUFFY_GPU_EXECUTION=baige CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MUJOCO_GL=egl
TASK_STARTED=$(date +%s)
printf '%s\n' "$TASK_STARTED" > "$TASK_OUTPUT/allocation-start-unix.txt"
"$TASK_PYTHON" 'scripts/[train]gpu_probe.py' --expected-count 1 --purpose training \
  --baige-container --output "$TASK_OUTPUT/gpu-probe.json"
FLUFFY_METAWORLD_REAL_QUALIFICATION=1 FLUFFY_META_RL_CUDA_QUALIFICATION=1 \
  "$TASK_PYTHON" -m pytest -q -p no:cacheprovider tests/test_metaworld_pilot.py \
  tests/test_metaworld_workers.py tests/test_meta_rl_cmm.py tests/test_meta_rl_replay_windows.py \
  tests/test_continuous_memory_machine.py > "$TASK_OUTPUT/tests.log" 2>&1
TASK_COMMON=(--benchmark ml10 --core cmm --source-commit "$TASK_SOURCE" --seed 1000 \
  --benchmark-seed 0 --num-envs 50 --attempts 5 --horizon 500 --parameter-budget 250000 \
  --ppo-epochs 10 --minibatch-trials 10 --replay-chunk-length 100 --ltm-tokens 32 \
  --device cuda --threads 4 --env-workers 8 --allocated-cpus 16 --eval-attempts 25 --eval-lanes 10)
"$TASK_PYTHON" -m fluffy_bassoon.meta_rl.metaworld_train "${TASK_COMMON[@]}" \
  --output-dir "$TASK_OUTPUT/qualification" --qualify-one-update --updates 1 --max-elapsed-seconds 1800
# A throughput checkpoint cannot silently become a learning initializer.
"$TASK_PYTHON" -m fluffy_bassoon.meta_rl.metaworld_train "${TASK_COMMON[@]}" \
  --output-dir "$TASK_OUTPUT/train" --updates 80 --eval-every 5 --max-elapsed-seconds 9600
printf '%s\n' "$(($(date +%s)-TASK_STARTED))" > "$TASK_OUTPUT/allocation-active-seconds.txt"
