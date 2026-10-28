#!/bin/bash
set -euo pipefail
TASK_SOURCE=${1:?exact source commit}
TASK_RUN=${2:?fresh run identity}
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_WORKTREE=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUTPUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_PYTHON=$TASK_ROOT/envs/metaworld-baselines-20261028-r1/bin/python
TASK_DEPS=$TASK_ROOT/deps/metta-robotics-1.4.2-r1
[[ $TASK_SOURCE =~ ^[0-9a-f]{40}$ ]]
[[ $TASK_RUN =~ ^metta-bench-qualification-2026102[89]-r[0-9]+$ ]]
[[ $(hostname) =~ ^job-[a-z0-9]+-master-0$ ]]
cd "$TASK_WORKTREE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
mkdir "$TASK_OUTPUT"
exec > >(tee "$TASK_OUTPUT/console.log") 2>&1
trap 'TASK_EXIT=$?; printf "%s\n" "$TASK_EXIT" > "$TASK_OUTPUT/exit-code.txt"' EXIT
export PYTHONPATH="$TASK_WORKTREE/src:$TASK_DEPS" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export FLUFFY_GPU_EXECUTION=baige CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MUJOCO_GL=egl
export FLUFFY_META_RL_CUDA_QUALIFICATION=1 FLUFFY_METTA_REAL_QUALIFICATION=1
"$TASK_PYTHON" 'scripts/[train]gpu_probe.py' --expected-count 1 --purpose training \
  --baige-container --output "$TASK_OUTPUT/gpu-probe.json"
"$TASK_PYTHON" -m pytest -q -p no:cacheprovider \
  tests/test_metta_benchmarks.py tests/test_metaworld_pilot.py tests/test_metaworld_workers.py \
  tests/test_meta_rl_replay_windows.py tests/test_meta_rl_ppo.py tests/test_meta_rl_checkpoint.py \
  tests/test_meta_rl_cmm.py tests/test_ctm_sync_options.py tests/test_meta_rl_cuda.py \
  > "$TASK_OUTPUT/tests.log" 2>&1
"$TASK_PYTHON" -m fluffy_bassoon.meta_rl.metta_qualification \
  --source-commit "$TASK_SOURCE" --output "$TASK_OUTPUT/pointmaze-headroom.json"
for TASK_BENCH in pointmaze halfcheetah-vel; do
  for TASK_CORE in persistent_ctm cmm gru transformer_xl; do
    if [[ $TASK_BENCH = pointmaze ]]; then
      TASK_ENVS=25; TASK_ATTEMPTS=10; TASK_HORIZON=500; TASK_EPOCHS=5; TASK_CHUNK=100
    else
      TASK_ENVS=50; TASK_ATTEMPTS=5; TASK_HORIZON=100; TASK_EPOCHS=10; TASK_CHUNK=100
    fi
    "$TASK_PYTHON" -m fluffy_bassoon.meta_rl.metaworld_train \
      --benchmark "$TASK_BENCH" --core "$TASK_CORE" --source-commit "$TASK_SOURCE" \
      --output-dir "$TASK_OUTPUT/$TASK_BENCH-$TASK_CORE" --qualify-one-update --updates 1 \
      --num-envs "$TASK_ENVS" --attempts "$TASK_ATTEMPTS" --horizon "$TASK_HORIZON" \
      --ppo-epochs "$TASK_EPOCHS" --minibatch-trials 10 --replay-chunk-length "$TASK_CHUNK" \
      --max-elapsed-seconds 1200 --device cuda --env-workers 8 --allocated-cpus 16
  done
done
