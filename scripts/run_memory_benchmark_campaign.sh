#!/bin/bash
# A bounded first-seed screen; confirmation choices require its saved evidence.
set -euo pipefail
TASK_SOURCE=${1:?exact source}
TASK_RUN=${2:?fresh campaign identity}
TASK_QUAL_RUN=${3:?successful qualification identity}
TASK_QUAL_SOURCE=${4:?qualified executable source}
TASK_DEADLINE=${5:?authorized campaign deadline, Unix seconds}
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_WORKTREE=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUTPUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_PYTHON=$TASK_ROOT/envs/metaworld-baselines-20261028-r1/bin/python
[[ $TASK_SOURCE =~ ^[0-9a-f]{40}$ && $TASK_QUAL_SOURCE =~ ^[0-9a-f]{40}$ ]]
[[ $TASK_RUN =~ ^metta-screen-baige-2026102[89]-r[0-9]+$ ]]
[[ $TASK_QUAL_RUN =~ ^metta-bench-qualification-2026102[89]-r[0-9]+$ ]]
[[ $TASK_DEADLINE =~ ^[0-9]{10}$ && $(hostname) =~ ^job-[a-z0-9]+-master-0$ ]]
cd "$TASK_WORKTREE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
test "$(cat "$TASK_ROOT/outputs/$TASK_QUAL_RUN/exit-code.txt")" = 0
git diff --quiet "$TASK_QUAL_SOURCE" "$TASK_SOURCE" -- src third_party
"$TASK_PYTHON" -c 'import json,sys; assert json.load(open(sys.argv[1]))["oracle_pass"]' \
  "$TASK_ROOT/outputs/$TASK_QUAL_RUN/pointmaze-headroom.json"
mkdir "$TASK_OUTPUT"
exec > >(tee "$TASK_OUTPUT/console.log") 2>&1
trap 'TASK_EXIT=$?; printf "%s\n" "$TASK_EXIT" > "$TASK_OUTPUT/exit-code.txt"' EXIT
export PYTHONPATH="$TASK_WORKTREE/src:$TASK_ROOT/deps/metta-robotics-1.4.2-r1"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 FLUFFY_GPU_EXECUTION=baige
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MUJOCO_GL=egl
TASK_STARTED=$(date +%s)
printf '%s\n' "$TASK_QUAL_RUN $TASK_QUAL_SOURCE" > "$TASK_OUTPUT/qualification-source.txt"
printf '%s\n' "$TASK_STARTED" > "$TASK_OUTPUT/allocation-start-unix.txt"
"$TASK_PYTHON" 'scripts/[train]gpu_probe.py' --expected-count 1 --purpose training \
  --baige-container --output "$TASK_OUTPUT/gpu-probe.json"
# All cells use the same physical GPU, worker backend, parameter budget and seed.
# Budgets differ across tasks; within each task every core has the same contract.
for TASK_BENCH in halfcheetah-vel pointmaze ml1-push; do
  if [[ $TASK_BENCH = ml1-push ]]; then
    "$TASK_PYTHON" -m fluffy_bassoon.meta_rl.metaworld_train \
      --source-commit "$TASK_SOURCE" --output-dir "$TASK_OUTPUT/qualification-ml1-cmm" \
      --benchmark ml1-push --core cmm --qualify-one-update --updates 1 \
      --num-envs 10 --attempts 2 --horizon 500 --ppo-epochs 1 --minibatch-trials 5 \
      --replay-chunk-length 100 --device cuda --env-workers 8 --allocated-cpus 16
  fi
  case "$TASK_BENCH" in
    halfcheetah-vel) TASK_ENVS=50; TASK_ATTEMPTS=5; TASK_HORIZON=100; TASK_EPOCHS=10;
      TASK_BUDGET=1000000; TASK_UPDATES=40; TASK_EVAL_ATTEMPTS=10; TASK_EVAL_LANES=24; TASK_EVAL_EVERY=10;;
    pointmaze) TASK_ENVS=25; TASK_ATTEMPTS=10; TASK_HORIZON=500; TASK_EPOCHS=5;
      TASK_BUDGET=1500000; TASK_UPDATES=100; TASK_EVAL_ATTEMPTS=10; TASK_EVAL_LANES=12; TASK_EVAL_EVERY=4;;
    ml1-push) TASK_ENVS=50; TASK_ATTEMPTS=5; TASK_HORIZON=500; TASK_EPOCHS=10;
      TASK_BUDGET=2500000; TASK_UPDATES=20; TASK_EVAL_ATTEMPTS=10; TASK_EVAL_LANES=10; TASK_EVAL_EVERY=5;;
  esac
  for TASK_CORE in gru transformer_xl persistent_ctm cmm; do
    TASK_NOW=$(date +%s)
    TASK_LEFT=$((TASK_DEADLINE-TASK_NOW-600))
    TASK_STAGE_LEFT=$((10800-TASK_NOW+TASK_STARTED-600))
    if ((TASK_LEFT < 300 || TASK_STAGE_LEFT < 300)); then
      printf '%s\n' "No new cell: campaign or three-hour screen budget reached." > "$TASK_OUTPUT/budget-stop.txt"
      exit 0
    fi
    TASK_CAP=1800
    ((TASK_LEFT >= TASK_CAP)) || TASK_CAP=$TASK_LEFT
    ((TASK_STAGE_LEFT >= TASK_CAP)) || TASK_CAP=$TASK_STAGE_LEFT
    TASK_CELL=$TASK_BENCH-$TASK_CORE-seed1000
    TASK_CELL_START=$(date +%s)
    printf '%s\n' "$TASK_CELL $TASK_CELL_START" >> "$TASK_OUTPUT/cell-starts.tsv"
    "$TASK_PYTHON" -m fluffy_bassoon.meta_rl.metaworld_train \
      --source-commit "$TASK_SOURCE" --output-dir "$TASK_OUTPUT/$TASK_CELL" \
      --benchmark "$TASK_BENCH" --core "$TASK_CORE" --seed 1000 --benchmark-seed 0 \
      --updates "$TASK_UPDATES" --target-environment-steps "$TASK_BUDGET" \
      --num-envs "$TASK_ENVS" --attempts "$TASK_ATTEMPTS" --horizon "$TASK_HORIZON" \
      --ppo-epochs "$TASK_EPOCHS" --minibatch-trials 10 --replay-chunk-length 100 \
      --eval-every "$TASK_EVAL_EVERY" --eval-attempts "$TASK_EVAL_ATTEMPTS" --eval-lanes "$TASK_EVAL_LANES" \
      --max-elapsed-seconds "$TASK_CAP" --device cuda --threads 4 --env-workers 8 --allocated-cpus 16
    printf '%s\n' "$TASK_CELL $(($(date +%s)-TASK_CELL_START))" >> "$TASK_OUTPUT/cell-allocation-seconds.tsv"
  done
done
printf '%s\n' "$(($(date +%s)-TASK_STARTED))" > "$TASK_OUTPUT/allocation-active-seconds.txt"
