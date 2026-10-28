#!/bin/bash
set -euo pipefail
TASK_SOURCE=${1:?exact source}
TASK_RUN=${2:?fresh run identity}
TASK_METHOD=${3:?ctm rl2 ecet amago}
TASK_SOFT_CAP=${4:?active seconds before final evaluation}
TASK_QUAL_RUN=${5:?successful qualification run identity}
TASK_QUAL_SOURCE=${6:?qualification source commit}
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_WORKTREE=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUTPUT=$TASK_ROOT/outputs/$TASK_RUN/$TASK_METHOD
TASK_QUAL_OUTPUT=$TASK_ROOT/outputs/$TASK_QUAL_RUN/$TASK_METHOD
TASK_PYTHON=$TASK_ROOT/envs/metaworld-baselines-20261028-r1/bin/python
[[ $TASK_SOURCE =~ ^[0-9a-f]{40}$ && $TASK_QUAL_SOURCE =~ ^[0-9a-f]{40}$ ]]
[[ $TASK_RUN =~ ^metaworld-ml10-matched-baige-20261028-r[0-9]+$ ]]
[[ $TASK_QUAL_RUN =~ ^metaworld-method-qual-baige-20261028-r[0-9]+$ ]]
[[ $TASK_METHOD =~ ^(ctm|rl2|ecet|amago)$ && $TASK_SOFT_CAP =~ ^[0-9]+$ ]]
[[ $(hostname) =~ ^job-[a-z0-9]+-master-0$ ]]
cd "$TASK_WORKTREE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
test "$(cat "$TASK_QUAL_OUTPUT/exit-code.txt")" = 0
# Launch/source/test/docs can change after admission, executable method code cannot.
git diff --quiet "$TASK_QUAL_SOURCE" "$TASK_SOURCE" -- src third_party
mkdir -p "$(dirname "$TASK_OUTPUT")"
mkdir "$TASK_OUTPUT"
exec > >(tee "$TASK_OUTPUT/console.log") 2>&1
trap 'TASK_EXIT=$?; printf "%s\n" "$TASK_EXIT" > "$TASK_OUTPUT/exit-code.txt"' EXIT
export PYTHONPATH="$TASK_WORKTREE/src" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export FLUFFY_GPU_EXECUTION=baige CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MUJOCO_GL=egl WANDB_MODE=disabled
TASK_STARTED=$(date +%s)
printf '%s\n' "$TASK_QUAL_RUN $TASK_QUAL_SOURCE" > "$TASK_OUTPUT/qualification-source.txt"
"$TASK_PYTHON" 'scripts/[train]gpu_probe.py' --expected-count 1 --purpose training \
  --baige-container --output "$TASK_OUTPUT/gpu-probe.json"
"$TASK_ROOT/tools/uv-20261028/bin/uv" pip freeze --python "$TASK_PYTHON" > "$TASK_OUTPUT/requirements-freeze.txt"
TASK_COMMON=(--source-commit "$TASK_SOURCE" --output-dir "$TASK_OUTPUT/train" --seed 1000 \
  --benchmark-seed 0 --num-envs 50 --attempts 5 --horizon 500 --updates 80 \
  --eval-every 5 --eval-attempts 25 --eval-lanes 10 --env-workers 8 --allocated-cpus 16 \
  --max-elapsed-seconds "$TASK_SOFT_CAP" --device cuda --threads 4)
case "$TASK_METHOD" in
ctm)
  "$TASK_PYTHON" -m fluffy_bassoon.meta_rl.metaworld_train "${TASK_COMMON[@]}" \
    --benchmark ml10 --core persistent_ctm --ctm-ticks 1 --ppo-epochs 10 \
    --minibatch-trials 10 --replay-chunk-length 100
  ;;
rl2)
  "$TASK_PYTHON" -m fluffy_bassoon.meta_rl.metaworld_rl2 "${TASK_COMMON[@]}" --ppo-epochs 10
  ;;
ecet)
  "$TASK_PYTHON" -m fluffy_bassoon.meta_rl.metaworld_ecet "${TASK_COMMON[@]}" --ppo-epochs 10
  ;;
amago)
  "$TASK_PYTHON" -m fluffy_bassoon.meta_rl.metaworld_amago \
    @third_party/amago/ml10-amago1-sdpa-50env-10m-comparison.args "${TASK_COMMON[@]}"
  ;;
esac
printf '%s\n' "$(($(date +%s)-TASK_STARTED))" > "$TASK_OUTPUT/allocation-active-seconds.txt"
