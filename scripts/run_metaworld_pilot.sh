#!/bin/bash
# Invoke only inside a fresh exclusive one-GPU Baige scheduler allocation.
set -euo pipefail
TASK_SOURCE=${1:?exact source commit}
TASK_RUN=${2:?fresh output identity}
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_WORKTREE=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUTPUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_PYTHON=$TASK_ROOT/envs/metaworld-3.1.1-r1/bin/python
[[ $TASK_SOURCE =~ ^[0-9a-f]{40}$ ]]
[[ $TASK_RUN =~ ^metaworld-ctm-pilot-baige-20261028-r[0-9]+$ ]]
[[ $(hostname) =~ ^job-[a-z0-9]+-master-0$ ]]
cd "$TASK_WORKTREE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
mkdir "$TASK_OUTPUT"
exec > >(tee "$TASK_OUTPUT/console.log") 2>&1
trap 'TASK_EXIT=$?; printf "%s\n" "$TASK_EXIT" > "$TASK_OUTPUT/exit-code.txt"' EXIT
export PYTHONPATH="$TASK_WORKTREE/src" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export FLUFFY_GPU_EXECUTION=baige
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MUJOCO_GL=egl
TASK_LAUNCH_STARTED=$(date +%s)
hostname
date -u
"$TASK_PYTHON" 'scripts/[train]gpu_probe.py' --expected-count 1 --purpose training \
  --baige-container --output "$TASK_OUTPUT/gpu-probe.json"
"$TASK_PYTHON" -c 'import importlib.metadata as m; print("\n".join(sorted({str(d.metadata["Name"])+"=="+d.version for d in m.distributions()})))' \
  > "$TASK_OUTPUT/package-freeze.txt"
FLUFFY_METAWORLD_REAL_QUALIFICATION=1 "$TASK_PYTHON" -m pytest -q -p no:cacheprovider \
  tests/test_metaworld_pilot.py tests/test_meta_rl_ppo.py tests/test_meta_rl_checkpoint.py \
  > "$TASK_OUTPUT/qualification.log" 2>&1
for TASK_BENCHMARK in ml1-push ml10; do
  for TASK_CORE in gru persistent_ctm transformer_xl; do
    "$TASK_PYTHON" -m fluffy_bassoon.meta_rl.metaworld_train \
      --benchmark "$TASK_BENCHMARK" --core "$TASK_CORE" --seed 1000 \
      --source-commit "$TASK_SOURCE" --output-dir "$TASK_OUTPUT/$TASK_BENCHMARK-$TASK_CORE-seed1000" \
      --num-envs 10 --attempts 5 --horizon 500 --updates 20 --ppo-epochs 1 \
      --minibatch-trials 2 --eval-every 2 --eval-attempts 5 --eval-lanes 10 \
      --max-elapsed-seconds 900 --threads 4
    "$TASK_PYTHON" -m fluffy_bassoon.meta_rl.metaworld_plot "$TASK_OUTPUT"
  done
done
printf '%s\n' "$(($(date +%s)-TASK_LAUNCH_STARTED))" > "$TASK_OUTPUT/allocation-active-seconds.txt"
