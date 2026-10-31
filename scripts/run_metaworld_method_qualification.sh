#!/bin/bash
set -euo pipefail
TASK_SOURCE=${1:?exact source commit}
TASK_RUN=${2:?fresh output identity}
TASK_METHOD=${3:?ctm, rl2, ecet or amago}
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_WORKTREE=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUTPUT=$TASK_ROOT/outputs/$TASK_RUN/$TASK_METHOD
TASK_PYTHON=$TASK_ROOT/envs/metaworld-baselines-20261028-r1/bin/python
[[ $TASK_SOURCE =~ ^[0-9a-f]{40}$ ]]
[[ $TASK_RUN =~ ^metaworld-method-qual-baige-20261028-r[0-9]+$ ]]
[[ $TASK_METHOD =~ ^(ctm|rl2|ecet|amago)$ ]]
[[ $(hostname) =~ ^job-[a-z0-9]+-master-0$ ]]
cd "$TASK_WORKTREE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
mkdir -p "$(dirname "$TASK_OUTPUT")"
mkdir "$TASK_OUTPUT"
exec > >(tee "$TASK_OUTPUT/console.log") 2>&1
trap 'TASK_EXIT=$?; printf "%s\n" "$TASK_EXIT" > "$TASK_OUTPUT/exit-code.txt"' EXIT
export PYTHONPATH="$TASK_WORKTREE/src" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export FLUFFY_GPU_EXECUTION=baige CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MUJOCO_GL=egl WANDB_MODE=disabled
export FLUFFY_META_RL_CUDA_QUALIFICATION=1
TASK_STARTED=$(date +%s)
"$TASK_PYTHON" 'scripts/[train]gpu_probe.py' --expected-count 1 --purpose training \
  --baige-container --output "$TASK_OUTPUT/gpu-probe.json"
"$TASK_ROOT/tools/uv-20261028/bin/uv" pip freeze --python "$TASK_PYTHON" > "$TASK_OUTPUT/requirements-freeze.txt"
case "$TASK_METHOD" in
ctm)
  FLUFFY_METAWORLD_REAL_QUALIFICATION=1 "$TASK_PYTHON" -m pytest -q -p no:cacheprovider \
    tests/test_meta_rl_ppo.py tests/test_meta_rl_checkpoint.py tests/test_meta_rl_replay_windows.py \
    tests/test_metaworld_pilot.py tests/test_meta_rl_cuda.py tests/test_meta_rl_structure.py \
    tests/test_ctm_sync_options.py tests/test_metaworld_workers.py \
    > "$TASK_OUTPUT/qualification.log" 2>&1
  "$TASK_PYTHON" -m fluffy_bassoon.meta_rl.metaworld_collect_timing \
    --source-commit "$TASK_SOURCE" --output-dir "$TASK_OUTPUT/collection-timing" \
    --num-envs 50 --workers 8 --allocated-cpus 16
  "$TASK_PYTHON" -m fluffy_bassoon.meta_rl.metaworld_train --benchmark ml10 --core persistent_ctm \
    --source-commit "$TASK_SOURCE" --output-dir "$TASK_OUTPUT/default-update" \
    --qualify-one-update --updates 1 --num-envs 50 --attempts 5 --horizon 500 \
    --ctm-ticks 1 --ppo-epochs 10 --minibatch-trials 10 --replay-chunk-length 100 \
    --eval-attempts 25 --eval-lanes 10 --max-elapsed-seconds 2400 --device cuda --env-workers 8 --allocated-cpus 16
  ;;
rl2)
  "$TASK_PYTHON" -m pytest -q -s -p no:cacheprovider tests/test_metaworld_rl2.py tests/test_metaworld_workers.py \
    > "$TASK_OUTPUT/qualification.log" 2>&1
  "$TASK_PYTHON" -m fluffy_bassoon.meta_rl.metaworld_rl2 \
    --source-commit "$TASK_SOURCE" --output-dir "$TASK_OUTPUT/default-update" \
    --qualify-one-update --updates 1 --num-envs 50 --attempts 5 --horizon 500 \
    --ppo-epochs 10 --eval-attempts 25 --eval-lanes 10 --max-elapsed-seconds 2400 --device cuda --env-workers 8 --allocated-cpus 16
  ;;
ecet)
  "$TASK_PYTHON" third_party/ecet/fetch_reference.py "$TASK_OUTPUT/ecet-reference"
  FLUFFY_ECET_UPSTREAM="$TASK_OUTPUT/ecet-reference" "$TASK_PYTHON" -m pytest -q -p no:cacheprovider \
    tests/test_metaworld_ecet.py -k 'not real_ml10' > "$TASK_OUTPUT/qualification.log" 2>&1
  FLUFFY_METAWORLD_REAL_QUALIFICATION=1 "$TASK_PYTHON" -m pytest -q -p no:cacheprovider \
    tests/test_metaworld_ecet.py::test_real_ml10_ecet_trial_and_behavior_replay \
    > "$TASK_OUTPUT/real-simulator.log" 2>&1
  "$TASK_PYTHON" -m fluffy_bassoon.meta_rl.metaworld_ecet \
    --source-commit "$TASK_SOURCE" --output-dir "$TASK_OUTPUT/default-update" \
    --qualify-one-update --updates 1 --num-envs 50 --attempts 5 --horizon 500 \
    --ppo-epochs 10 --minibatch-size 32 --eval-attempts 25 --eval-lanes 10 \
    --max-elapsed-seconds 2400 --device cuda --env-workers 8 --allocated-cpus 16
  ;;
amago)
  "$TASK_PYTHON" -m pytest -q -p no:cacheprovider tests/test_metaworld_amago.py \
    > "$TASK_OUTPUT/qualification.log" 2>&1
  FLUFFY_METAWORLD_REAL_QUALIFICATION=1 FLUFFY_AMAGO_ATTENTION=torch-sdpa \
    "$TASK_PYTHON" -m pytest -q -p no:cacheprovider tests/test_metaworld_amago.py -k real_ml10 \
    > "$TASK_OUTPUT/real-simulator.log" 2>&1
  FLUFFY_AMAGO_DEFAULT_CAPACITY=1 FLUFFY_AMAGO_ATTENTION=torch-sdpa \
    FLUFFY_AMAGO_CAPACITY_OUTPUT="$TASK_OUTPUT/default-capacity.json" \
    "$TASK_PYTHON" -m pytest -q -s -p no:cacheprovider tests/test_metaworld_amago.py -k default_b24_l2000 \
    > "$TASK_OUTPUT/default-capacity.log" 2>&1
  ;;
esac
printf '%s\n' "$(($(date +%s)-TASK_STARTED))" > "$TASK_OUTPUT/allocation-active-seconds.txt"
