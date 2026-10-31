#!/usr/bin/env bash
# CPU-only validation of a pinned production batching candidate; never loads a VLA.
set -Eeuo pipefail
TASK_COMMIT=${1:?exact 40-hex source commit}
TASK_RUN=${2:?fresh validation identity}
[[ "$TASK_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$TASK_RUN" =~ ^robotwin-production-batch-cpu-[a-zA-Z0-9-]+$ ]]
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_WORKTREE=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUTPUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_PYTHON=$TASK_ROOT/envs/robotwin-vlrla-20261029-r1/bin/python
test "$(uname -s)" = Linux
test ! -e "$TASK_OUTPUT"
cd "$TASK_WORKTREE"
test "$(git rev-parse HEAD)" = "$TASK_COMMIT"
test -z "$(git status --porcelain)"
mkdir -p "$TASK_OUTPUT"
exec > >(tee "$TASK_OUTPUT/console.log") 2>&1
trap 'TASK_EXIT=$?; printf "%s\n" "$TASK_EXIT" > "$TASK_OUTPUT/exit-code.txt"' EXIT
export CUDA_VISIBLE_DEVICES='' PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=$TASK_WORKTREE/src
export TMPDIR=/dev/shm
timeout 360 "$TASK_PYTHON" -m pytest -q -x --tb=short -p no:cacheprovider \
  --basetemp "/dev/shm/$TASK_RUN" \
  tests/test_robotwin_noise.py tests/test_robotwin_assets.py \
  tests/test_robotwin_fast_observations.py tests/test_vla_noise_ppo.py \
  tests/test_vla_noise_sac.py tests/test_vla_noise_ppo_review.py \
  tests/test_robotwin_campaign_budget.py tests/test_robotwin_corpus.py \
  tests/test_robotwin_campaign_driver.py tests/test_robotwin_parallel_campaign.py \
  tests/test_vla_noise_batched_collection.py tests/test_collection_admission.py \
  tests/test_robotwin_production_dispatch.py tests/test_robotwin_production_vector.py \
  tests/test_robotwin_packed_ppo_probe.py
