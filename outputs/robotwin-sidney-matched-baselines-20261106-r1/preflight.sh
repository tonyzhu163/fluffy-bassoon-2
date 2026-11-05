#!/usr/bin/env bash
set -Eeuo pipefail
TASK_SOURCE=${1:?exact reviewed source}
TASK_OUT=${2:?fresh metadata-only output}
TASK_ALLOCATED_HOST=${3:?exact owned allocated job hostname}
[[ $(uname -s) == Linux && $(hostname) == "$TASK_ALLOCATED_HOST" && $TASK_ALLOCATED_HOST == job-* ]]
TASK_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_RUN=$TASK_CODE/outputs/robotwin-sidney-matched-baselines-20261106-r1
TASK_SHARED=$TASK_CODE/outputs/robotwin-sidney-rlinf-ppo-20261105-r1
[[ $(git -C "$TASK_CODE" rev-parse HEAD) == "$TASK_SOURCE" && ! -e $TASK_OUT ]]
[[ -z $(git -C "$TASK_CODE" status --porcelain --untracked-files=no) ]]
mkdir -p "$TASK_OUT"
export REPO_PATH=/mowenjie/fluffy-bassoon-2/sources/robotwin-baseline-smoke-20261028-r1/RLinf
export ROBOTWIN_PATH=/mowenjie/fluffy-bassoon-2/sources/robotwin-baseline-smoke-20261028-r1/RoboTwin
export ROBOTWIN_ASSETS_PATH=$ROBOTWIN_PATH ASSETS_PATH=$ROBOTWIN_PATH
export EMBODIED_PATH=$REPO_PATH/examples/embodiment
export PYTHONPATH=$TASK_CODE/src:$REPO_PATH:$ROBOTWIN_PATH:$TASK_RUN:$TASK_SHARED
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NATIVE_PPO_WORLD=4 NATIVE_PPO_PLACEMENT=0-3 NATIVE_PPO_MODE=train
export NATIVE_PPO_OUTPUT=$TASK_OUT NATIVE_PPO_ACCOUNTING_ROOT=$TASK_OUT/action-accounting
export NATIVE_PPO_ATTEMPT=metadata-only NATIVE_PPO_DEADLINE=0
export NATIVE_PPO_MODEL=/mowenjie/fluffy-bassoon-2/cache/robotwin-sidney-rlinf-20261105-r1
export NATIVE_PPO_INITIALIZER_REVISION=e49e2ab6c11f07511573b67261bd129e88d0a416
/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python \
  "$TASK_RUN/configuration.py" validate_all "$TASK_ALLOCATED_HOST" > "$TASK_OUT/RESULT.json" 2> "$TASK_OUT/stderr.log"
cat "$TASK_OUT/RESULT.json"
