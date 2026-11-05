#!/usr/bin/env bash
set -Eeuo pipefail
TASK_SOURCE=${1:?exact reviewed project commit}
TASK_OUT=${2:?fresh CPU output}
TASK_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
[[ $(uname -s) == Linux && $(hostname) == job-* ]]
[[ $(git -C "$TASK_CODE" rev-parse HEAD) == "$TASK_SOURCE" ]]
export TASK_SOURCE TASK_CODE
export REPO_PATH=/mowenjie/fluffy-bassoon-2/sources/robotwin-baseline-smoke-20261028-r1/RLinf
export ROBOTWIN_PATH=/mowenjie/fluffy-bassoon-2/sources/robotwin-baseline-smoke-20261028-r1/RoboTwin
export EMBODIED_PATH=$REPO_PATH/examples/embodiment ROBOTWIN_ASSETS_PATH=$ROBOTWIN_PATH
export PYTHONPATH=$TASK_CODE/src:$REPO_PATH:$ROBOTWIN_PATH
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export NATIVE_PPO_OUTPUT=$TASK_OUT NATIVE_PPO_ACCOUNTING_ROOT=$TASK_OUT/accounting
export NATIVE_PPO_WORLD=8 NATIVE_PPO_PLACEMENT=0-7 NATIVE_PPO_DEADLINE=9999999999 NATIVE_PPO_MODE=train
export NATIVE_PPO_MODEL=/mowenjie/fluffy-bassoon-2/cache/robotwin-ppo-pilot-20261105-r1-restore02/pi05-motus-effc6e1a-r1
export NATIVE_PPO_INITIALIZER_REVISION=effc6e1a99bd4d0edf9447e8ba313def67649cfe
exec /mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python \
  "$TASK_CODE/outputs/robotwin-ppo-source-pilot-20261105-r1/cpu_check.py" "$TASK_OUT"
