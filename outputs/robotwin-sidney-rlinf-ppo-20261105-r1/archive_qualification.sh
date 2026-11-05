#!/usr/bin/env bash
set -Eeuo pipefail
TASK_SOURCE=${1:?exact reviewed source}
TASK_OUT=${2:?fresh output}
TASK_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_RUN_CODE=$TASK_CODE/outputs/robotwin-sidney-rlinf-ppo-20261105-r1
TASK_PYTHON=/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python
[[ $(hostname) == job-* && $(git -C "$TASK_CODE" rev-parse HEAD) == "$TASK_SOURCE" ]]
[[ ! -e $TASK_OUT && -z $(git -C "$TASK_CODE" status --porcelain --untracked-files=no) ]]
mkdir -p "$TASK_OUT"
exec > "$TASK_OUT/worker.log" 2>&1
export TASK_SOURCE TASK_OUT
export REPO_PATH=/mowenjie/fluffy-bassoon-2/sources/robotwin-baseline-smoke-20261028-r1/RLinf
export EMBODIED_PATH=$REPO_PATH/examples/embodiment
export PYTHONPATH=$TASK_CODE/src:$REPO_PATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NATIVE_PPO_WORLD=4 NATIVE_PPO_PLACEMENT=0-3 NATIVE_PPO_MODE=train
export NATIVE_PPO_OUTPUT=$TASK_OUT NATIVE_PPO_ACCOUNTING_ROOT=$TASK_OUT/not-used
export NATIVE_PPO_MODEL=/mowenjie/fluffy-bassoon-2/cache/robotwin-sidney-rlinf-20261105-r1
export NATIVE_PPO_INITIALIZER_REVISION=e49e2ab6c11f07511573b67261bd129e88d0a416
export NATIVE_PPO_DEADLINE=$(($(date +%s)+1500))
export ROBOTWIN_ASSETS_PATH=/mowenjie/fluffy-bassoon-2/sources/robotwin-baseline-smoke-20261028-r1/RoboTwin
TASK_STARTED=$(date +%s)
finish() {
  TASK_RC=$?
  trap - EXIT
  printf '{"exit_code":%s,"worker_seconds":%s,"gpu_seconds":0}\n' "$TASK_RC" "$(($(date +%s)-TASK_STARTED))" > "$TASK_OUT/EXIT.json"
}
trap finish EXIT
cd "$TASK_CODE"
"$TASK_PYTHON" -m pytest -q tests/test_rlinf_native_ppo.py \
  -k 'original_rollout or worker_rng or sampler_and_eval_rng or native_history_multirank or actor_sidecar_requires' > "$TASK_OUT/tests.log" 2>&1
timeout --signal=INT --kill-after=15s 1200 "$TASK_PYTHON" "$TASK_RUN_CODE/archive_qualification.py"
