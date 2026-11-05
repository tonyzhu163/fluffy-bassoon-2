#!/usr/bin/env bash
set -Eeuo pipefail
TASK_SOURCE=${1:?exact reviewed source}
TASK_OUT=${2:?fresh output}
TASK_GPUS=${3:-0}
[[ $TASK_GPUS == 0 || $TASK_GPUS == 1 ]]
TASK_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_RUN_CODE=$TASK_CODE/outputs/robotwin-sidney-rlinf-ppo-20261105-r1
TASK_PYTHON=/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python
[[ $(hostname) == job-* && $(git -C "$TASK_CODE" rev-parse HEAD) == "$TASK_SOURCE" ]]
[[ ! -e $TASK_OUT && -z $(git -C "$TASK_CODE" status --porcelain --untracked-files=no) ]]
mkdir -p "$TASK_OUT"
exec > "$TASK_OUT/worker.log" 2>&1
export TASK_SOURCE TASK_OUT TASK_GPUS
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
  TASK_ELAPSED=$(($(date +%s)-TASK_STARTED))
  printf '{"exit_code":%s,"worker_seconds":%s,"allocated_gpus":%s,"worker_gpu_seconds":%s}\n' "$TASK_RC" "$TASK_ELAPSED" "$TASK_GPUS" "$((TASK_GPUS*TASK_ELAPSED))" > "$TASK_OUT/EXIT.json"
}
trap finish EXIT
if [[ $TASK_GPUS == 1 ]]; then
  export FLUFFY_GPU_EXECUTION=baige
  if [[ ${CUDA_VISIBLE_DEVICES+x} != x ]]; then export CUDA_VISIBLE_DEVICES=0; fi
  "$TASK_PYTHON" "$TASK_CODE/scripts/[train]gpu_probe.py" --baige-container \
    --baige-exclusive-training --purpose training --expected-count 1 --output "$TASK_OUT/gpu-probe.json"
fi
cd "$TASK_CODE"
"$TASK_PYTHON" -m pytest -q tests/test_rlinf_native_ppo.py \
  -k 'original_rollout or worker_rng or sampler_and_eval_rng or native_history_multirank or actor_sidecar_requires' > "$TASK_OUT/tests.log" 2>&1
timeout --signal=INT --kill-after=15s 1200 "$TASK_PYTHON" "$TASK_RUN_CODE/archive_qualification.py"
