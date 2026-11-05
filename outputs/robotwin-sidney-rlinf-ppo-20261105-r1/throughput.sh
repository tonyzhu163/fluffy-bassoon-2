#!/usr/bin/env bash
set -Eeuo pipefail
TASK_SOURCE=${1:?source}
TASK_OUT=${2:?fresh benchmark output}
TASK_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_RUN_CODE=$TASK_CODE/outputs/robotwin-sidney-rlinf-ppo-20261105-r1
TASK_PYTHON=/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python
[[ $(hostname) == job-* && $(git -C "$TASK_CODE" rev-parse HEAD) == "$TASK_SOURCE" ]]
[[ -z $(git -C "$TASK_CODE" status --porcelain --untracked-files=no) && ! -e $TASK_OUT ]]
mkdir -p "$TASK_OUT"
export TASK_CODE TASK_SOURCE
export REPO_PATH=/mowenjie/fluffy-bassoon-2/sources/robotwin-baseline-smoke-20261028-r1/RLinf
export ROBOTWIN_ASSETS_PATH=/mowenjie/fluffy-bassoon-2/sources/robotwin-baseline-smoke-20261028-r1/RoboTwin
export EMBODIED_PATH=$REPO_PATH/examples/embodiment
export PYTHONPATH=$TASK_RUN_CODE:$TASK_CODE/src:$REPO_PATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
if [[ ${CUDA_VISIBLE_DEVICES+x} != x ]]; then export CUDA_VISIBLE_DEVICES=0,1,2,3; fi
export NATIVE_PPO_WORLD=4 NATIVE_PPO_PLACEMENT=0-3 NATIVE_PPO_MODE=train
export NATIVE_PPO_MODEL=/mowenjie/fluffy-bassoon-2/cache/robotwin-sidney-rlinf-20261105-r1
export NATIVE_PPO_INITIALIZER_REVISION=e49e2ab6c11f07511573b67261bd129e88d0a416
export NATIVE_PPO_DEADLINE=$(($(date +%s)+1500))
"$TASK_PYTHON" "$TASK_CODE/scripts/[train]gpu_probe.py" --baige-container \
  --baige-exclusive-training --purpose training --expected-count 4 --output "$TASK_OUT/gpu-probe.json" > "$TASK_OUT/probe.log" 2>&1
export LYCO_BAIGE_GPU_MAP=$TASK_OUT/gpu-probe.json
if [[ ${TASK_SINGLE_GROUP_SWEEP:-0} == 1 ]]; then
  export NATIVE_PPO_OUTPUT=$TASK_OUT NATIVE_PPO_ACCOUNTING_ROOT=$TASK_OUT/not-used
  export PYTORCH_ALLOC_CONF=expandable_segments:True
  set +e
  timeout --signal=INT --kill-after=15s 1200 "$TASK_PYTHON" "$TASK_RUN_CODE/throughput.py" \
    actor.micro_batch_size=64 > "$TASK_OUT/console.log" 2>&1
  TASK_RC=$?
  set -e
  printf '{"exit_code":%s}\n' "$TASK_RC" > "$TASK_OUT/EXIT.json"
  "$TASK_PYTHON" -m ray.scripts.scripts stop --force > "$TASK_OUT/ray-stop.log" 2>&1 || true
  exit "$TASK_RC"
fi
for TASK_MICRO in 64 128 256 512; do
  export NATIVE_PPO_OUTPUT=$TASK_OUT/micro-$TASK_MICRO
  export NATIVE_PPO_ACCOUNTING_ROOT=$NATIVE_PPO_OUTPUT/not-used
  mkdir -p "$NATIVE_PPO_OUTPUT"
  set +e
  timeout --signal=INT --kill-after=15s 300 "$TASK_PYTHON" "$TASK_RUN_CODE/throughput.py" \
    "actor.micro_batch_size=$TASK_MICRO" > "$NATIVE_PPO_OUTPUT/console.log" 2>&1
  TASK_RC=$?
  set -e
  printf '{"exit_code":%s}\n' "$TASK_RC" > "$NATIVE_PPO_OUTPUT/EXIT.json"
  # This exclusive scheduler container contains only this benchmark's local
  # Ray cluster; clear it between candidates, including collective OOM cases.
  "$TASK_PYTHON" -m ray.scripts.scripts stop --force > "$NATIVE_PPO_OUTPUT/ray-stop.log" 2>&1 || true
  if [[ $TASK_RC != 0 ]] && ! grep -q -E 'OutOfMemoryError|CUDA out of memory' "$NATIVE_PPO_OUTPUT/console.log"; then
    exit "$TASK_RC"
  fi
done
