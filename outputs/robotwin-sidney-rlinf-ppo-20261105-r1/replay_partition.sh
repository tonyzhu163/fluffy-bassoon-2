#!/usr/bin/env bash
set -Eeuo pipefail
TASK_SOURCE=${1:?exact source}
TASK_OUT=${2:?fresh output}
TASK_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_PYTHON=/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python
[[ $(uname -s) == Linux && $(hostname) == job-* ]]
[[ $(git -C "$TASK_CODE" rev-parse HEAD) == "$TASK_SOURCE" ]]
[[ -z $(git -C "$TASK_CODE" status --porcelain --untracked-files=no) && ! -e $TASK_OUT ]]
mkdir -p "$TASK_OUT"
export TASK_SOURCE TASK_OUT
export REPO_PATH=/mowenjie/fluffy-bassoon-2/sources/robotwin-baseline-smoke-20261028-r1/RLinf
export PYTHONPATH=$TASK_CODE/src:$REPO_PATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
if [[ ${CUDA_VISIBLE_DEVICES+x} != x ]]; then export CUDA_VISIBLE_DEVICES=0; fi
TASK_STARTED=$(date +%s)
finish() {
  TASK_RC=$?;trap - EXIT;TASK_ENDED=$(date +%s)
  printf '{"exit_code":%s,"worker_gpu_seconds":%s}\n' "$TASK_RC" "$((TASK_ENDED-TASK_STARTED))" > "$TASK_OUT/EXIT.json"
}
trap finish EXIT
"$TASK_PYTHON" "$TASK_CODE/scripts/[train]gpu_probe.py" --baige-container \
  --baige-exclusive-training --purpose training --expected-count 1 --output "$TASK_OUT/gpu-probe.json" > "$TASK_OUT/probe.log" 2>&1
timeout --signal=TERM --kill-after=10s 240 "$TASK_PYTHON" "$(dirname "${BASH_SOURCE[0]}")/replay_partition.py" > "$TASK_OUT/replay.log" 2>&1
