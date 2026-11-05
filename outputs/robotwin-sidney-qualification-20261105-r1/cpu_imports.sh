#!/usr/bin/env bash
set -Eeuo pipefail
TASK_SOURCE=${1:?exact reviewed commit}
TASK_OUT=${2:?fresh output}
TASK_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_LEROBOT=/mowenjie/fluffy-bassoon-2/sources/lerobot-sidney-92f96f33-20261105-r1
TASK_PYTHON=/mowenjie/fluffy-bassoon-2/envs/lerobot-sidney-92f96f33-20261105-r1/bin/python
[[ $(uname -s) == Linux && $(hostname) == job-* ]]
[[ $(git -C "$TASK_CODE" rev-parse HEAD) == "$TASK_SOURCE" ]]
[[ $(git -C "$TASK_LEROBOT" rev-parse HEAD) == 92f96f33b3a842ead3a115cc4f701ba2ad8aa3b5 ]]
[[ -z $(git -C "$TASK_CODE" status --porcelain --untracked-files=no) ]]
[[ ! -e $TASK_OUT ]]
mkdir -p "$TASK_OUT"
export PYTHONPATH=$TASK_LEROBOT/src
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
timeout --signal=TERM --kill-after=10s 240 "$TASK_PYTHON" \
  "$TASK_CODE/outputs/robotwin-sidney-qualification-20261105-r1/cpu_imports.py" \
  --output "$TASK_OUT/probe" 2>&1 | tee "$TASK_OUT/probe.log"
