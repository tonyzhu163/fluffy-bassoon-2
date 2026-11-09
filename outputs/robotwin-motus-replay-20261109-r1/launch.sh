#!/usr/bin/env bash
# Usage inside a 1-GPU Baige job, from a clean checkout at the reviewed commit:
#   bash outputs/robotwin-motus-replay-20261109-r1/launch.sh <commit> <attempt>
set -Eeuo pipefail
TASK_SOURCE=${1:?reviewed source commit}
TASK_ATTEMPT=${2:-01}
[[ $TASK_ATTEMPT =~ ^[0-9]{2}$ ]]
TASK_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_ID=replay-$TASK_ATTEMPT
TASK_LIMIT=2700
TASK_OUT=/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1/motus-replay-20261109-r1/$TASK_ID
TASK_LOCAL=/tmp/rt-motus-$TASK_ID
TASK_PYTHON=/mowenjie/fluffy-bassoon-2/envs/motus-reference-openpi-20261109-r2/bin/python
ARCHIVE=/mowenjie/fluffy-bassoon-2/inputs/robotwin-phone-clean50-3dc3b798-20261031-r1/aloha-agilex_clean_50.zip
MODEL=/mowenjie/fluffy-bassoon-2/cache/robotwin-ppo-pilot-20261105-r1-restore02/pi05-motus-effc6e1a-r1
export TASK_CODE TASK_SOURCE TASK_OUT TASK_LOCAL GIT_OPTIONAL_LOCKS=0
[[ $(uname -s) == Linux && $(hostname) == job-* ]]
[[ $(git -C "$TASK_CODE" rev-parse HEAD) == "$TASK_SOURCE" ]]
[[ -z $(git -C "$TASK_CODE" status --porcelain --untracked-files=no) ]]
[[ ! -e $TASK_OUT && ! -e $TASK_LOCAL ]]
[[ -f $ARCHIVE && -d $MODEL ]]
mkdir -p "$(dirname "$TASK_OUT")"
mkdir "$TASK_OUT" "$TASK_LOCAL"
exec > "$TASK_OUT/worker.log" 2>&1
TASK_STARTED=$(date +%s)
finish() {
 TASK_RC=$?
 trap - EXIT
 TASK_ENDED=$(date +%s)
 printf '{"exit_code":%s,"started_unix":%s,"ended_unix":%s,"worker_gpu_seconds":%s}\n' \
  "$TASK_RC" "$TASK_STARTED" "$TASK_ENDED" "$((TASK_ENDED-TASK_STARTED))" > "$TASK_OUT/EXIT.json"
}
trap finish EXIT
export PYTHONPATH=$TASK_CODE/src
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export FLUFFY_GPU_EXECUTION=baige JAX_PLATFORMS=cpu PYTORCH_ALLOC_CONF=expandable_segments:True
if [[ ${CUDA_VISIBLE_DEVICES+x} != x ]]; then export CUDA_VISIBLE_DEVICES=0; fi
"$TASK_PYTHON" "$TASK_CODE/scripts/[train]gpu_probe.py" --baige-container \
 --purpose evaluation --expected-count 1 --output "$TASK_OUT/gpu-probe.json"
export XDG_CACHE_HOME=$TASK_LOCAL/cache OPENPI_DATA_HOME=$TASK_LOCAL/openpi
export HF_HOME=$TASK_LOCAL/hf TORCH_HOME=$TASK_LOCAL/torch TMPDIR=$TASK_LOCAL/tmp
mkdir -p "$XDG_CACHE_HOME" "$OPENPI_DATA_HOME/big_vision" "$HF_HOME" "$TORCH_HOME" "$TMPDIR"
cp /mowenjie/fluffy-bassoon-2/cache/dsrl-pi0-task57-20261101-prep-r7/big_vision/paligemma_tokenizer.model \
 "$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model"
SCRIPT=$TASK_CODE/outputs/robotwin-motus-replay-20261109-r1/replay.py
"$TASK_PYTHON" "$SCRIPT" --archive "$ARCHIVE" --model "$MODEL" --output "$TASK_OUT" --check-only
timeout --signal=INT --kill-after=20s $((TASK_LIMIT-120)) "$TASK_PYTHON" "$SCRIPT" \
 --archive "$ARCHIVE" --model "$MODEL" --output "$TASK_OUT" --episodes 20 --stride 4 \
 > "$TASK_OUT/replay.log" 2>&1
