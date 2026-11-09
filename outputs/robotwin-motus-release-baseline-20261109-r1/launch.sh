#!/usr/bin/env bash
set -Eeuo pipefail
TASK_SOURCE=${1:?reviewed source commit}
TASK_EPISODES=${2:-2}
TASK_ATTEMPT=${3:-01}
TASK_MODE=${4:-policy}
[[ $TASK_EPISODES == 2 || $TASK_EPISODES == 100 || ( $TASK_EPISODES == 16 && $TASK_MODE == gap ) ]]
[[ $TASK_ATTEMPT =~ ^[0-9]{2}$ ]]
[[ $TASK_MODE == policy || $TASK_MODE == render || $TASK_MODE == sidney || $TASK_MODE == gap || $TASK_MODE == wrist || $TASK_MODE == wrist-random || $TASK_MODE == ppo40 || $TASK_MODE == grpo40 ]]
TASK_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_ID=native-smoke-$TASK_ATTEMPT
TASK_LIMIT=3300
if [[ $TASK_EPISODES == 100 ]]; then TASK_ID=native-phone100-$TASK_ATTEMPT; TASK_LIMIT=21600; fi
TASK_EXTRA=()
if [[ $TASK_MODE == render ]]; then
 [[ $TASK_EPISODES == 2 ]]
 TASK_ID=native-render-$TASK_ATTEMPT; TASK_LIMIT=840; TASK_EXTRA=(--render-only)
fi
TASK_OUT=/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1/motus-release-baseline-20261109-r1/$TASK_ID
TASK_LOCAL=/tmp/rt-motus-release-$TASK_ID
TASK_EVALUATOR=$TASK_CODE/outputs/robotwin-motus-release-baseline-20261109-r1/evaluate.py
if [[ $TASK_MODE == sidney || $TASK_MODE == gap || $TASK_MODE == wrist || $TASK_MODE == wrist-random || $TASK_MODE == ppo40 || $TASK_MODE == grpo40 ]]; then
 TASK_EVALUATOR=$TASK_CODE/outputs/robotwin-sidney-native-sft-20261109-r1/run.py
 TASK_OUT=/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1/sidney-native-sft-20261109-r1/$TASK_ID
 TASK_LOCAL=/tmp/rt-sidney-sft-$TASK_ID
 export CUBLAS_WORKSPACE_CONFIG=:4096:8
fi
TASK_PYTHON=/mowenjie/fluffy-bassoon-2/envs/motus-reference-openpi-20261109-r2/bin/python
if [[ $TASK_MODE == gap ]]; then
 [[ $TASK_EPISODES == 16 ]]
 TASK_OUT=/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1/sidney-gap-20261109-r1/diagnostic-$TASK_ATTEMPT
 TASK_LOCAL=/tmp/rt-sidney-gap-$TASK_ATTEMPT
 TASK_EXTRA=(--diagnostic)
 TASK_LIMIT=14400
fi
if [[ $TASK_MODE == wrist || $TASK_MODE == wrist-random ]]; then
 [[ $TASK_EPISODES == 100 ]]
 TASK_OUT=/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1/sidney-gap-20261109-r1/wrist-$TASK_ATTEMPT
 TASK_LOCAL=/tmp/rt-sidney-wrist-$TASK_ATTEMPT
 TASK_EXTRA=(--wrist-near 0.01)
 TASK_LIMIT=10800
 if [[ $TASK_MODE == wrist-random ]]; then
  TASK_OUT=${TASK_OUT%/*}/wrist-random-$TASK_ATTEMPT
  TASK_LOCAL=/tmp/rt-sidney-wrist-random-$TASK_ATTEMPT
  TASK_EXTRA+=(--wrist-domain demo_randomized)
 fi
fi
if [[ $TASK_MODE == ppo40 || $TASK_MODE == grpo40 ]]; then
 [[ $TASK_EPISODES == 100 ]]
 TASK_OUT=/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1/sidney-native-endpoints-20261109-r1/$TASK_MODE-$TASK_ATTEMPT
 TASK_LOCAL=/tmp/rt-sidney-$TASK_MODE-$TASK_ATTEMPT
 TASK_EXTRA=(--wrist-near 0.01 --wrist-domain both --endpoint "$TASK_MODE")
fi
export TASK_CODE TASK_SOURCE TASK_OUT TASK_LOCAL TASK_LIMIT GIT_OPTIONAL_LOCKS=0
[[ $(uname -s) == Linux && $(hostname) == job-* ]]
[[ $(git -C "$TASK_CODE" rev-parse HEAD) == "$TASK_SOURCE" ]]
[[ -z $(git -C "$TASK_CODE" status --porcelain --untracked-files=no) ]]
[[ ! -e $TASK_OUT && ! -e $TASK_LOCAL ]]
[[ -d /mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1 ]]
mkdir -p "$(dirname "$TASK_OUT")"
mkdir "$TASK_OUT" "$TASK_LOCAL"
exec > "$TASK_OUT/worker.log" 2>&1
TASK_STARTED=$(date +%s)
export TASK_DEADLINE=$((TASK_STARTED+TASK_LIMIT))
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
export FLUFFY_GPU_EXECUTION=baige NCCL_P2P_DISABLE=1 JAX_PLATFORMS=cpu
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTORCH_ALLOC_CONF=expandable_segments:True
if [[ ${CUDA_VISIBLE_DEVICES+x} != x ]]; then export CUDA_VISIBLE_DEVICES=0; fi
"$TASK_PYTHON" "$TASK_CODE/scripts/[train]gpu_probe.py" --baige-container \
 --purpose evaluation --expected-count 1 --output "$TASK_OUT/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP=$TASK_OUT/gpu-probe.json
export NATIVE_SOURCE=/mowenjie/fluffy-bassoon-2/sources/robotwin-motus-native-fa8d97b0-20261109-r1
export NATIVE_PPO_MODEL=/mowenjie/fluffy-bassoon-2/cache/robotwin-ppo-pilot-20261105-r1-restore02/pi05-motus-effc6e1a-r1
"$TASK_PYTHON" - <<'PY'
import json,os,shutil
from pathlib import Path
probe=json.loads((Path(os.environ['TASK_OUT'])/'gpu-probe.json').read_text())
assert probe['status']=='passed' and len(probe['devices'])==1
assert probe['devices'][0]['cuda_total_bytes']>=90*2**30
assert shutil.disk_usage(os.environ['TASK_LOCAL']).free>=32*2**30
assert os.path.ismount('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1') or os.path.ismount('/mnt/bos/adgfyl')
path=Path(os.environ['TASK_OUT'])/'write-probe'
with path.open('xb') as f:f.write(b'native-release-evaluation');f.flush();os.fsync(f.fileno())
assert path.read_bytes()==b'native-release-evaluation';path.unlink()
PY
export XDG_CACHE_HOME=$TASK_LOCAL/cache OPENPI_DATA_HOME=$TASK_LOCAL/openpi
export HF_HOME=$TASK_LOCAL/hf TORCH_HOME=$TASK_LOCAL/torch TMPDIR=$TASK_LOCAL/tmp
mkdir -p "$XDG_CACHE_HOME" "$OPENPI_DATA_HOME/big_vision" "$HF_HOME" "$TORCH_HOME" "$TMPDIR" /root/.sapien/physx
cp /mowenjie/fluffy-bassoon-2/cache/dsrl-pi0-task57-20261101-prep-r7/big_vision/paligemma_tokenizer.model \
 "$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model"
if [[ ! -e /root/.sapien/physx/105.1-physx-5.3.1.patch0 ]]; then
 ln -s /mowenjie/fluffy-bassoon/inputs/maniskill-scene-assets-20261028-r1/physx/105.1-physx-5.3.1.patch0 \
 /root/.sapien/physx/105.1-physx-5.3.1.patch0
fi
# Copy exact pinned native source into job-local NVMe; use existing shared
# assets without modifying either their files or the upstream checkout.
git -C "$NATIVE_SOURCE" archive HEAD envs task_config description script | tar -x -C "$TASK_LOCAL"
"$TASK_PYTHON" "$TASK_CODE/outputs/robotwin-motus-release-baseline-20261109-r1/prepare_assets.py" \
 "$TASK_LOCAL/assets" "$TASK_OUT/ASSET-PATHS.json"
if [[ -f "$NATIVE_SOURCE/Times New Roman.ttf" ]]; then
 ln -s "$NATIVE_SOURCE/Times New Roman.ttf" "$TASK_LOCAL/Times New Roman.ttf"
fi
export PYTHONPATH=$TASK_LOCAL:$TASK_LOCAL/description/utils:$TASK_LOCAL/script:$PYTHONPATH
cd "$TASK_LOCAL"
TASK_REMAINING=$((TASK_DEADLINE-$(date +%s)-60))
[[ $TASK_REMAINING -gt 300 ]]
timeout --signal=INT --kill-after=20s "$TASK_REMAINING" "$TASK_PYTHON" \
 "$TASK_EVALUATOR" \
 --output "$TASK_OUT" --episodes "$TASK_EPISODES" "${TASK_EXTRA[@]}" > "$TASK_OUT/evaluation.log" 2>&1
if grep -Eq 'OIDN Error|unsupported device type|invalid handle' "$TASK_OUT/evaluation.log"; then
 echo 'Native denoiser qualification failed' >&2
 exit 1
fi
