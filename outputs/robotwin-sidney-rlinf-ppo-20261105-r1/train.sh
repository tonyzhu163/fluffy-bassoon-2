#!/usr/bin/env bash
set -Eeuo pipefail
TASK_SOURCE=${1:?reviewed exact source}
TASK_OUT=${2:?fresh training output}
TASK_BRIDGE=${3:?completed bridge output}
TASK_PRIOR_GPU_SECONDS=${4:?bridge scheduler charge}
TASK_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_RUN_CODE=$TASK_CODE/outputs/robotwin-sidney-rlinf-ppo-20261105-r1
TASK_PYTHON=/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python
[[ $(uname -s) == Linux && $(hostname) == job-* ]]
[[ $(git -C "$TASK_CODE" rev-parse HEAD) == "$TASK_SOURCE" ]]
[[ -z $(git -C "$TASK_CODE" status --porcelain --untracked-files=no) ]]
[[ ! -e $TASK_OUT && $TASK_PRIOR_GPU_SECONDS =~ ^[0-9]+$ ]]
# Four additional GPU-hours for the bridge and this initial allocation.
# Eight GPUs,25 minutes plus60 seconds/GPU scheduler overhead.
[[ $((TASK_PRIOR_GPU_SECONDS+8*1560)) -le 14400 ]]
mkdir -p "$TASK_OUT"
export TASK_SOURCE TASK_BRIDGE
export REPO_PATH=/mowenjie/fluffy-bassoon-2/sources/robotwin-baseline-smoke-20261028-r1/RLinf
export ROBOTWIN_PATH=/mowenjie/fluffy-bassoon-2/sources/robotwin-baseline-smoke-20261028-r1/RoboTwin
export ROBOTWIN_ASSETS_PATH=$ROBOTWIN_PATH ASSETS_PATH=$ROBOTWIN_PATH
export EMBODIED_PATH=$REPO_PATH/examples/embodiment
export PYTHONPATH=$TASK_CODE/src:$REPO_PATH:$ROBOTWIN_PATH
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export ROBOT_PLATFORM=ALOHA MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export FLUFFY_GPU_EXECUTION=baige NCCL_P2P_DISABLE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ ${CUDA_VISIBLE_DEVICES+x} != x ]]; then export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; fi
export NATIVE_PPO_WORLD=8 NATIVE_PPO_PLACEMENT=0-7 NATIVE_PPO_MODE=train
export NATIVE_PPO_OUTPUT=$TASK_OUT NATIVE_PPO_ACCOUNTING_ROOT=$TASK_OUT/action-accounting
export NATIVE_PPO_ATTEMPT=attempt-0001
export NATIVE_PPO_MODEL=/mowenjie/fluffy-bassoon-2/cache/robotwin-sidney-rlinf-20261105-r1
export NATIVE_PPO_INITIALIZER_REVISION=e49e2ab6c11f07511573b67261bd129e88d0a416
TASK_STARTED=$(date +%s)
export NATIVE_PPO_DEADLINE=$((TASK_STARTED+1440))
finish() {
  TASK_RC=$?
  trap - EXIT
  TASK_ENDED=$(date +%s)
  printf '{"exit_code":%s,"started_unix":%s,"ended_unix":%s,"worker_gpu_seconds":%s,"prior_bridge_gpu_seconds":%s}\n' \
    "$TASK_RC" "$TASK_STARTED" "$TASK_ENDED" "$((8*(TASK_ENDED-TASK_STARTED)))" "$TASK_PRIOR_GPU_SECONDS" > "$TASK_OUT/EXIT.json"
}
trap finish EXIT
"$TASK_PYTHON" "$TASK_CODE/scripts/[train]gpu_probe.py" --baige-container \
  --baige-exclusive-training --purpose training --expected-count 8 --output "$TASK_OUT/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP=$TASK_OUT/gpu-probe.json
[[ $(git -C "$REPO_PATH" rev-parse HEAD) == c605b9d55ed93a180119b099e6f836abff293b08 ]]
[[ $(git -C "$ROBOTWIN_PATH" rev-parse HEAD) == dca9ec682688821c42944eacd7f5dd9bfb15396f ]]
[[ -z $(git -C "$REPO_PATH" status --porcelain --untracked-files=no) ]]
[[ -z $(git -C "$ROBOTWIN_PATH" status --porcelain --untracked-files=no) ]]
mkdir -p /root/.sapien/physx
if [[ ! -e /root/.sapien/physx/105.1-physx-5.3.1.patch0 ]]; then
  ln -s /mowenjie/fluffy-bassoon/inputs/maniskill-scene-assets-20261028-r1/physx/105.1-physx-5.3.1.patch0 /root/.sapien/physx/105.1-physx-5.3.1.patch0
fi
TASK_MICRO=$("$TASK_PYTHON" - <<'PY'
import hashlib,json,os
from pathlib import Path
bridge=Path(os.environ['TASK_BRIDGE'])
result=json.loads((bridge/'rlinf/RESULT.json').read_text())
assert result['status']=='complete' and result['optimizer_updates']==0
assert result['episode']['complete'] and result['episode']['task']=='place_phone_stand'
assert result['execution_microbatch'] in (8,16,32)
receipt=json.loads((bridge/'REUSED-INPUTS.json').read_text())['conversion']
cache=Path(os.environ['NATIVE_PPO_MODEL'])
for filename,key in [('model.safetensors','model_sha256'),('sidney-processors.json','processors_sha256')]:
    h=hashlib.sha256()
    with (cache/filename).open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024**2),b''):h.update(block)
    assert h.hexdigest()==receipt[key]
Path(os.environ['NATIVE_PPO_OUTPUT'],'bridge-admission.json').write_text(json.dumps(result,indent=2)+'\n')
print(result['execution_microbatch'])
PY
)
cd "$TASK_CODE"
# Compose/validate the actual source configuration inside the allocation before
# model initialization. It keeps global2048 and only tunes execution microbatch.
"$TASK_PYTHON" "$TASK_RUN_CODE/validate_config.py" "$TASK_CODE" "$TASK_MICRO"
timeout --signal=INT --kill-after=20s 1380 "$TASK_PYTHON" "$TASK_RUN_CODE/train_entry.py" \
  --config-path "$TASK_CODE/configs/native_ppo" --config-name sidney_phone_two \
  "actor.micro_batch_size=$TASK_MICRO" > "$TASK_OUT/train.log" 2>&1
