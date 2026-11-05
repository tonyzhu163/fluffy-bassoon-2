#!/usr/bin/env bash
set -Eeuo pipefail
TASK_SOURCE=${1:?reviewed exact source}
TASK_OUT=${2:?fresh attempt output}
TASK_CACHE=${3:?fresh converted model directory}
TASK_PRIOR=${4:-}
TASK_FP32=${5:-}
TASK_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_RUN_CODE=$TASK_CODE/outputs/robotwin-sidney-rlinf-ppo-20261105-r1
TASK_PYTHON=/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python
[[ $(uname -s) == Linux && $(hostname) == job-* ]]
[[ $(git -C "$TASK_CODE" rev-parse HEAD) == "$TASK_SOURCE" ]]
[[ -z $(git -C "$TASK_CODE" status --porcelain --untracked-files=no) ]]
[[ ! -e $TASK_OUT ]]
if [[ -z $TASK_PRIOR ]]; then [[ ! -e $TASK_CACHE ]]; fi
mkdir -p "$TASK_OUT"
export TASK_SOURCE
export REPO_PATH=/mowenjie/fluffy-bassoon-2/sources/robotwin-baseline-smoke-20261028-r1/RLinf
export ROBOTWIN_PATH=/mowenjie/fluffy-bassoon-2/sources/robotwin-baseline-smoke-20261028-r1/RoboTwin
export ASSETS_PATH=$ROBOTWIN_PATH EMBODIED_PATH=$REPO_PATH/examples/embodiment
export PYTHONPATH=$TASK_CODE/src:$REPO_PATH:$ROBOTWIN_PATH
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export ROBOT_PLATFORM=ALOHA MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export FLUFFY_GPU_EXECUTION=baige NCCL_P2P_DISABLE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ ${CUDA_VISIBLE_DEVICES+x} != x ]]; then export CUDA_VISIBLE_DEVICES=0; fi
TASK_STARTED=$(date +%s)
finish() {
  TASK_RC=$?
  trap - EXIT
  TASK_ENDED=$(date +%s)
  printf '{"exit_code":%s,"started_unix":%s,"ended_unix":%s,"worker_gpu_seconds":%s}\n' \
    "$TASK_RC" "$TASK_STARTED" "$TASK_ENDED" "$((TASK_ENDED-TASK_STARTED))" > "$TASK_OUT/EXIT.json"
}
trap finish EXIT
"$TASK_PYTHON" "$TASK_CODE/scripts/[train]gpu_probe.py" --baige-container \
  --baige-exclusive-training --purpose training --expected-count 1 --output "$TASK_OUT/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP=$TASK_OUT/gpu-probe.json
[[ $(git -C "$REPO_PATH" rev-parse HEAD) == c605b9d55ed93a180119b099e6f836abff293b08 ]]
[[ $(git -C "$ROBOTWIN_PATH" rev-parse HEAD) == dca9ec682688821c42944eacd7f5dd9bfb15396f ]]
[[ -z $(git -C "$REPO_PATH" status --porcelain --untracked-files=no) ]]
[[ -z $(git -C "$ROBOTWIN_PATH" status --porcelain --untracked-files=no) ]]
mkdir -p /root/.sapien/physx
if [[ ! -e /root/.sapien/physx/105.1-physx-5.3.1.patch0 ]]; then
  ln -s /mowenjie/fluffy-bassoon/inputs/maniskill-scene-assets-20261028-r1/physx/105.1-physx-5.3.1.patch0 /root/.sapien/physx/105.1-physx-5.3.1.patch0
fi
cd "$TASK_CODE"
if [[ -z $TASK_PRIOR ]]; then
  "$TASK_PYTHON" -m pytest -q tests/test_robotwin_noise.py \
    --junitxml="$TASK_OUT/core-tests.xml" > "$TASK_OUT/core-tests.log" 2>&1
  "$TASK_PYTHON" "$TASK_RUN_CODE/convert.py" \
    --source /mowenjie/fluffy-bassoon-2/cache/robotwin-sidney-qualification-20261105-r1 \
    --output "$TASK_CACHE" --receipt "$TASK_OUT/CONVERSION.json" > "$TASK_OUT/conversion.log" 2>&1
  "$TASK_PYTHON" "$TASK_RUN_CODE/scene_vectors.py" --output "$TASK_OUT/vectors" > "$TASK_OUT/scenes.log" 2>&1
  "$TASK_PYTHON" "$TASK_RUN_CODE/native_replay.py" --vectors "$TASK_OUT/vectors" \
    --output "$TASK_OUT/native-reference" > "$TASK_OUT/native-replay.log" 2>&1
  TASK_INPUTS=$TASK_OUT
else
  "$TASK_PYTHON" - "$TASK_PRIOR" "$TASK_CACHE" "$TASK_OUT" <<'PY'
import hashlib,json,sys
from pathlib import Path
prior,cache,out=map(Path,sys.argv[1:])
assert json.loads((prior/'EXIT.json').read_text())['exit_code']==1
receipt=json.loads((prior/'CONVERSION.json').read_text())
assert str(cache)==receipt['output']
for filename,key in [('model.safetensors','model_sha256'),('sidney-processors.json','processors_sha256')]:
    digest=hashlib.sha256()
    with (cache/filename).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):digest.update(block)
    assert digest.hexdigest()==receipt[key]
assert len(list((prior/'native-reference').glob('*-actions.npz')))==3
(out/'REUSED-INPUTS.json').write_text(json.dumps(dict(prior=str(prior),conversion=receipt),indent=2)+'\n')
PY
  TASK_INPUTS=$TASK_PRIOR
fi
TASK_EXTRA=()
if [[ -n $TASK_FP32 ]]; then TASK_EXTRA+=(--fp32-reference "$TASK_FP32"); fi
timeout --signal=TERM --kill-after=15s 900 "$TASK_PYTHON" "$TASK_RUN_CODE/qualify_rlinf.py" \
  --vectors "$TASK_INPUTS/vectors" --reference "$TASK_INPUTS/native-reference" \
  --model "$TASK_CACHE" --output "$TASK_OUT/rlinf" "${TASK_EXTRA[@]}" > "$TASK_OUT/rlinf.log" 2>&1
export ROBOTWIN_ASSETS_PATH=$ROBOTWIN_PATH
export NATIVE_PPO_WORLD=8 NATIVE_PPO_PLACEMENT=0-7 NATIVE_PPO_MODE=train
export NATIVE_PPO_OUTPUT=$TASK_OUT NATIVE_PPO_ACCOUNTING_ROOT=$TASK_OUT/action-accounting
export NATIVE_PPO_MODEL=$TASK_CACHE NATIVE_PPO_DEADLINE=$((TASK_STARTED+1800))
export NATIVE_PPO_INITIALIZER_REVISION=e49e2ab6c11f07511573b67261bd129e88d0a416
TASK_MICRO=$("$TASK_PYTHON" -c 'import json,sys;print(json.load(open(sys.argv[1]))["execution_microbatch"])' "$TASK_OUT/rlinf/RESULT.json")
"$TASK_PYTHON" "$TASK_RUN_CODE/validate_config.py" "$TASK_CODE" "$TASK_MICRO" > "$TASK_OUT/config-validation.log" 2>&1
