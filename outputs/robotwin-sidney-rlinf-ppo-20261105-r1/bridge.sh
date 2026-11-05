#!/usr/bin/env bash
set -Eeuo pipefail
TASK_SOURCE=${1:?reviewed exact source}
TASK_OUT=${2:?fresh attempt output}
TASK_CACHE=${3:?fresh converted model directory}
TASK_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_RUN_CODE=$TASK_CODE/outputs/robotwin-sidney-rlinf-ppo-20261105-r1
TASK_PYTHON=/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python
[[ $(uname -s) == Linux && $(hostname) == job-* ]]
[[ $(git -C "$TASK_CODE" rev-parse HEAD) == "$TASK_SOURCE" ]]
[[ -z $(git -C "$TASK_CODE" status --porcelain --untracked-files=no) ]]
[[ ! -e $TASK_OUT && ! -e $TASK_CACHE ]]
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
"$TASK_PYTHON" -m pytest -q tests/test_robotwin_noise.py \
  --junitxml="$TASK_OUT/core-tests.xml" > "$TASK_OUT/core-tests.log" 2>&1
"$TASK_PYTHON" "$TASK_RUN_CODE/convert.py" \
  --source /mowenjie/fluffy-bassoon-2/cache/robotwin-sidney-qualification-20261105-r1 \
  --output "$TASK_CACHE" --receipt "$TASK_OUT/CONVERSION.json" > "$TASK_OUT/conversion.log" 2>&1
"$TASK_PYTHON" "$TASK_RUN_CODE/scene_vectors.py" --output "$TASK_OUT/vectors" > "$TASK_OUT/scenes.log" 2>&1
"$TASK_PYTHON" "$TASK_RUN_CODE/native_replay.py" --vectors "$TASK_OUT/vectors" \
  --output "$TASK_OUT/native-reference" > "$TASK_OUT/native-replay.log" 2>&1
timeout --signal=TERM --kill-after=15s 900 "$TASK_PYTHON" "$TASK_RUN_CODE/qualify_rlinf.py" \
  --vectors "$TASK_OUT/vectors" --reference "$TASK_OUT/native-reference" \
  --model "$TASK_CACHE" --output "$TASK_OUT/rlinf" > "$TASK_OUT/rlinf.log" 2>&1
