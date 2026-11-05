#!/usr/bin/env bash
set -Eeuo pipefail
TASK_SOURCE=${1:?exact reviewed project commit}
TASK_OUT=${2:?fresh qualification output}
TASK_TASK=${3:-place_phone_stand}
TASK_EPISODES=${4:-1}
TASK_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_RUN_CODE=$TASK_CODE/outputs/robotwin-sidney-qualification-20261105-r1
TASK_PYTHON=/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python
[[ $(uname -s) == Linux && $(hostname) == job-* ]]
[[ $(git -C "$TASK_CODE" rev-parse HEAD) == "$TASK_SOURCE" ]]
[[ -z $(git -C "$TASK_CODE" status --porcelain --untracked-files=no) ]]
[[ ! -e $TASK_OUT ]]
mkdir -p "$TASK_OUT"
export TASK_SOURCE
export REPO_PATH=/mowenjie/fluffy-bassoon-2/sources/robotwin-baseline-smoke-20261028-r1/RLinf
export ROBOTWIN_PATH=/mowenjie/fluffy-bassoon-2/sources/robotwin-baseline-smoke-20261028-r1/RoboTwin
export ASSETS_PATH=$ROBOTWIN_PATH
export EMBODIED_PATH=$REPO_PATH/examples/embodiment
export PYTHONPATH=$TASK_CODE/src:$REPO_PATH:$ROBOTWIN_PATH
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export ROBOT_PLATFORM=ALOHA MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export FLUFFY_GPU_EXECUTION=baige NCCL_P2P_DISABLE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# Baige exposes its allocation through NVIDIA_VISIBLE_DEVICES but can leave
# CUDA_VISIBLE_DEVICES unset. Preserve an explicit scheduler value; otherwise
# select ordinal zero inside this one-GPU container. The unchanged probe still
# requires the scheduler UUID, CUDA PCI mapping and own-process tensor check.
if [[ ${CUDA_VISIBLE_DEVICES+x} != x ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi
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
# Match the existing qualified RoboTwin launcher in a fresh container.
mkdir -p /root/.sapien/physx
if [[ ! -e /root/.sapien/physx/105.1-physx-5.3.1.patch0 ]]; then
  ln -s /mowenjie/fluffy-bassoon/inputs/maniskill-scene-assets-20261028-r1/physx/105.1-physx-5.3.1.patch0 \
    /root/.sapien/physx/105.1-physx-5.3.1.patch0
fi
TASK_MODEL_PYTHON=/mowenjie/fluffy-bassoon-2/envs/lerobot-sidney-92f96f33-20261105-r2/bin/python
PYTHONPATH=/mowenjie/fluffy-bassoon-2/sources/lerobot-sidney-92f96f33-20261105-r1/src \
  timeout --signal=TERM --kill-after=10s 240 \
  "$TASK_MODEL_PYTHON" "$TASK_RUN_CODE/cpu_imports.py" --expected-visible-gpus 1 \
  --output "$TASK_OUT/cpu-imports" > "$TASK_OUT/cpu-imports.log" 2>&1
timeout --signal=TERM --kill-after=15s 1600 "$TASK_PYTHON" "$TASK_RUN_CODE/qualify.py" \
  --output "$TASK_OUT/native" --task "$TASK_TASK" --episodes "$TASK_EPISODES" \
  2>&1 | tee "$TASK_OUT/qualification.log"
