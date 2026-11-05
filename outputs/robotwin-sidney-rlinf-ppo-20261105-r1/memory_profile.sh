#!/usr/bin/env bash
set -Eeuo pipefail
TASK_SOURCE=${1:?source};TASK_OUT=${2:?fresh output}
TASK_SCENES=${3:-16};[[ $TASK_SCENES == 16 || $TASK_SCENES == 64 ]]
TASK_PROFILE_MODE=${4:-full};[[ $TASK_PROFILE_MODE == full || $TASK_PROFILE_MODE == paired ]]
TASK_ACCOUNTING_RANK=${5:-0};[[ $TASK_ACCOUNTING_RANK =~ ^[0-3]$ ]]
TASK_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_PYTHON=/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python
[[ $(hostname) == job-* && $(git -C "$TASK_CODE" rev-parse HEAD) == "$TASK_SOURCE" && ! -e $TASK_OUT ]]
mkdir -p "$TASK_OUT"
exec > "$TASK_OUT/worker.log" 2>&1
export TASK_OUT TASK_SOURCE TASK_SCENES TASK_PROFILE_MODE TASK_ACCOUNTING_RANK
export REPO_PATH=/mowenjie/fluffy-bassoon-2/sources/robotwin-baseline-smoke-20261028-r1/RLinf
export ROBOTWIN_PATH=/mowenjie/fluffy-bassoon-2/sources/robotwin-baseline-smoke-20261028-r1/RoboTwin
export ROBOTWIN_ASSETS_PATH=$ROBOTWIN_PATH ASSETS_PATH=$ROBOTWIN_PATH
export PYTHONPATH=$TASK_CODE/src:$REPO_PATH:$ROBOTWIN_PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export FLUFFY_GPU_EXECUTION=baige OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl ROBOT_PLATFORM=ALOHA
if [[ ${CUDA_VISIBLE_DEVICES+x} != x ]]; then export CUDA_VISIBLE_DEVICES=0; fi
TASK_STARTED=$(date +%s)
finish() { TASK_RC=$?;trap - EXIT;printf '{"exit_code":%s,"worker_gpu_seconds":%s}\n' "$TASK_RC" "$(($(date +%s)-TASK_STARTED))" > "$TASK_OUT/EXIT.json"; }
trap finish EXIT
"$TASK_PYTHON" "$TASK_CODE/scripts/[train]gpu_probe.py" --baige-container --baige-exclusive-training --purpose training --expected-count 1 --output "$TASK_OUT/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP=$TASK_OUT/gpu-probe.json
mkdir -p /root/.sapien/physx
if [[ ! -e /root/.sapien/physx/105.1-physx-5.3.1.patch0 ]]; then
  ln -s /mowenjie/fluffy-bassoon/inputs/maniskill-scene-assets-20261028-r1/physx/105.1-physx-5.3.1.patch0 /root/.sapien/physx/105.1-physx-5.3.1.patch0
fi
TASK_LIMIT=1100;if [[ $TASK_SCENES == 64 ]]; then TASK_LIMIT=1700; fi
if [[ $TASK_PROFILE_MODE == paired ]]; then TASK_LIMIT=2300; fi
timeout --signal=INT --kill-after=15s "$TASK_LIMIT" "$TASK_PYTHON" "$(dirname "${BASH_SOURCE[0]}")/memory_profile.py" > "$TASK_OUT/profile.log" 2>&1
