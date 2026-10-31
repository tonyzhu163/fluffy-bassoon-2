#!/usr/bin/env bash
# Root/launch owner alone funds and dispatches.8x45min OR4x90min, same global2048.
set -Eeuo pipefail
TASK_STARTED=$(date +%s)
TASK_SOURCE=${1:?exact clean combined source}
TASK_RUN=${2:?fresh immutable native pilot run ID}
TASK_WORLD=${3:?explicit4or8 allocated GPUs}
TASK_FUNDING=${4:?reviewed approved funding JSON, never generated here}
TASK_RESUME=${5:-}
TASK_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_PYTHON=${NATIVE_PYTHON:-/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python}
TASK_OUTPUT=/mowenjie/fluffy-bassoon-2/outputs/$TASK_RUN
export REPO_PATH=/mowenjie/fluffy-bassoon-2/sources/robotwin-baseline-smoke-20261028-r1/RLinf
export ROBOTWIN_PATH=/mowenjie/fluffy-bassoon-2/sources/robotwin-baseline-smoke-20261028-r1/RoboTwin
export EMBODIED_PATH=$REPO_PATH/examples/embodiment
export PYTHONPATH=$TASK_CODE/src:$REPO_PATH:$ROBOTWIN_PATH
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl ROBOT_PLATFORM=ALOHA NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export NATIVE_PPO_WORLD=$TASK_WORLD NATIVE_PPO_PLACEMENT=0-$((TASK_WORLD-1))
export NATIVE_PPO_OUTPUT=$TASK_OUTPUT
export NATIVE_PPO_ACCOUNTING_ROOT=${NATIVE_PPO_ACCOUNTING_ROOT:-$TASK_OUTPUT/action-accounting}
export NATIVE_PPO_MODE=${NATIVE_PPO_MODE:-train}
export NATIVE_PPO_ATTEMPT=${NATIVE_PPO_ATTEMPT:-attempt-0001}
export NATIVE_PPO_MODEL=${NATIVE_PPO_MODEL:-/mowenjie/fluffy-bassoon-2/cache/robotwin/pi05-sft-adjust-bottle-fa8df6ed}
export NATIVE_PPO_INITIALIZER_REVISION=${NATIVE_PPO_INITIALIZER_REVISION:-fa8df6ed103db0f5549c122f3a17c00ba6426c98}
export NATIVE_PPO_RESUME_DIR=$TASK_RESUME
: "${ROBOTWIN_ASSETS_PATH:?set the existing qualified native RoboTwin assets directory}"
cd "$TASK_CODE"
task_cleanup() {
  TASK_RC=$?
  trap - EXIT
  if [ -f "$TASK_OUTPUT/worker-time.json" ]; then
    "$TASK_PYTHON" -m fluffy_bassoon.meta_rl.rlinf_native_ppo finish \
      --output "$TASK_OUTPUT" --exit-code "$TASK_RC" || true
    printf '%s\n' "$TASK_RC" > "$TASK_OUTPUT/exit-code.txt"
  fi
  exit "$TASK_RC"
}
trap task_cleanup EXIT
"$TASK_PYTHON" -m fluffy_bassoon.meta_rl.rlinf_native_ppo preflight \
  --code "$TASK_CODE" --source "$TASK_SOURCE" --output "$TASK_OUTPUT" \
  --started-unix "$TASK_STARTED" --world "$TASK_WORLD" --funding "$TASK_FUNDING"
export NATIVE_PPO_DEADLINE
NATIVE_PPO_DEADLINE=$("$TASK_PYTHON" -c 'import json,sys;print(json.load(open(sys.argv[1]))["deadline_unix"])' "$TASK_OUTPUT/preflight.json")
"$TASK_PYTHON" "$TASK_CODE/scripts/[train]gpu_probe.py" --baige-container \
  --baige-exclusive-training --purpose training --expected-count "$TASK_WORLD" \
  --output "$TASK_OUTPUT/gpu-probe.json"
TASK_REMAINING=$("$TASK_PYTHON" -c 'import os,time;print(int(float(os.environ["NATIVE_PPO_DEADLINE"])-time.time()-20))')
test "$TASK_REMAINING" -gt 90
timeout --signal=INT --kill-after=20s "$TASK_REMAINING" "$TASK_PYTHON" \
  "$TASK_CODE/scripts/diagnostics/rlinf_native_ppo_entry.py" \
  --config-path "$TASK_CODE/configs/native_ppo" --config-name adjust_bottle_published_batch_two \
  "runner.resume_dir=${TASK_RESUME:-null}"
