#!/usr/bin/env bash
# Execution-only continuation; one method, one existing child ledger, no imports.
# Allocate1GPU/16CPU/128GiB/shm8;13h scheduler with automatic retries off.
set -euo pipefail
TASK_COMMIT=${1:?actual qualified execution commit required}
TASK_RUN=${2:?fresh robotwin-native-continuation identity required}
TASK_METHOD=${3:?ppo, sac or ppo_reset required}
TASK_PARENT=${4:?original immutable parent manifest required}
TASK_MANIFEST=${5:?audited initial selected-trial/boundary manifest required}
TASK_MANIFEST_SHA=${6:?initial manifest SHA256 required}
TASK_MINUTES=${7:-780}
[[ "$TASK_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$TASK_RUN" =~ ^robotwin-native-continuation-[a-z0-9_-]+$ ]]
[[ "$TASK_METHOD" =~ ^(ppo|sac|ppo_reset)$ ]]
[[ "$TASK_MANIFEST_SHA" =~ ^[0-9a-f]{64}$ ]]
[[ "$TASK_MINUTES" =~ ^[0-9]+$ ]]
(( TASK_MINUTES >= 90 && TASK_MINUTES <= 780 ))
TASK_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_CODE=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_PYTHON=$TASK_ROOT/envs/robotwin-vlrla-20261029-r1/bin/python
TASK_RLINF=$TASK_ROOT/sources/robotwin-baseline-smoke-20261028-r1/RLinf
TASK_ROBOTWIN=$TASK_ROOT/sources/robotwin-baseline-smoke-20261028-r1/RoboTwin
TASK_MODEL=$TASK_ROOT/cache/robotwin/pi05-motus-effc6e1a-r1
TASK_NORM=$TASK_MODEL/assets/pi0.5_clean_randomize_joint_training/norm_stats.json
cd "$TASK_CODE"
test "$(git rev-parse HEAD)" = "$TASK_COMMIT"
test -z "$(git status --porcelain)"
test ! -e "$TASK_OUT"
mkdir -p "$TASK_OUT"
exec > >(tee "$TASK_OUT/console.log") 2>&1
trap 'TASK_EXIT=$?; printf "%s\n" "$TASK_EXIT" > "$TASK_OUT/exit-code.txt"' EXIT
ulimit -c 0
export PATH="$(dirname "$TASK_PYTHON"):$PATH"
export PYTHONPATH="$TASK_CODE/src:$TASK_RLINF:$TASK_ROBOTWIN"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 PYTHONDONTWRITEBYTECODE=1 FLUFFY_GPU_EXECUTION=baige
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME=/mowenjie/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MPLBACKEND=Agg
export OPENPI_DATA_HOME=/mowenjie/fluffy-bassoon/inputs/rlinf-openpi-tokenizer-befaa248-20261028-r1
export ROBOT_PLATFORM=ALOHA ASSETS_PATH="$TASK_ROBOTWIN" ROBOTWIN_PATH="$TASK_ROBOTWIN"
export EMBODIED_PATH="$TASK_RLINF/examples/embodiment" REPO_PATH="$TASK_RLINF"
export FB2_ROBOTWIN_RENDER_RECORDS="$TASK_OUT/renderer-bindings"
mkdir -p /root/.sapien/physx
if [ ! -e /root/.sapien/physx/105.1-physx-5.3.1.patch0 ]; then
    ln -s /mowenjie/fluffy-bassoon/inputs/maniskill-scene-assets-20261028-r1/physx/105.1-physx-5.3.1.patch0 /root/.sapien/physx/105.1-physx-5.3.1.patch0
fi
cp "$TASK_SCRIPT_DIR/job.sh" "$TASK_SCRIPT_DIR/supervise.py" "$TASK_OUT/"
TASK_ENVELOPE=$(( TASK_MINUTES * 60 - 600 ))
if (( TASK_ENVELOPE > 43200 )); then TASK_ENVELOPE=43200; fi
TASK_EMERGENCY=$(( TASK_MINUTES * 60 - 180 ))
timeout --signal=TERM --kill-after=30s "$TASK_EMERGENCY" "$TASK_PYTHON" \
  "$TASK_SCRIPT_DIR/supervise.py" \
  --source-commit "$TASK_COMMIT" --run-id "$TASK_RUN" --method "$TASK_METHOD" \
  --parent-manifest "$TASK_PARENT" --resume-trial-manifest "$TASK_MANIFEST" \
  --resume-trial-manifest-sha256 "$TASK_MANIFEST_SHA" \
  --code-root "$TASK_CODE" --output "$TASK_OUT" --robotwin-root "$TASK_ROBOTWIN" \
  --rlinf-root "$TASK_RLINF" --model-path "$TASK_MODEL" --norm-stats "$TASK_NORM" \
  --wall-seconds "$TASK_ENVELOPE"
