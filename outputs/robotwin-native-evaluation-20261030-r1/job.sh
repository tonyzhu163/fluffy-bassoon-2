#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_RUN=robotwin-native-evaluation-20261030-r1
TASK_PIN=2fd2096906e52aabeaeb984f333fe1aaa6c4e196
TASK_CODE=$TASK_ROOT/worktrees/robotwin-history-live-wrapper-20261030-r2
TASK_OUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_PYTHON=$TASK_ROOT/envs/robotwin-vlrla-20261029-r1/bin/python
TASK_RLINF=$TASK_ROOT/sources/robotwin-baseline-smoke-20261028-r1/RLinf
TASK_ROBOTWIN=$TASK_ROOT/sources/robotwin-baseline-smoke-20261028-r1/RoboTwin
cd "$TASK_CODE"
test "$(git rev-parse HEAD)" = "$TASK_PIN"
test -z "$(git status --porcelain)"
test ! -e "$TASK_OUT"
mkdir -p "$TASK_OUT"
exec > >(tee "$TASK_OUT/console.log") 2>&1
trap 'TASK_EXIT=$?; printf "%s\n" "$TASK_EXIT" > "$TASK_OUT/exit-code.txt"' EXIT
ulimit -c 0
cp "${BASH_SOURCE[0]}" "$TASK_OUT/launch-script.sh"
export PATH="$(dirname "$TASK_PYTHON"):$PATH"
export PYTHONPATH="$TASK_CODE/src:$TASK_RLINF:$TASK_ROBOTWIN"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 PYTHONDONTWRITEBYTECODE=1 FLUFFY_GPU_EXECUTION=baige
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME=/mowenjie/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MPLBACKEND=Agg
export OPENPI_DATA_HOME=/mowenjie/fluffy-bassoon/inputs/rlinf-openpi-tokenizer-befaa248-20261028-r1
export ROBOT_PLATFORM=ALOHA ASSETS_PATH="$TASK_ROBOTWIN" ROBOTWIN_PATH="$TASK_ROBOTWIN"
export EMBODIED_PATH="$TASK_RLINF/examples/embodiment" REPO_PATH="$TASK_RLINF"
mkdir -p /root/.sapien/physx
if [ ! -e /root/.sapien/physx/105.1-physx-5.3.1.patch0 ]; then
 ln -s /mowenjie/fluffy-bassoon/inputs/maniskill-scene-assets-20261028-r1/physx/105.1-physx-5.3.1.patch0 /root/.sapien/physx/105.1-physx-5.3.1.patch0
fi
"$TASK_PYTHON" -c 'import json,socket;print(json.dumps(dict(hostname=socket.gethostname(),gpus=1,cpus=16,memory_gib=128,shared_memory_gib=8)))' > "$TASK_OUT/allocation-context.json"
timeout --signal=TERM --kill-after=10s 5330 "$TASK_PYTHON" \
 -m fluffy_bassoon.meta_rl.robotwin_history_live \
 --plan "$TASK_CODE/outputs/robotwin-native-evaluation-plan-20261030-r1/plan.json" \
 --plan-sha b911dd972cc30e6f6b02cacf831eafa84738bd66ebd032824bb5fc94d340287c \
 --output "$TASK_OUT/evaluation" --campaign-root "$TASK_OUT/evaluation-ledger" \
 --campaign-id "$TASK_RUN" --source-commit "$TASK_PIN" --expected-hostname "$(hostname)" \
 --robotwin-root "$TASK_ROBOTWIN" --rlinf-root "$TASK_RLINF" \
 --wall-seconds 5300 --prior-gpu-seconds 0
