#!/usr/bin/env bash
# One fresh, exclusive one-GPU allocation; two independent N6 PPO learners.
set -Eeuo pipefail
TASK_SOURCE=${1:?exact source commit}
TASK_RUN=${2:?fresh packed qualification identity}
[[ "$TASK_SOURCE" =~ ^[0-9a-f]{40}$ ]]
[[ "$TASK_RUN" =~ ^robotwin-packed-ppo-[a-zA-Z0-9-]+$ ]]
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_WORKTREE=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUTPUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_PYTHON=$TASK_ROOT/envs/robotwin-vlrla-20261029-r1/bin/python
TASK_NATIVE=$TASK_ROOT/sources/robotwin-baseline-smoke-20261028-r1
test ! -e "$TASK_OUTPUT"
mkdir -p "$TASK_OUTPUT"
exec > >(tee "$TASK_OUTPUT/console.log") 2>&1
trap 'TASK_RC=$?; printf "%s\n" "$TASK_RC" > "$TASK_OUTPUT/exit-code.txt"' EXIT
cd "$TASK_WORKTREE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export FLUFFY_GPU_EXECUTION=baige CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH=$TASK_WORKTREE/src:$TASK_NATIVE/RLinf:$TASK_NATIVE/RoboTwin
export HF_HOME=/mowenjie/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLBACKEND=Agg
export OPENPI_DATA_HOME=/mowenjie/fluffy-bassoon/inputs/rlinf-openpi-tokenizer-befaa248-20261028-r1
export ROBOT_PLATFORM=ALOHA ASSETS_PATH=$TASK_NATIVE/RoboTwin ROBOTWIN_PATH=$TASK_NATIVE/RoboTwin
export EMBODIED_PATH=$TASK_NATIVE/RLinf/examples/embodiment REPO_PATH=$TASK_NATIVE/RLinf
mkdir -p /root/.sapien/physx
if [ ! -e /root/.sapien/physx/105.1-physx-5.3.1.patch0 ]; then
  ln -s /mowenjie/fluffy-bassoon/inputs/maniskill-scene-assets-20261028-r1/physx/105.1-physx-5.3.1.patch0 /root/.sapien/physx/105.1-physx-5.3.1.patch0
fi
# Probe once while the allocation is idle; both owned learners then share it.
"$TASK_PYTHON" 'scripts/[train]gpu_probe.py' --expected-count 1 --purpose training \
  --baige-container --output "$TASK_OUTPUT/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP=$TASK_OUTPUT/gpu-probe.json
timeout --signal=INT --kill-after=90s 2610 "$TASK_PYTHON" \
  scripts/diagnostics/robotwin_packed_ppo_probe.py --source-commit "$TASK_SOURCE" \
  --worktree "$TASK_WORKTREE" --output "$TASK_OUTPUT/packed" \
  --gpu-probe "$TASK_OUTPUT/gpu-probe.json" --maximum-seconds 2520
