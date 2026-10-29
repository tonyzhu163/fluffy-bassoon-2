#!/usr/bin/env bash
# One allocated GPU; preflight then the actual, resumable matched-budget run.
set -euo pipefail
TASK_COMMIT=${1:?40-hex source commit required}
TASK_RUN=${2:?new output identity required}
TASK_RESUME=${3:-}
[[ "$TASK_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$TASK_RUN" =~ ^robotwin-noise-matched-[a-zA-Z0-9-]+$ ]]
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_WORKTREE=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUTPUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_PYTHON=$TASK_ROOT/envs/robotwin-vlrla-20261029-r1/bin/python
TASK_RLINF=$TASK_ROOT/sources/robotwin-baseline-smoke-20261028-r1/RLinf
TASK_ROBOTWIN=$TASK_ROOT/sources/robotwin-baseline-smoke-20261028-r1/RoboTwin
TASK_MODEL=$TASK_ROOT/cache/robotwin/pi05-motus-effc6e1a-r1
TASK_NORM=$TASK_MODEL/assets/pi0.5_clean_randomize_joint_training/norm_stats.json
test ! -e "$TASK_OUTPUT"
mkdir -p "$TASK_OUTPUT"
exec > >(tee "$TASK_OUTPUT/console.log") 2>&1
trap 'TASK_EXIT=$?; printf "%s\n" "$TASK_EXIT" > "$TASK_OUTPUT/exit-code.txt"' EXIT
export PATH="$(dirname "$TASK_PYTHON"):$PATH"
export PYTHONPATH="$TASK_WORKTREE/src:$TASK_RLINF:$TASK_ROBOTWIN"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 FLUFFY_GPU_EXECUTION=baige
export HF_HOME=/mowenjie/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MPLBACKEND=Agg
export OPENPI_DATA_HOME=/mowenjie/fluffy-bassoon/inputs/rlinf-openpi-tokenizer-befaa248-20261028-r1
export ROBOT_PLATFORM=ALOHA ASSETS_PATH="$TASK_ROBOTWIN" ROBOTWIN_PATH="$TASK_ROBOTWIN"
export EMBODIED_PATH="$TASK_RLINF/examples/embodiment" REPO_PATH="$TASK_RLINF"
export FB2_ROBOTWIN_RENDER_RECORDS="$TASK_OUTPUT/renderer-bindings"
mkdir -p /root/.sapien/physx
if [ ! -e /root/.sapien/physx/105.1-physx-5.3.1.patch0 ]; then
    ln -s /mowenjie/fluffy-bassoon/inputs/maniskill-scene-assets-20261028-r1/physx/105.1-physx-5.3.1.patch0 /root/.sapien/physx/105.1-physx-5.3.1.patch0
fi
cd "$TASK_WORKTREE"
test "$(git rev-parse HEAD)" = "$TASK_COMMIT"
test -z "$(git status --porcelain)"
"$TASK_PYTHON" "scripts/[train]gpu_probe.py" --expected-count 1 \
    --output "$TASK_OUTPUT/gpu-probe.json" --purpose training --baige-container
export LYCO_BAIGE_GPU_MAP="$TASK_OUTPUT/gpu-probe.json"
timeout 300 "$TASK_PYTHON" -m pytest -q \
    tests/test_robotwin_noise.py tests/test_vla_noise_ppo.py tests/test_vla_noise_sac.py \
    tests/test_vla_noise_ppo_review.py
TASK_ARGS=(--source-commit "$TASK_COMMIT" --model-path "$TASK_MODEL" \
    --norm-stats "$TASK_NORM" --robotwin-root "$TASK_ROBOTWIN" --rlinf-root "$TASK_RLINF" \
    --output "$TASK_OUTPUT/experiment" --source-steps 10240000 \
    --ppo-trials-per-update 256 --train-episodes 4 --eval-episodes 8 \
    --methods ppo sac ppo_reset --native-horizon 32 --episode-limit 400 \
    --task-config-name demo_clean \
    --sac-warmup 500 --sac-batch-size 256 --sac-utd 20 \
    --wall-seconds 2220 --boundary-reserve-seconds 180)
if [ -n "$TASK_RESUME" ]; then
    TASK_ARGS+=(--resume "$TASK_RESUME")
fi
timeout --signal=TERM --kill-after=30s 2400 "$TASK_PYTHON" \
    scripts/diagnostics/robotwin_noise_compare.py "${TASK_ARGS[@]}"
