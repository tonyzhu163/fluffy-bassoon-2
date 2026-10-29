#!/usr/bin/env bash
# Run in one exclusive Baige scheduler container. Outer timeout: 10,740 s.
set -euo pipefail
TASK_SOURCE=${1:?reviewed 40-character source commit}
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_RUN=metaworld-pi05-noise-baige-20261029-r1
TASK_WORKTREE=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUTPUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_PYTHON=$TASK_ROOT/envs/metaworld-3.1.1-r1/bin/python
TASK_UV=/mowenjie/bin/uv
TASK_RLINF=/mowenjie/fluffy-bassoon/third_party/rlinf-e1a82ebf-complete-heads-fd8ce503-20261028-r1
TASK_MODEL=$TASK_ROOT/inputs/rlinf-pi05-metaworld-sft-60b2d03-20261029-r1
TASK_STARTED=$(date +%s)
[[ $TASK_SOURCE =~ ^[0-9a-f]{40}$ ]]
[[ $(hostname) =~ ^job-[a-z0-9]+-master-0$ ]]
test -n "${KUBERNETES_SERVICE_HOST:-}"
test -n "${NVIDIA_VISIBLE_DEVICES:-}"
test -s "$TASK_OUTPUT/preparation/model-receipt.json"
test ! -e "$TASK_OUTPUT/exit-code.txt"
test ! -e "$TASK_OUTPUT/s0"
cd "$TASK_WORKTREE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
exec > >(tee "$TASK_OUTPUT/console.log") 2>&1
trap 'TASK_EXIT=$?; printf "%s\n" "$TASK_EXIT" > "$TASK_OUTPUT/exit-code.txt"; date -Iseconds > "$TASK_OUTPUT/ended-at.txt"' EXIT
date -Iseconds > "$TASK_OUTPUT/started-at.txt"
printf '%s\n' "$TASK_SOURCE" > "$TASK_OUTPUT/source-commit.txt"
hostname > "$TASK_OUTPUT/hostname.txt"
export PYTHONPATH="$TASK_WORKTREE/src:$TASK_RLINF" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export FLUFFY_GPU_EXECUTION=baige CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export HF_HOME=/mowenjie/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export OPENPI_DATA_HOME=/mowenjie/fluffy-bassoon/inputs/rlinf-openpi-tokenizer-befaa248-20261028-r1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
"$TASK_UV" --version
printf '%s  %s\n' 6e877ae8e2c9c33a2b8d1e3a8a60649ef39b0268da60a3ee71bc5e33cbca7221 "$TASK_MODEL/model.safetensors" | sha256sum -c -
test -s "$TASK_MODEL/lerobot/metaworld_mt50/norm_stats.json"
"$TASK_UV" run --no-project --no-python-downloads --python "$TASK_PYTHON" python \
  'scripts/[train]gpu_probe.py' --expected-count 1 --purpose evaluation --baige-container \
  --output "$TASK_OUTPUT/qualification-gpu-probe.json"
"$TASK_UV" run --no-project --no-python-downloads --python "$TASK_PYTHON" python -m pytest \
  -q -p no:cacheprovider tests/test_metaworld_pi05.py tests/test_libero_split.py \
  > "$TASK_OUTPUT/qualification-tests.log" 2>&1
cat "$TASK_OUTPUT/qualification-tests.log"
if grep -Eq '[0-9]+ skipped' "$TASK_OUTPUT/qualification-tests.log"; then
  echo 'Real Meta-World qualification must not skip.'
  exit 4
fi
TASK_REMAINING=$((10440 - $(date +%s) + TASK_STARTED))
test "$TASK_REMAINING" -ge 600
# Reach with both visibilities precedes push; preserve partial outcomes at the deadline.
"$TASK_UV" run --no-project --no-python-downloads --python "$TASK_PYTHON" python \
  scripts/diagnostics/metaworld_pi05_noise_cache.py \
  --rlinf-root "$TASK_RLINF" --model-path "$TASK_MODEL" \
  --tasks reach-v3,push-v3 --visibility visible,hidden \
  --scenes 4 --arms 16 --repeats 2 --time-budget-seconds "$TASK_REMAINING" \
  --output "$TASK_OUTPUT/s0"
test -z "$(git status --porcelain)"
