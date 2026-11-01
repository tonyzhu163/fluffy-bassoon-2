#!/bin/bash
# Stage 0 noise-arm screen on one exclusive Baige GPU (PLAN.md, Stage 0).
# Usage, inside the allocated container, from a clean worktree at the source commit:
#   bash outputs/noise-mode-icrl-20261029-r1/launch-s0.sh <40-hex source> noise-mode-icrl-20261029-s0-r<N>
# Runs the CPU tests, then S0a (determinism), S0b (C2 physics) and S0c (C1 objects) in
# priority order with per-run time budgets summing to 6,900 s, then the CPU gate analysis.
set -euo pipefail
TASK_SOURCE=${1:?source commit}
TASK_RUN=${2:?fresh run identity}
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_WORKTREE=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUTPUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_OLD=/mowenjie/fluffy-bassoon
TASK_PYTHON=$TASK_OLD/envs/maniskill-pi05-rlinf-e1a82ebf-20261028-r1/bin/python
TASK_RLINF=$TASK_OLD/third_party/rlinf-e1a82ebf-complete-heads-fd8ce503-20261028-r1
TASK_MODEL=$TASK_OLD/inputs/rlinf-pi05-maniskill-rl-a0cbb898-20261028-r1/checkpoints/global_step_150/actor
[[ $TASK_SOURCE =~ ^[0-9a-f]{40}$ ]]
[[ $TASK_RUN =~ ^noise-mode-icrl-20261029-s0-r[0-9]+$ ]]
[[ $(hostname) =~ ^job-[a-z0-9]+-master-0$ ]]
test -n "${KUBERNETES_SERVICE_HOST:-}"
test -n "${NVIDIA_VISIBLE_DEVICES:-}"
cd "$TASK_WORKTREE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
mkdir "$TASK_OUTPUT"
exec > >(tee "$TASK_OUTPUT/console.log") 2>&1
trap 'TASK_EXIT=$?; printf "%s\n" "$TASK_EXIT" > "$TASK_OUTPUT/exit-code.txt"' EXIT
export PYTHONPATH="$TASK_WORKTREE/src" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export FLUFFY_GPU_EXECUTION=baige CUDA_VISIBLE_DEVICES=0
unset FLUFFY_META_RL_CUDA_QUALIFICATION
export HF_HOME=/mowenjie/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MANISKILL_ASSET_DIR="$TASK_OLD/inputs/rlinf-maniskill-assets-c23fc188-20261028-r1"
export MS_ASSET_DIR="$TASK_OLD/inputs/maniskill-scene-assets-20261028-r1"
export OPENPI_DATA_HOME="$TASK_OLD/inputs/rlinf-openpi-tokenizer-befaa248-20261028-r1"
mkdir -p /root/.sapien/physx
if test ! -e /root/.sapien/physx/105.1-physx-5.3.1.patch0; then
  test ! -L /root/.sapien/physx/105.1-physx-5.3.1.patch0
  ln -s "$MS_ASSET_DIR/physx/105.1-physx-5.3.1.patch0" /root/.sapien/physx/105.1-physx-5.3.1.patch0
fi
printf '%s  %s\n' 01b8a68a0085cafb64ce3b0716732305e78921086b948654c2a4ce9edc5b71c9 "$TASK_RLINF/rlinf/models/embodiment/openpi/checkpoint.py" | sha256sum -c -
CUDA_VISIBLE_DEVICES='' "$TASK_PYTHON" -m pytest -q -p no:cacheprovider \
  tests/test_maniskill_noise.py tests/test_maniskill_pi05.py tests/test_maniskill_rotation_poc.py \
  > "$TASK_OUTPUT/qualification.log" 2>&1
TASK_COMMON=(--source-commit "$TASK_SOURCE" --rlinf-root "$TASK_RLINF" --model-path "$TASK_MODEL"
  --norm-stats "$TASK_MODEL/assets/global_step_150/meta/norm_stats.json"
  --expected-model-sha256 0adb5b40f064760ff245f53ae0a4e078b86680b65f77d1f5991a173543edd5a1)
TASK_CACHE=scripts/diagnostics/maniskill_pi05_noise_cache.py
"$TASK_PYTHON" "$TASK_CACHE" "${TASK_COMMON[@]}" --output-dir "$TASK_OUTPUT/s0a" \
  --variants 1:1 --objects 0,1 --scenes-per-object 1 --repeats 2 --num-envs 34 --time-budget-seconds 900
"$TASK_PYTHON" "$TASK_CACHE" "${TASK_COMMON[@]}" --output-dir "$TASK_OUTPUT/s0b" \
  --variants 1:1,0.3:1,1:4,0.3:4 --objects 0,1,2,3,4,5,6,7 --scenes-per-object 2 --num-envs 34 \
  --time-budget-seconds 3600
"$TASK_PYTHON" "$TASK_CACHE" "${TASK_COMMON[@]}" --output-dir "$TASK_OUTPUT/s0c" \
  --variants 1:1 --objects 0,1,2,3,4,5,6,7,16,17,18,19 --scenes-per-object 4 --num-envs 48 \
  --time-budget-seconds 2400
# Gate analysis reads saved artifacts only and can be rerun on CPU; record failures instead of stopping.
run_gates() {
  local name=$1; shift
  "$TASK_PYTHON" -m fluffy_bassoon.meta_rl.noise_screen --cache-dir "$TASK_OUTPUT/$name" \
    --output "$TASK_OUTPUT/$name/gates.json" "$@" || printf '%s %s\n' "$name" "$?" >> "$TASK_OUTPUT/gate-failures.txt"
}
run_gates s0a
run_gates s0b --condition variant --train 0,1,2,3 --evaluate 0,1,2,3
run_gates s0c --condition object --train 0:0,0:1,0:2,0:3,0:4,0:5,0:6,0:7 --evaluate 0:16,0:17,0:18,0:19
