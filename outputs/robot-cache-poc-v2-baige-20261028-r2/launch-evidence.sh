#!/bin/bash
# Immutable launcher for the approved bounded native robot POC.
# Scheduler allocation: 2 GPUs, enough CPU for two 4-thread live workers and
# three concurrent 2-thread CPU workers; this script never obtains resources.
set -euo pipefail

TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_RUN=pi05-live-bridge-poc-baige-20261028-r1
TASK_SOURCE=c3e9cca68b93b6e777cbe5314e130bcaf85c8e14
TASK_WORKTREE=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUTPUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_CPU_OUTPUT=$TASK_ROOT/outputs/robot-cache-poc-v2-baige-20261028-r2
TASK_OLD=/mowenjie/fluffy-bassoon
TASK_PYTHON=$TASK_OLD/envs/maniskill-pi05-rlinf-e1a82ebf-20261028-r1/bin/python
TASK_RLINF=$TASK_OLD/third_party/rlinf-e1a82ebf-complete-heads-fd8ce503-20261028-r1
TASK_MODEL=$TASK_OLD/inputs/rlinf-pi05-maniskill-rl-a0cbb898-20261028-r1/checkpoints/global_step_150/actor
TASK_MODEL_SHA=0adb5b40f064760ff245f53ae0a4e078b86680b65f77d1f5991a173543edd5a1
TASK_CACHE=$TASK_OLD/outputs/maniskill-pi05-baige-gates-20261028-r7/execution/headroom-rl/outcomes.jsonl
TASK_CACHE_SHA=74910dc755b959fe86ec0a3d4cf1c05957ba57510c64bbb1bb2c85ba56e41575

[[ $TASK_SOURCE =~ ^[0-9a-f]{40}$ ]]
[[ $(hostname) =~ ^job-[a-z0-9]+-master-0$ ]]
test -n "${KUBERNETES_SERVICE_HOST:-}"
test -n "${NVIDIA_VISIBLE_DEVICES:-}"
cd "$TASK_WORKTREE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
mkdir "$TASK_OUTPUT"
mkdir "$TASK_CPU_OUTPUT"
exec > >(tee "$TASK_OUTPUT/console.log") 2>&1
trap 'TASK_EXIT=$?; printf "%s\n" "$TASK_EXIT" > "$TASK_OUTPUT/exit-code.txt"' EXIT

export PYTHONPATH="$TASK_WORKTREE/src" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export FLUFFY_GPU_EXECUTION=baige CUDA_VISIBLE_DEVICES=0,1
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
test -d /root/.sapien/physx/105.1-physx-5.3.1.patch0

hostname
date -u
printf '%s  %s\n' 01b8a68a0085cafb64ce3b0716732305e78921086b948654c2a4ce9edc5b71c9 "$TASK_RLINF/rlinf/models/embodiment/openpi/checkpoint.py" | sha256sum -c -
printf '%s  %s\n' "$TASK_CACHE_SHA" "$TASK_CACHE" | sha256sum -c -

# One parent probe attests both scheduler GPUs before any model worker starts.
"$TASK_PYTHON" 'scripts/[train]gpu_probe.py' --expected-count 2 --purpose training \
  --baige-container --output "$TASK_OUTPUT/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP="$TASK_OUTPUT/gpu-probe.json"
"$TASK_PYTHON" - "$TASK_OUTPUT/gpu-probe.json" > "$TASK_OUTPUT/gpu-uuids.tsv" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1]))
assert report['status'] == 'passed' and report['allocation_type'] == 'baige_container'
rows = report['devices']
assert len(rows) == 2
by_token = {row['visibility_token']: row['uuid'] for row in rows}
assert set(by_token) == {'0', '1'} and len(set(by_token.values())) == 2
print(by_token['0'], by_token['1'], sep='\t')
PY
read -r TASK_GPU0_UUID TASK_GPU1_UUID < "$TASK_OUTPUT/gpu-uuids.tsv"

# These CPU checks still execute inside the actual scheduled container.
CUDA_VISIBLE_DEVICES='' "$TASK_PYTHON" -m pytest -q -p no:cacheprovider \
  tests/test_pi05_feature_bridge.py tests/test_pi05_bridge_policy.py \
  tests/test_pi05_bridge_train.py tests/test_maniskill_rotation_poc.py \
  > "$TASK_OUTPUT/qualification.log" 2>&1

TASK_PIDS=()
TASK_NAMES=()
for TASK_CORE in persistent_ctm transformer_xl; do
  if test "$TASK_CORE" = persistent_ctm; then
    TASK_TOKEN=0
    TASK_UUID=$TASK_GPU0_UUID
    TASK_TICKS=4
  else
    TASK_TOKEN=1
    TASK_UUID=$TASK_GPU1_UUID
    TASK_TICKS=1
  fi
  CUDA_VISIBLE_DEVICES="$TASK_TOKEN" LYCO_BAIGE_ISOLATED_UUID="$TASK_UUID" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
    "$TASK_PYTHON" -m fluffy_bassoon.meta_rl.pi05_bridge_train \
    --source-commit "$TASK_SOURCE" --output-dir "$TASK_OUTPUT/$TASK_CORE-seed0" \
    --rlinf-root "$TASK_RLINF" --model-path "$TASK_MODEL" \
    --norm-stats "$TASK_MODEL/assets/global_step_150/meta/norm_stats.json" \
    --expected-model-sha256 "$TASK_MODEL_SHA" --allocation-probe "$TASK_OUTPUT/gpu-probe.json" \
    --expected-gpu-uuid "$TASK_UUID" --core "$TASK_CORE" --ticks "$TASK_TICKS" --width 64 \
    --num-envs 2 --attempts 4 --objects 0 1 --seed 0 --updates 100 --ppo-epochs 4 \
    --learning-rate .0001 --value-coef .01 --reward-offset 0 \
    --noise-scale .05 --residual-scale 1 --target-kl .05 \
    --eval-at 0 10 50 100 --eval-batches 1 --threads 4 --max-elapsed-seconds 9000 \
    > "$TASK_OUTPUT/$TASK_CORE-seed0.log" 2>&1 &
  TASK_PIDS+=("$!")
  TASK_NAMES+=("live-$TASK_CORE-seed0")
done

# Scalar v2 is a separate bounded plumbing control: three CPU workers, same SHA.
for TASK_CORE in persistent_ctm transformer_xl gru; do
  (
    for TASK_SEED in 0 1 2; do
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    "$TASK_PYTHON" -m fluffy_bassoon.meta_rl.maniskill_rotation \
    --source-commit "$TASK_SOURCE" --cache "$TASK_CACHE" --cache-sha256 "$TASK_CACHE_SHA" \
    --expected-hostname "$(hostname)" --core "$TASK_CORE" --seed "$TASK_SEED" --updates 500 \
    --num-envs 64 --attempts 8 --eval-trials 256 --eval-every 100 \
    --value-coef .01 --reward-offset .5 --threads 2 --max-elapsed-seconds 900 \
    --output-dir "$TASK_CPU_OUTPUT/$TASK_CORE-seed$TASK_SEED" \
    > "$TASK_CPU_OUTPUT/$TASK_CORE-seed$TASK_SEED.log" 2>&1
    done
  ) &
  TASK_PIDS+=("$!")
  TASK_NAMES+=("scalar-$TASK_CORE-seeds012")
done

printf 'name\tpid\n' > "$TASK_OUTPUT/worker-pids.tsv"
for TASK_INDEX in "${!TASK_PIDS[@]}"; do
  printf '%s\t%s\n' "${TASK_NAMES[$TASK_INDEX]}" "${TASK_PIDS[$TASK_INDEX]}" >> "$TASK_OUTPUT/worker-pids.tsv"
done

# Wait for every owned worker even when an earlier worker fails. Never hide failures.
TASK_FAILED=0
printf 'name\tpid\texit_code\n' > "$TASK_OUTPUT/worker-exit-codes.tsv"
for TASK_INDEX in "${!TASK_PIDS[@]}"; do
  if wait "${TASK_PIDS[$TASK_INDEX]}"; then
    TASK_CHILD_EXIT=0
  else
    TASK_CHILD_EXIT=$?
    TASK_FAILED=1
  fi
  printf '%s\t%s\t%s\n' "${TASK_NAMES[$TASK_INDEX]}" "${TASK_PIDS[$TASK_INDEX]}" "$TASK_CHILD_EXIT" \
    >> "$TASK_OUTPUT/worker-exit-codes.tsv"
done
exit "$TASK_FAILED"
