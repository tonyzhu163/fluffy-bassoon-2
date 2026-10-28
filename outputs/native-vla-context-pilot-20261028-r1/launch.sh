#!/bin/bash
set -euo pipefail
TASK_SOURCE=${1:?reviewed source SHA}
TASK_RUN=${2:?fresh output identity}
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_OLD=/mowenjie/fluffy-bassoon
TASK_WORKTREE=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUTPUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_PYTHON=$TASK_OLD/envs/maniskill-pi05-rlinf-e1a82ebf-20261028-r1/bin/python
TASK_RLINF=$TASK_OLD/third_party/rlinf-e1a82ebf-complete-heads-fd8ce503-20261028-r1
TASK_MODEL=$TASK_OLD/inputs/rlinf-pi05-maniskill-rl-a0cbb898-20261028-r1/checkpoints/global_step_150/actor
[[ $TASK_SOURCE =~ ^[0-9a-f]{40}$ ]]
[[ $TASK_RUN =~ ^native-vla-context-pilot-[a-z0-9-]+$ ]]
[[ $(hostname) =~ ^job-[a-z0-9]+-master-0$ ]]
test -n "${KUBERNETES_SERVICE_HOST:-}"
cd "$TASK_WORKTREE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
mkdir "$TASK_OUTPUT"
exec > >(tee "$TASK_OUTPUT/console.log") 2>&1
trap 'TASK_EXIT=$?; printf "%s\n" "$TASK_EXIT" > "$TASK_OUTPUT/exit-code.txt"' EXIT
export PYTHONPATH="$TASK_WORKTREE/src" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export FLUFFY_GPU_EXECUTION=baige CUDA_VISIBLE_DEVICES=0
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
"$TASK_PYTHON" 'scripts/[train]gpu_probe.py' --expected-count 1 --purpose training \
  --baige-container --output "$TASK_OUTPUT/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP="$TASK_OUTPUT/gpu-probe.json"
TASK_GPU_UUID=$("$TASK_PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["status"]=="passed" and len(p["devices"])==1; print(p["devices"][0]["uuid"])' "$TASK_OUTPUT/gpu-probe.json")
export LYCO_BAIGE_ISOLATED_UUID="$TASK_GPU_UUID"
CUDA_VISIBLE_DEVICES='' timeout 180 "$TASK_PYTHON" -m pytest -q -p no:cacheprovider \
  tests/test_pi05_context_policy.py > "$TASK_OUTPUT/qualification.log" 2>&1
timeout 180 "$TASK_PYTHON" -m pytest -q -p no:cacheprovider \
  tests/test_pi05_context_policy.py -k collector >> "$TASK_OUTPUT/qualification.log" 2>&1
"$TASK_PYTHON" -m fluffy_bassoon.meta_rl.pi05_context_pilot \
  --source-commit "$TASK_SOURCE" --rlinf-root "$TASK_RLINF" --model-path "$TASK_MODEL" \
  --norm-stats "$TASK_MODEL/assets/global_step_150/meta/norm_stats.json" \
  --expected-model-sha256 0adb5b40f064760ff245f53ae0a4e078b86680b65f77d1f5991a173543edd5a1 \
  --allocation-probe "$TASK_OUTPUT/gpu-probe.json" --expected-gpu-uuid "$TASK_GPU_UUID" \
  --rank 4 --gate-fit-steps 12 --updates 20 --seed 0 --threads 4 \
  --max-elapsed-seconds 2700 --output-dir "$TASK_OUTPUT/execution"
