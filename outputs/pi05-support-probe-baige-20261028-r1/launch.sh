#!/bin/bash
# One exclusive scheduler GPU; no model work is allowed on the development host.
set -euo pipefail
TASK_SOURCE=${1:?exact source commit}
TASK_RUN=${2:?fresh run identity}
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_WORKTREE=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUTPUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_OLD=/mowenjie/fluffy-bassoon
TASK_PYTHON=$TASK_OLD/envs/maniskill-pi05-rlinf-e1a82ebf-20261028-r1/bin/python
TASK_RLINF=$TASK_OLD/third_party/rlinf-e1a82ebf-complete-heads-fd8ce503-20261028-r1
TASK_MODEL=$TASK_OLD/inputs/rlinf-pi05-maniskill-rl-a0cbb898-20261028-r1/checkpoints/global_step_150/actor
[[ $TASK_SOURCE =~ ^[0-9a-f]{40}$ ]]
[[ $TASK_RUN =~ ^pi05-support-probe-baige-20261028-r[0-9]+$ ]]
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
test -d /root/.sapien/physx/105.1-physx-5.3.1.patch0
hostname
date -u
printf '%s  %s\n' 01b8a68a0085cafb64ce3b0716732305e78921086b948654c2a4ce9edc5b71c9 "$TASK_RLINF/rlinf/models/embodiment/openpi/checkpoint.py" | sha256sum -c -
"$TASK_PYTHON" 'scripts/[train]gpu_probe.py' --expected-count 1 --purpose training \
  --baige-container --output "$TASK_OUTPUT/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP="$TASK_OUTPUT/gpu-probe.json"
TASK_UUID=$("$TASK_PYTHON" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"]=="passed" and len(r["devices"])==1; print(r["devices"][0]["uuid"])' "$TASK_OUTPUT/gpu-probe.json")
export LYCO_BAIGE_ISOLATED_UUID="$TASK_UUID"
CUDA_VISIBLE_DEVICES='' "$TASK_PYTHON" -m pytest -q -p no:cacheprovider \
  tests/test_pi05_feature_bridge.py tests/test_pi05_bridge_policy.py \
  tests/test_pi05_bridge_train.py tests/test_benchmark_warmstart.py \
  tests/test_pi05_interface_gate.py tests/test_pi05_balanced_plan.py \
  tests/test_pi05_fresh_probe.py tests/test_pi05_support_probe_run.py \
  > "$TASK_OUTPUT/qualification.log" 2>&1
"$TASK_PYTHON" -m fluffy_bassoon.meta_rl.pi05_support_probe_run \
  --source-commit "$TASK_SOURCE" --output-dir "$TASK_OUTPUT/execution" \
  --rlinf-root "$TASK_RLINF" --model-path "$TASK_MODEL" \
  --norm-stats "$TASK_MODEL/assets/global_step_150/meta/norm_stats.json" \
  --expected-model-sha256 0adb5b40f064760ff245f53ae0a4e078b86680b65f77d1f5991a173543edd5a1 \
  --allocation-probe "$TASK_OUTPUT/gpu-probe.json" --expected-gpu-uuid "$TASK_UUID" \
  --updates 100 --fit-seconds 600 --max-elapsed-seconds 4800 --threads 4 \
  --direct-ctm-checkpoint "$TASK_ROOT/inputs/$TASK_RUN/direct-ctm.pt" \
  --direct-ctm-sha256 "$(cut -d ' ' -f1 "$TASK_ROOT/inputs/$TASK_RUN/direct-ctm.sha256")" \
  --direct-xl-checkpoint "$TASK_ROOT/inputs/$TASK_RUN/direct-xl.pt" \
  --direct-xl-sha256 "$(cut -d ' ' -f1 "$TASK_ROOT/inputs/$TASK_RUN/direct-xl.sha256")"
