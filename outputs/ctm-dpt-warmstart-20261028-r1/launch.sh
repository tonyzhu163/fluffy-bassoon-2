#!/bin/bash
# Bounded first candidates. Source SHA and fresh run ID are launch arguments.
set -euo pipefail
TASK_SOURCE=${1:?reviewed source commit}
TASK_RUN=${2:?fresh run identity}
TASK_MODE=${3:-all}
[[ $TASK_MODE = all || $TASK_MODE = native ]]
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_OLD=/mowenjie/fluffy-bassoon
TASK_WORKTREE=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUTPUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_PYTHON=$TASK_OLD/envs/maniskill-pi05-rlinf-e1a82ebf-20261028-r1/bin/python
TASK_RLINF=$TASK_OLD/third_party/rlinf-e1a82ebf-complete-heads-fd8ce503-20261028-r1
TASK_MODEL=$TASK_OLD/inputs/rlinf-pi05-maniskill-rl-a0cbb898-20261028-r1/checkpoints/global_step_150/actor
TASK_MODEL_SHA=0adb5b40f064760ff245f53ae0a4e078b86680b65f77d1f5991a173543edd5a1
TASK_CACHE=$TASK_OLD/outputs/maniskill-pi05-baige-gates-20261028-r7/execution/headroom-rl/outcomes.jsonl
TASK_CACHE_SHA=74910dc755b959fe86ec0a3d4cf1c05957ba57510c64bbb1bb2c85ba56e41575
[[ $TASK_SOURCE =~ ^[0-9a-f]{40}$ ]]
[[ $TASK_RUN =~ ^ctm-dpt-warmstart-[a-z0-9-]+$ ]]
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
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
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

# Repeat the relevant suite on the final main commit, then actual CLI smoke.
CUDA_VISIBLE_DEVICES='' timeout 480 "$TASK_PYTHON" -m pytest -q -p no:cacheprovider \
  tests/test_benchmark_warmstart.py tests/test_meta_rl_checkpoint.py tests/test_meta_rl_train.py \
  tests/test_maniskill_rotation_poc.py tests/test_pi05_bridge_train.py tests/test_pi05_bridge_policy.py \
  tests/test_pi05_feature_bridge.py > "$TASK_OUTPUT/qualification.log" 2>&1

run_compact() {
  local task=$1
  local extra=()
  local eval_trials=32
  if test "$task" = cached_rotation; then extra=(--cache "$TASK_CACHE" --cache-sha256 "$TASK_CACHE_SHA"); fi
  if test "$task" = darkroom || test "$task" = keydoor; then eval_trials=8; fi
  CUDA_VISIBLE_DEVICES='' "$TASK_PYTHON" -m fluffy_bassoon.meta_rl.warmstart_train \
    --task "$task" --core persistent_ctm --width 64 --ticks 4 --attempts 8 \
    --train-trials 256 --validation-trials 16 --batch-trials 8 --updates 200 \
    --eval-every 200 --eval-trials "$eval_trials" --max-elapsed-seconds 300 --threads 2 \
    --source-commit "$TASK_SOURCE" --expected-hostname "$(hostname)" --seed 0 \
    --output-dir "$TASK_OUTPUT/compact-$task" "${extra[@]}" > "$TASK_OUTPUT/compact-$task.log" 2>&1
}

run_native() {
  local common=(--source-commit "$TASK_SOURCE" --rlinf-root "$TASK_RLINF" --model-path "$TASK_MODEL"
    --norm-stats "$TASK_MODEL/assets/global_step_150/meta/norm_stats.json" --expected-model-sha256 "$TASK_MODEL_SHA"
    --allocation-probe "$TASK_OUTPUT/gpu-probe.json" --expected-gpu-uuid "$TASK_GPU_UUID"
    --core persistent_ctm --width 64 --ticks 4 --num-envs 2 --attempts 4 --objects 0 1
    --noise-scale .05 --residual-scale 1 --threads 4 --seed 0)
  "$TASK_PYTHON" -m fluffy_bassoon.meta_rl.pi05_bridge_train "${common[@]}" \
    --collect-dpt --updates 2 --eval-at 2 --eval-batches 2 --max-elapsed-seconds 1800 \
    --output-dir "$TASK_OUTPUT/native-collection" > "$TASK_OUTPUT/native-collection.log" 2>&1
  # This is a small competence gate, not a significance test or adaptation claim.
  if "$TASK_PYTHON" - "$TASK_OUTPUT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
summary = json.loads((root / 'native-collection/summary.json').read_text())
evaluation = summary['last_evaluation']
normal = evaluation['normal']['mean_success'] if evaluation else None
oracle = evaluation['oracle_label_policy']['mean_success'] if evaluation else None
passed = bool(summary['iteration'] == 2 and oracle is not None and oracle >= .25 and oracle >= normal)
(root / 'native-label-gate.json').write_text(json.dumps(dict(passed=passed, baseline=normal, oracle=oracle,
    criterion='two complete history batches; paired oracle success >= 0.25 and >= frozen baseline',
    interpretation='small pilot competence check, not statistical evidence of superiority'), indent=2) + '\n')
sys.exit(0 if passed else 3)
PY
  then
    local manifest=$TASK_OUTPUT/native-collection/histories/manifest.json
    local manifest_sha
    manifest_sha=$(sha256sum "$manifest" | cut -d' ' -f1)
    "$TASK_PYTHON" -m fluffy_bassoon.meta_rl.pi05_warmstart_train \
      --histories "$manifest" --histories-sha256 "$manifest_sha" --core persistent_ctm --width 64 --ticks 4 \
      --updates 100 --eval-every 50 --max-elapsed-seconds 300 --threads 4 --seed 0 \
      --source-commit "$TASK_SOURCE" --expected-hostname "$(hostname)" \
      --output-dir "$TASK_OUTPUT/native-warmstart" > "$TASK_OUTPUT/native-warmstart.log" 2>&1
    local warm_sha
    warm_sha=$(sha256sum "$TASK_OUTPUT/native-warmstart/warmstart.pt" | cut -d' ' -f1)
    # Frozen candidate evaluation at iteration zero, followed by one PPO update
    # to qualify the actual initializer -> RL transition. Keep both cost ledgers.
    "$TASK_PYTHON" -m fluffy_bassoon.meta_rl.pi05_bridge_train "${common[@]}" \
      --warmstart "$TASK_OUTPUT/native-warmstart/warmstart.pt" --warmstart-sha256 "$warm_sha" \
      --updates 1 --eval-at 0 --eval-batches 1 --max-elapsed-seconds 900 --target-kl .05 \
      --output-dir "$TASK_OUTPUT/native-ppo-smoke" > "$TASK_OUTPUT/native-ppo-smoke.log" 2>&1
  else
    echo 'Native label competence gate did not pass; no native warm start was trained.'
  fi
}

TASK_PIDS=()
TASK_NAMES=()
if test "$TASK_MODE" = all; then
for TASK_GROUP in 'bandit bernoulli nonstationary' 'reach darkroom' 'keydoor cached_rotation'; do
  (for TASK_BENCH in $TASK_GROUP; do run_compact "$TASK_BENCH"; done) &
  TASK_PIDS+=("$!")
  TASK_NAMES+=("$TASK_GROUP")
done
fi
run_native &
TASK_PIDS+=("$!")
TASK_NAMES+=(native)
printf 'name\tpid\n' > "$TASK_OUTPUT/worker-pids.tsv"
for TASK_INDEX in "${!TASK_PIDS[@]}"; do
  printf '%s\t%s\n' "${TASK_NAMES[$TASK_INDEX]}" "${TASK_PIDS[$TASK_INDEX]}" >> "$TASK_OUTPUT/worker-pids.tsv"
done
TASK_FAILED=0
printf 'name\tpid\texit_code\n' > "$TASK_OUTPUT/worker-exit-codes.tsv"
for TASK_INDEX in "${!TASK_PIDS[@]}"; do
  if wait "${TASK_PIDS[$TASK_INDEX]}"; then TASK_CHILD_EXIT=0; else TASK_CHILD_EXIT=$?; TASK_FAILED=1; fi
  printf '%s\t%s\t%s\n' "${TASK_NAMES[$TASK_INDEX]}" "${TASK_PIDS[$TASK_INDEX]}" "$TASK_CHILD_EXIT" >> "$TASK_OUTPUT/worker-exit-codes.tsv"
done
exit "$TASK_FAILED"
