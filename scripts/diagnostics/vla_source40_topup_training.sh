#!/usr/bin/env bash
# Existing authorized allocation only. Immutable dataset generation; no scheduling.
set -Eeuo pipefail
TASK_STARTED=$(date +%s)
TASK_SOURCE=${1:?reviewed student-training commit}
TASK_RUN=${2:?fresh dataset-generation/training identity}
TASK_PLAN=${3:?immutable topup source plan}
TASK_PLAN_SHA=${4:?explicit plan file SHA256}
TASK_LEDGER=${5:?canonical live topup ledger}
TASK_SLICE=${6:?allocated slice seconds excluding60startup allowance}
TASK_WORKER=${7:?cumulative training-worker seconds inclsetup/retries}
TASK_RESUME=${8:-}
case "$TASK_RESUME" in ''|--resume) ;; *) exit 2;; esac
TASK_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_OLD=/mowenjie/fluffy-bassoon
TASK_OUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_MODEL=$TASK_OLD/envs/openpi-215abfb/bin/python
TASK_PROBE=$TASK_OLD/envs/lerobot-v044-8fff0fde/bin/python
TASK_TEST=$TASK_ROOT/envs/pytest-overlay-20261029-r1
TASK_CUDA=$TASK_ROOT/outputs/libero-pi05-student-cuda-20261029-r1/cu128-overlay
cd "$TASK_CODE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
test "$(sha256sum "$TASK_PLAN" | cut -d' ' -f1)" = "$TASK_PLAN_SHA"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export PYTHONPATH=$TASK_CODE/src
TASK_ATTEMPT=$("$TASK_MODEL" - "$TASK_OUT" "$TASK_SOURCE" "$TASK_RUN" "$TASK_PLAN_SHA" \
  "$TASK_LEDGER" "$TASK_SLICE" "$TASK_WORKER" "$TASK_STARTED" "$TASK_RESUME" <<'PYTIME'
import sys
from fluffy_bassoon.meta_rl.vla_source40_ppo_worker import begin
output, source, run, plan_sha, ledger, seconds, worker, started, resume = sys.argv[1:]
assert int(worker) <= 3000
contract=dict(operation='source40_topup_immutable_prefix_fitting',source_commit=source,run_id=run,
    plan_file_sha256=plan_sha,canonical_ledger_path=ledger,scales=[25,100,500],seed=0,
    optimizer=dict(batch_size=8,learning_rate=3e-4,width=128,layers=2,heads=4,max_window=128,context_length=300))
print(begin(output,contract,slice_seconds=int(seconds),worker_seconds=int(worker),
    resume=resume=='--resume',now=float(started)))
PYTIME
)
exec > >(tee "$TASK_ATTEMPT/console.log") 2>&1
task_cleanup() {
  TASK_RC=$?
  trap - EXIT
  PYTHONPATH=$TASK_CODE/src "$TASK_MODEL" -m fluffy_bassoon.meta_rl.vla_source40_ppo_worker finish \
    --root "$TASK_OUT" --attempt "$TASK_ATTEMPT" --exit-code "$TASK_RC" || true
  printf '%s\n' "$TASK_RC" > "$TASK_OUT/exit-code.txt"
  exit "$TASK_RC"
}
trap task_cleanup EXIT
exec 9> "$TASK_OUT/launcher.lock"
flock -n 9
export FLUFFY_GPU_EXECUTION=baige OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
"$TASK_PROBE" 'scripts/[train]gpu_probe.py' --expected-count 1 --purpose training \
  --baige-container --baige-exclusive-training --output "$TASK_ATTEMPT/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP=$TASK_ATTEMPT/gpu-probe.json
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$TASK_TEST:$TASK_CODE/src \
  "$TASK_PROBE" -m pytest -q -p no:cacheprovider tests/test_vla_source40_topup_assembly.py \
  tests/test_vla_source40_training.py tests/test_vla_source40_subset_training.py \
  > "$TASK_ATTEMPT/focused-tests.log" 2>&1
export PYTHONPATH=$TASK_CUDA:$TASK_CODE/src
if [ ! -f "$TASK_OUT/snapshot/admission.json" ]; then
  test ! -e "$TASK_OUT/experiment"
  if [ -e "$TASK_OUT/snapshot" ]; then mv "$TASK_OUT/snapshot" "$TASK_OUT/snapshot-uncommitted-$(date +%s%N)"; fi
  "$TASK_MODEL" -m fluffy_bassoon.meta_rl.vla_source40_topup_assembly capture \
    --plan "$TASK_PLAN" --plan-sha256 "$TASK_PLAN_SHA" --ledger "$TASK_LEDGER" --output "$TASK_OUT/snapshot"
fi
TASK_ADMISSION=$TASK_OUT/snapshot/admission.json
TASK_ADMISSION_SHA=$(sha256sum "$TASK_ADMISSION" | cut -d' ' -f1)
if [ ! -f "$TASK_OUT/assembly/topup-inputs.json" ]; then
  test ! -e "$TASK_OUT/experiment"
  if [ -e "$TASK_OUT/assembly" ]; then mv "$TASK_OUT/assembly" "$TASK_OUT/assembly-uncommitted-$(date +%s%N)"; fi
  "$TASK_MODEL" -m fluffy_bassoon.meta_rl.vla_source40_topup_assembly assemble \
    --admission "$TASK_ADMISSION" --admission-sha256 "$TASK_ADMISSION_SHA" --output "$TASK_OUT/assembly"
fi
TASK_MANIFEST=$TASK_OUT/assembly/topup-inputs.json
TASK_MANIFEST_SHA=$(sha256sum "$TASK_MANIFEST" | cut -d' ' -f1)
TASK_REMAINING=$((TASK_STARTED + TASK_SLICE - $(date +%s) - 30))
test "$TASK_REMAINING" -gt 120
TASK_ARGS=()
if [ "$TASK_RESUME" = --resume ] && [ -d "$TASK_OUT/experiment" ]; then TASK_ARGS=(--resume); fi
timeout --signal=INT --kill-after=20s "${TASK_REMAINING}s" "$TASK_MODEL" \
  -m fluffy_bassoon.meta_rl.vla_source40_topup_assembly train \
  --manifest "$TASK_MANIFEST" --manifest-sha256 "$TASK_MANIFEST_SHA" --output "$TASK_OUT/experiment" \
  --source-commit "$TASK_SOURCE" --expected-hostname "$(hostname)" "${TASK_ARGS[@]}"
