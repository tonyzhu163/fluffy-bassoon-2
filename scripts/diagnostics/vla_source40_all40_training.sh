#!/usr/bin/env bash
# Prepared offline stage only. Root must allocate/reassign its authorized GPU budget.
set -Eeuo pipefail
TASK_SOURCE=${1:?exact reviewed offline-training source}
TASK_RUN=${2:?fresh all40 offline run identity}
TASK_CAMPAIGN_SHA=${3:?explicit canonical campaign.json file SHA256}
TASK_LEDGER_SHA=${4:?explicit completed SAC ledger.json file SHA256}
TASK_RESUME=${5:-}
case "$TASK_RESUME" in ''|--resume) ;; *) exit 2;; esac
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_OLD=/mowenjie/fluffy-bassoon
TASK_WT=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_PLAN=$TASK_ROOT/outputs/libero40-source-1m-20261029-r1
TASK_MODEL=$TASK_OLD/envs/openpi-215abfb/bin/python
TASK_PROBE=$TASK_OLD/envs/lerobot-v044-8fff0fde/bin/python
TASK_CUDA=$TASK_ROOT/outputs/libero-pi05-student-cuda-20261029-r1/cu128-overlay
TASK_TEST=$TASK_ROOT/envs/pytest-overlay-20261029-r1
TASK_ARGS=()
if [ "$TASK_RESUME" = --resume ]; then
  test -f "$TASK_OUT/assembly/all40-inputs.json"
  test -f "$TASK_OUT/experiment/contract.json"
  TASK_ARGS=(--resume)
else
  test ! -e "$TASK_OUT"
  mkdir -p "$TASK_OUT"
fi
cd "$TASK_WT"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
test "$(sha256sum "$TASK_PLAN/campaign.json" | cut -d' ' -f1)" = "$TASK_CAMPAIGN_SHA"
test "$(sha256sum "$TASK_PLAN/ledger.json" | cut -d' ' -f1)" = "$TASK_LEDGER_SHA"
exec 9> "$TASK_OUT/launcher.lock"
flock -n 9
TASK_ATTEMPT=$TASK_OUT/attempts/$(date +%s%N)
mkdir -p "$TASK_ATTEMPT"
exec > >(tee "$TASK_ATTEMPT/console.log") 2>&1
task_cleanup() {
  TASK_RC=$?
  trap - EXIT
  printf '%s\n' "$TASK_RC" > "$TASK_ATTEMPT/exit-code.txt"
  printf '%s\n' "$TASK_RC" > "$TASK_OUT/exit-code.txt"
  exit "$TASK_RC"
}
trap task_cleanup EXIT
export FLUFFY_GPU_EXECUTION=baige CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONPATH=$TASK_WT/src
"$TASK_PROBE" 'scripts/[train]gpu_probe.py' --expected-count 1 --purpose training \
  --baige-container --baige-exclusive-training --output "$TASK_ATTEMPT/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP=$TASK_ATTEMPT/gpu-probe.json
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$TASK_TEST:$TASK_WT/src \
  "$TASK_PROBE" -m pytest -q -p no:cacheprovider \
  tests/test_vla_source40_offline_pipeline.py tests/test_vla_source40_subset_training.py \
  tests/test_vla_source40_training.py tests/test_vla_source40.py tests/test_vla_history.py tests/test_gpu_probe.py \
  > "$TASK_ATTEMPT/focused-tests.log" 2>&1
export PYTHONPATH=$TASK_CUDA:$TASK_WT/src
if [ "$TASK_RESUME" != --resume ]; then
  "$TASK_MODEL" -m fluffy_bassoon.meta_rl.vla_source40_offline_pipeline assemble \
    --project-root "$TASK_ROOT" --campaign "$TASK_PLAN/campaign.json" --campaign-sha256 "$TASK_CAMPAIGN_SHA" \
    --ledger "$TASK_PLAN/ledger.json" --ledger-sha256 "$TASK_LEDGER_SHA" --output "$TASK_OUT/assembly" \
    > "$TASK_ATTEMPT/assembly-receipt.json"
fi
TASK_MANIFEST=$TASK_OUT/assembly/all40-inputs.json
TASK_MANIFEST_SHA=$(sha256sum "$TASK_MANIFEST" | cut -d' ' -f1)
timeout --signal=INT --kill-after=30s 2820 "$TASK_MODEL" \
  -m fluffy_bassoon.meta_rl.vla_source40_offline_pipeline train \
  --manifest "$TASK_MANIFEST" --manifest-sha256 "$TASK_MANIFEST_SHA" --output "$TASK_OUT/experiment" \
  --source-commit "$TASK_SOURCE" --expected-hostname "$(hostname)" "${TASK_ARGS[@]}"
