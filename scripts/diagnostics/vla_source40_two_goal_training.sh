#!/usr/bin/env bash
# Existing one-GPU allocation, <=50min; two source goals, no new robot actions.
set -Eeuo pipefail
TASK_SOURCE=${1:?exact reviewed integrated commit}
TASK_RUN=${2:?fresh two-goal training identity}
TASK_LV_SHA=${3:?independently verified completed L00-validation index SHA256}
TASK_RESUME=${4:-}
case "$TASK_RESUME" in ''|--resume) ;; *) exit 2;; esac
[[ "$TASK_LV_SHA" =~ ^[0-9a-f]{64}$ ]]
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_OLD=/mowenjie/fluffy-bassoon
TASK_WT=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_MODEL=$TASK_OLD/envs/openpi-215abfb/bin/python
TASK_PROBE=$TASK_OLD/envs/lerobot-v044-8fff0fde/bin/python
TASK_CUDA=$TASK_ROOT/outputs/libero-pi05-student-cuda-20261029-r1/cu128-overlay
TASK_TEST=$TASK_ROOT/envs/pytest-overlay-20261029-r1
TASK_ST=$TASK_ROOT/outputs/libero40-1m-sac-20261029-s00-train-r1/experiment/index.json
TASK_SV=$TASK_ROOT/outputs/libero40-pi05-source-smoke-20261029-s00-validation-r1/experiment/index.json
TASK_LT=$TASK_ROOT/outputs/libero40-1m-sac-20261029-l00-train-r1/experiment/index.json
TASK_LV=$TASK_ROOT/outputs/libero40-1m-sac-20261030-l00-validation-r2/experiment/index.json
TASK_ST_SHA=21836bacb78aa9c9d8b5fc7dab307c2c8e38ff7af535bc78b8d8dab767113fbd
TASK_LT_SHA=9eb4c3716d0c71b2f07580d146a879ff6756d4639b75c0751a3163c3d16f5f93
TASK_SV_SHA=976f969cc724d49ecb932b77933f0b1513b8cb6500c15a9242f012e8336b5ba7
test "$(cat "$TASK_ROOT/outputs/libero40-1m-sac-20261030-l00-validation-r2/exit-code.txt")" = 0
test "$(sha256sum "$TASK_ST" | cut -d' ' -f1)" = "$TASK_ST_SHA"
test "$(sha256sum "$TASK_SV" | cut -d' ' -f1)" = "$TASK_SV_SHA"
test "$(sha256sum "$TASK_LT" | cut -d' ' -f1)" = "$TASK_LT_SHA"
test "$(sha256sum "$TASK_LV" | cut -d' ' -f1)" = "$TASK_LV_SHA"
TASK_ARGS=()
if [ "$TASK_RESUME" = --resume ]; then
  test -f "$TASK_OUT/experiment/contract.json"
  TASK_ARGS=(--resume)
else
  test ! -e "$TASK_OUT"
  mkdir -p "$TASK_OUT"
fi
cd "$TASK_WT"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
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
  "$TASK_PROBE" -m pytest -q -p no:cacheprovider tests/test_vla_source40_two_goal_qualification.py \
  tests/test_vla_source40_subset_training.py tests/test_vla_source40_training.py \
  tests/test_vla_source40_data_scales.py tests/test_vla_source40_campaign.py tests/test_vla_history.py tests/test_gpu_probe.py \
  > "$TASK_ATTEMPT/focused-tests.log" 2>&1
export PYTHONPATH=$TASK_CUDA:$TASK_WT/src
if [ "$TASK_RESUME" != --resume ]; then
  "$TASK_MODEL" -m fluffy_bassoon.meta_rl.vla_source40_two_goal_qualification \
    --qualification-subset --train-index "$TASK_ST" "$TASK_LT" \
    --train-sha256 "$TASK_ST_SHA" "$TASK_LT_SHA" \
    --validation-index "$TASK_SV" "$TASK_LV" --validation-sha256 "$TASK_SV_SHA" "$TASK_LV_SHA" \
    --output-dir "$TASK_OUT/assembly" > "$TASK_ATTEMPT/qualification.json"
fi
TASK_ASSEMBLY_SHA=$("$TASK_MODEL" - "$TASK_OUT/assembly/two-goal-qualification.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['assembly_file_sha256'])
PY
)
timeout --signal=INT --kill-after=30s 2820 "$TASK_MODEL" \
  -m fluffy_bassoon.meta_rl.vla_source40_subset_training --qualification-subset \
  --train-index "$TASK_ST" "$TASK_LT" --validation-index "$TASK_SV" "$TASK_LV" \
  --assembly "$TASK_OUT/assembly/assembly.json" --assembly-file-sha256 "$TASK_ASSEMBLY_SHA" \
  --output "$TASK_OUT/experiment" --scales 25 100 500 --seed 0 \
  --source-commit "$TASK_SOURCE" --expected-hostname "$(hostname)" \
  --max-controller-seconds 600 --max-total-seconds 2700 "${TASK_ARGS[@]}"
