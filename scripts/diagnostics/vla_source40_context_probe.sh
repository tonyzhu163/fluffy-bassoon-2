#!/usr/bin/env bash
# Offline model-only context probe: <=one allocated GPU, <=10 minutes, zero simulator actions.
set -Eeuo pipefail
TASK_SOURCE=${1:?exact reviewed probe commit}
TASK_RUN=${2:?fresh probe output identity}
TASK_TRAIN_RUN=${3:?completed subset-scale training identity}
TASK_EVAL_RUN=${4:?completed target evaluation identity}
TASK_GROUP=${5:-group-008}
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_OLD=/mowenjie/fluffy-bassoon
TASK_WT=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_SCALES=$TASK_ROOT/outputs/$TASK_TRAIN_RUN/experiment/scales.json
TASK_GROUP_INDEX=$TASK_ROOT/outputs/$TASK_EVAL_RUN/experiment/$TASK_GROUP/index.json
TASK_MODEL=$TASK_OLD/envs/openpi-215abfb/bin/python
TASK_PROBE=$TASK_OLD/envs/lerobot-v044-8fff0fde/bin/python
TASK_CUDA=$TASK_ROOT/outputs/libero-pi05-student-cuda-20261029-r1/cu128-overlay
TASK_TEST=$TASK_ROOT/envs/pytest-overlay-20261029-r1
case "$TASK_RUN/$TASK_TRAIN_RUN/$TASK_EVAL_RUN/$TASK_GROUP" in *'..'*|/*) exit 2;; esac
test ! -e "$TASK_OUT"
test -f "$TASK_SCALES"
test -f "$TASK_GROUP_INDEX"
mkdir -p "$TASK_OUT"
cd "$TASK_WT"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
exec > >(tee "$TASK_OUT/console.log") 2>&1
task_cleanup() {
  TASK_RC=$?
  trap - EXIT
  printf '%s\n' "$TASK_RC" > "$TASK_OUT/exit-code.txt"
  exit "$TASK_RC"
}
trap task_cleanup EXIT
export FLUFFY_GPU_EXECUTION=baige CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONPATH=$TASK_WT/src
"$TASK_PROBE" 'scripts/[train]gpu_probe.py' --expected-count 1 --purpose training \
  --baige-container --baige-exclusive-training --output "$TASK_OUT/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP=$TASK_OUT/gpu-probe.json
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$TASK_TEST:$TASK_WT/src \
  "$TASK_PROBE" -m pytest -q -p no:cacheprovider tests/test_vla_source40_context_probe.py \
  > "$TASK_OUT/focused-tests.log" 2>&1
TASK_SCALES_SHA=$(sha256sum "$TASK_SCALES" | cut -d' ' -f1)
TASK_GROUP_SHA=$(sha256sum "$TASK_GROUP_INDEX" | cut -d' ' -f1)
export PYTHONPATH=$TASK_CUDA:$TASK_WT/src
timeout --signal=INT --kill-after=15s 480 "$TASK_MODEL" \
  -m fluffy_bassoon.meta_rl.vla_source40_context_probe \
  --scales "$TASK_SCALES" --scales-sha256 "$TASK_SCALES_SHA" \
  --group-index "$TASK_GROUP_INDEX" --group-index-sha256 "$TASK_GROUP_SHA" \
  --output "$TASK_OUT/experiment" --source-commit "$TASK_SOURCE" \
  --expected-hostname "$(hostname)" --device cuda:0 --max-elapsed-seconds 420
