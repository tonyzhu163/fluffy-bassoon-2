#!/usr/bin/env bash
# One GPU, read-only production replay; CPU staging bridge starts separately.
set -Eeuo pipefail
: "${TASK_META:?}" "${TASK_CODE_PIN:?}"
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_OUT=$TASK_ROOT/outputs/robocasa-flow-sde-critic-parity-20261104-r1
TASK_RUNTIME=$TASK_ROOT/sources/flow10m-0988b5f9
TASK_VENDOR=$TASK_ROOT/sources/robocasa365-public-pi05-nnx-leaves-20261101-r1/RLinf
TASK_PY=$TASK_ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
TASK_SCRIPT=$TASK_META/outputs/robocasa-flow-sde-critic-parity-20261104-r1/REPLAY.py
TASK_STARTED=$(date +%s)
test ! -e "$TASK_OUT/gpu"
mkdir "$TASK_OUT/gpu"
exec > >(tee "$TASK_OUT/gpu/console.log") 2>&1
finish() {
 TASK_RC=$?
 trap - EXIT
 TASK_FINISHED=$(date +%s)
 printf '{"started_unix":%s,"finished_unix":%s,"allocated_gpus":1,"observed_gpu_seconds":%s,"exit_code":%s}\n' \
  "$TASK_STARTED" "$TASK_FINISHED" "$((TASK_FINISHED-TASK_STARTED))" "$TASK_RC" > "$TASK_OUT/gpu/clock.json"
 exit "$TASK_RC"
}
trap finish EXIT
test "$(git -C "$TASK_META" rev-parse HEAD)" = "$TASK_CODE_PIN"
test -z "$(git -C "$TASK_META" status --porcelain)"
test "$(git -C "$TASK_RUNTIME" rev-parse HEAD)" = 0988b5f9e9bf31ac3db673cc2860959611b9ae12
test -z "$(git -C "$TASK_RUNTIME" status --porcelain)"
test "$(git -C "$TASK_VENDOR" rev-parse HEAD)" = 8a1233c3fae6f5905adfda7684c455842b0ecfb8
test -z "$(git -C "$TASK_VENDOR" status --porcelain)"
export PYTHONPATH=$TASK_RUNTIME/src:$TASK_VENDOR
export OPENPI_DATA_HOME=$TASK_ROOT/setups/robocasa365-native-20261101-r1/cache/.cache/openpi
export NCCL_P2P_DISABLE=1 FLUFFY_GPU_EXECUTION=baige
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
"$TASK_PY" -B "$TASK_RUNTIME/scripts/[train]gpu_probe.py" --expected-count 1 --purpose training \
 --baige-container --baige-exclusive-training --output "$TASK_OUT/gpu/gpu-probe.json"
TASK_REMAINING=$((TASK_STARTED+2520-$(date +%s)))
test "$TASK_REMAINING" -gt 60
timeout --signal=TERM --kill-after=30s "$TASK_REMAINING" "$TASK_PY" -B "$TASK_SCRIPT" run \
 --prepared "$TASK_OUT/fixtures-r1/PREPARED.json" --output "$TASK_OUT/gpu/replay" --staging "$TASK_OUT/staging-r1"
