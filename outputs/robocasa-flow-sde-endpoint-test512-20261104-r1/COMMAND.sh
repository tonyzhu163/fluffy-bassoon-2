#!/usr/bin/env bash
# One scheduler GPU per task (TASK_NAME = CloseToasterOvenDoor or OpenDrawer).
# Start the staging bridge first on the dev host with the BOS mount:
#   python endpoint.py stage --prepared $RUN/PREPARED.json --task $TASK_NAME \
#     --staging $OUT/$TASK_NAME/staging-r1 --output $OUT/$TASK_NAME/gpu-r1 --deadline <unix>
# After both tasks finish, on CPU:
#   python endpoint.py combine --prepared $RUN/PREPARED.json --outputs $OUT --destination $OUT/HOLM.json
set -Eeuo pipefail
: "${TASK_META:?}" "${TASK_CODE_PIN:?}" "${TASK_NAME:?}"
TASK_ROOT=/mowenjie/fluffy-bassoon-2
RUN=$TASK_META/outputs/robocasa-flow-sde-endpoint-test512-20261104-r1
OUT=$TASK_ROOT/outputs/robocasa-flow-sde-endpoint-test512-20261104-r1
TASK_OUTPUT=$OUT/$TASK_NAME/gpu-r1
TASK_STAGING=$OUT/$TASK_NAME/staging-r1
TASK_RUNTIME=$TASK_ROOT/sources/flow10m-0988b5f9
TASK_PY=$TASK_ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
# Reuse the qualified eval-only workers without modifying their run.
TASK_EVAL=$TASK_META/outputs/robocasa-flow-sde-checkpoint-selection-20261104-r1
TASK_STARTED=$(date +%s)
test ! -e "$TASK_OUTPUT"
mkdir -p "$TASK_OUTPUT"
exec > >(tee "$TASK_OUTPUT/console.log") 2>&1
finish() {
 TASK_RC=$?
 trap - EXIT
 TASK_FINISHED=$(date +%s)
 printf '{"started_unix":%s,"finished_unix":%s,"allocated_gpus":1,"observed_gpu_seconds":%s,"exit_code":%s}\n' \
  "$TASK_STARTED" "$TASK_FINISHED" "$((TASK_FINISHED-TASK_STARTED))" "$TASK_RC" > "$TASK_OUTPUT/clock.json"
 exit "$TASK_RC"
}
trap finish EXIT
test "$(git -C "$TASK_META" rev-parse HEAD)" = "$TASK_CODE_PIN"
test -z "$(git -C "$TASK_META" status --porcelain)"
test "$(git -C "$TASK_RUNTIME" rev-parse HEAD)" = 0988b5f9e9bf31ac3db673cc2860959611b9ae12
test -z "$(git -C "$TASK_RUNTIME" status --porcelain)"
TASK_RLINF=$TASK_ROOT/sources/robocasa365-public-pi05-nnx-leaves-20261101-r1/RLinf
test "$(git -C "$TASK_RLINF" rev-parse HEAD)" = 8a1233c3fae6f5905adfda7684c455842b0ecfb8
test -z "$(git -C "$TASK_RLINF" status --porcelain)"
export PYTHONPATH=$RUN:$TASK_EVAL:$TASK_RUNTIME/src:$TASK_RLINF
export OPENPI_DATA_HOME=$TASK_ROOT/setups/robocasa365-native-20261101-r1/cache/.cache/openpi
export NCCL_P2P_DISABLE=1 FLUFFY_GPU_EXECUTION=baige
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
"$TASK_PY" -B "$TASK_RUNTIME/scripts/[train]gpu_probe.py" --expected-count 1 --purpose evaluation \
 --baige-container --output "$TASK_OUTPUT/gpu-probe.json"
# 128 banks x ~68 s at the slowest qualified rate + ~64 reloads: ~2.8 h. 230-minute job.
TASK_DEADLINE=$((TASK_STARTED+13500))
timeout --signal=TERM --kill-after=30s 13590 "$TASK_PY" -B "$RUN/endpoint.py" run \
 --prepared "$RUN/PREPARED.json" --task "$TASK_NAME" --staging "$TASK_STAGING" \
 --output "$TASK_OUTPUT" --deadline "$TASK_DEADLINE"
