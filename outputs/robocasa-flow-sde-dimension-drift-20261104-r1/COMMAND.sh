#!/usr/bin/env bash
# One-GPU Baige job. Start the CPU stage bridge first on the dev host (BOS mount):
#   python3 $TASK_META/outputs/robocasa-flow-sde-dimension-drift-20261104-r1/DIMENSION-DRIFT.py stage \
#     --root $SRC --output $OUT/staging --result-root $OUT/gpu
# and the CPU metrics join (any Linux CPU, no GPU):
#   python3 .../DIMENSION-DRIFT.py metrics --root $SRC --output $OUT/metrics
set -Eeuo pipefail
: "${TASK_META:?}" "${TASK_CODE_PIN:?}"
TASK_ROOT=/mowenjie/fluffy-bassoon-2
SRC=$TASK_ROOT/outputs/robocasa-flow-sde-10m-per-task-20261103-r1
OUT=$TASK_ROOT/outputs/robocasa-flow-sde-dimension-drift-20261104-r1
TASK_OUTPUT=$OUT/gpu
TASK_RUNTIME=$TASK_ROOT/sources/flow10m-0988b5f9
TASK_PY=$TASK_ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
TASK_SCRIPT=$TASK_META/outputs/robocasa-flow-sde-dimension-drift-20261104-r1/DIMENSION-DRIFT.py
export TASK_STARTED=$(date +%s)
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
export PYTHONPATH=$TASK_RUNTIME/src:$TASK_ROOT/sources/robocasa365-public-pi05-nnx-leaves-20261101-r1/RLinf
export OPENPI_DATA_HOME=$TASK_ROOT/setups/robocasa365-native-20261101-r1/cache/.cache/openpi
export NCCL_P2P_DISABLE=1 FLUFFY_GPU_EXECUTION=baige
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
"$TASK_PY" -B "$TASK_RUNTIME/scripts/[train]gpu_probe.py" --expected-count 1 --purpose training \
 --baige-container --baige-exclusive-training --output "$TASK_OUTPUT/gpu-probe.json"
# R3 took 4285 s for the same 80 forwards; one shared deadline inside a 180-minute job.
TASK_DEADLINE=$((TASK_STARTED+10200))
# Drawer→Counter first, then the gripper-using control (CoffeeSetupMug), then the rest.
# Must match ORDER in DIMENSION-DRIFT.py (the stage bridge walks the same order).
for TASK_NAME in PickPlaceDrawerToCounter CoffeeSetupMug TurnOffStove OpenDrawer CloseToasterOvenDoor; do
 TASK_REMAINING=$((TASK_DEADLINE-$(date +%s)))
 test "$TASK_REMAINING" -gt 60
 timeout --signal=TERM --kill-after=30s "$TASK_REMAINING" "$TASK_PY" -B "$TASK_SCRIPT" run \
  --prepared "$SRC/offline-initial-ratio-inputs-r2/PREPARED.json" \
  --reference "$SRC/offline-initial-ratio-gpu-r3" \
  --task "$TASK_NAME" --output "$TASK_OUTPUT/$TASK_NAME" --staging "$OUT/staging"
done
