#!/usr/bin/env bash
# One-GPU Baige job. Before it, on allocated Linux CPU:
#   python MATCHED-DRIFT.py prepare --root $SRC --output $OUT/prepared
# and on the dev host with the BOS mount, alongside the job:
#   python MATCHED-DRIFT.py stage --root $SRC --output $OUT/staging --result-root $OUT/gpu
set -Eeuo pipefail
: "${TASK_META:?}" "${TASK_CODE_PIN:?}"
TASK_ROOT=/mowenjie/fluffy-bassoon-2
SRC=$TASK_ROOT/outputs/robocasa-flow-sde-10m-per-task-20261103-r1
OUT=$TASK_ROOT/outputs/robocasa-flow-sde-matched-denoise-drift-20261104-r1
TASK_OUTPUT=$OUT/gpu
TASK_RUNTIME=$TASK_ROOT/sources/flow10m-0988b5f9
TASK_PY=$TASK_ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
TASK_SCRIPT=$TASK_META/outputs/robocasa-flow-sde-matched-denoise-drift-20261104-r1/MATCHED-DRIFT.py
export TASK_STARTED=$(date +%s)
test ! -e "$TASK_OUTPUT"
test -s "$OUT/prepared/PREPARED.json"
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
# 80 checkpoint loads (R3: 4285 s) plus up to 10 B16 forwards each; one deadline in a 240-minute job.
TASK_DEADLINE=$((TASK_STARTED+13800))
# Must match ORDER in MATCHED-DRIFT.py (the stage bridge walks the same order).
for TASK_NAME in PickPlaceDrawerToCounter OpenDrawer CoffeeSetupMug CloseToasterOvenDoor TurnOffStove; do
 TASK_REMAINING=$((TASK_DEADLINE-$(date +%s)))
 test "$TASK_REMAINING" -gt 60
 timeout --signal=TERM --kill-after=30s "$TASK_REMAINING" "$TASK_PY" -B "$TASK_SCRIPT" run \
  --prepared "$OUT/prepared/PREPARED.json" \
  --task "$TASK_NAME" --output "$TASK_OUTPUT/$TASK_NAME" --staging "$OUT/staging"
done
