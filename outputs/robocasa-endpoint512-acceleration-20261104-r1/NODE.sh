#!/usr/bin/env bash
set -Eeuo pipefail
: "${TASK_META:?}" "${TASK_CODE_PIN:?}" "${TASK_MANIFEST:?}" "${TASK_ATTEMPT:?}"
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_RUN=$TASK_META/outputs/robocasa-endpoint512-acceleration-20261104-r1
TASK_RUNTIME=$TASK_ROOT/sources/flow10m-0988b5f9
TASK_RLINF=$TASK_ROOT/sources/robocasa365-public-pi05-nnx-leaves-20261101-r1/RLinf
TASK_EVAL=$TASK_ROOT/sources/stove-drift-endpoint512-d13dc18d/outputs/robocasa-flow-sde-checkpoint-selection-20261104-r1
TASK_PY=$TASK_ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
# Baige PyTorch jobs have master-0 followed by worker-0 through worker-6.
# Bind the replica identity to its scheduler hostname.
if [[ ${HOSTNAME:-} == *-master-0 ]]; then
 TASK_INDEX=0
elif [[ ${HOSTNAME:-} =~ -worker-([0-6])$ ]]; then
 TASK_INDEX=$((${BASH_REMATCH[1]}+1))
else
 echo 'No unambiguous scheduler replica index' >&2
 exit 1
fi
[[ $TASK_INDEX =~ ^[0-7]$ ]]
TASK_NODE=$(printf 'node-%02d' "$TASK_INDEX")
TASK_OUTPUT=$TASK_ROOT/outputs/robocasa-endpoint512-acceleration-20261104-r1/$TASK_ATTEMPT/$TASK_NODE
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
printf '{"index":%s,"hostname":"%s","scheduler_rank":"%s"}\n' "$TASK_INDEX" "$HOSTNAME" "${RANK:-}" > "$TASK_OUTPUT/REPLICA.json"
test "$(git -C "$TASK_META" rev-parse HEAD)" = "$TASK_CODE_PIN"
test -z "$(git -C "$TASK_META" status --porcelain)"
test "$(git -C "$TASK_RUNTIME" rev-parse HEAD)" = 0988b5f9e9bf31ac3db673cc2860959611b9ae12
test -z "$(git -C "$TASK_RUNTIME" status --porcelain)"
test "$(git -C "$TASK_RLINF" rev-parse HEAD)" = 8a1233c3fae6f5905adfda7684c455842b0ecfb8
test -z "$(git -C "$TASK_RLINF" status --porcelain)"
export PYTHONPATH=$TASK_RUN:$TASK_EVAL:$TASK_RUNTIME/src:$TASK_RLINF
export OPENPI_DATA_HOME=$TASK_ROOT/setups/robocasa365-native-20261101-r1/cache/.cache/openpi
export NCCL_P2P_DISABLE=1 FLUFFY_GPU_EXECUTION=baige
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
"$TASK_PY" -B "$TASK_RUNTIME/scripts/[train]gpu_probe.py" --expected-count 1 --purpose evaluation \
 --baige-container --output "$TASK_OUTPUT/gpu-probe.json"
TASK_DEADLINE=$((TASK_STARTED+1050))
timeout --signal=TERM --kill-after=30s 1050 "$TASK_PY" -B "$TASK_RUN/parallel_endpoint.py" worker \
 --manifest "$TASK_MANIFEST" --index "$TASK_INDEX" --deadline "$TASK_DEADLINE"
"$TASK_PY" -B "$TASK_RUN/parallel_endpoint.py" barrier --manifest "$TASK_MANIFEST" \
 --deadline "$((TASK_STARTED+1065))"
