#!/usr/bin/env bash
set -Eeuo pipefail
: "${TASK_META:?}" "${TASK_CODE_PIN:?}"
ROOT=/mowenjie/fluffy-bassoon-2
RUN=robocasa-ppo40-matched-dev64-20261104-r1
RUNTIME=$ROOT/sources/flow10m-0988b5f9
RLINF=$ROOT/sources/robocasa365-public-pi05-nnx-leaves-20261101-r1/RLinf
PY=$ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
OUT=$ROOT/outputs/$RUN/gpu-r1
READY=$ROOT/outputs/$RUN/staging-r1/READY-FOUR.json
test -s "$READY"
test "$(git -C "$TASK_META" rev-parse HEAD)" = "$TASK_CODE_PIN"
test "$(git -C "$RUNTIME" rev-parse HEAD)" = 0988b5f9e9bf31ac3db673cc2860959611b9ae12
test "$(git -C "$RLINF" rev-parse HEAD)" = 8a1233c3fae6f5905adfda7684c455842b0ecfb8
for p in "$TASK_META" "$RUNTIME" "$RLINF"; do test -z "$(git -C "$p" status --porcelain)"; done
mkdir "$OUT"
exec > >(tee "$OUT/console.log") 2>&1
begin=$(date +%s)
finish() {
 rc=$?
 printf '{"start":%d,"end":%d,"exit_code":%d,"hostname":"%s"}\n' "$begin" "$(date +%s)" "$rc" "$HOSTNAME" > "$OUT/clock.json"
}
trap finish EXIT
export TASK_CODE_PIN PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$TASK_META/outputs/robocasa-flow-sde-checkpoint-selection-20261104-r1:$RUNTIME/src:$RLINF
export OPENPI_DATA_HOME=$ROOT/setups/robocasa365-native-20261101-r1/cache/.cache/openpi
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl FLUFFY_GPU_EXECUTION=baige
"$PY" "$RUNTIME/scripts/[train]gpu_probe.py" --expected-count 1 --purpose evaluation --baige-container --output "$OUT/gpu-probe.json"
timeout --kill-after=30s 4350 "$PY" "$TASK_META/outputs/$RUN/evaluate.py" "$READY" "$OUT" "$((begin+4350))"
