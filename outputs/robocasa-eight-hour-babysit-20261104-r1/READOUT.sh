#!/usr/bin/env bash
set -Eeuo pipefail
: "${BABYSIT_CODE:?}" "${BABYSIT_COMMIT:?}" "${BABYSIT_METHOD:?}" "${BABYSIT_SPLIT:?}" "${BABYSIT_ATTEMPT:?}"
ROOT=/mowenjie/fluffy-bassoon-2
RUN=outputs/robocasa-eight-hour-babysit-20261104-r1
PY=$ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
export RLINF_PATH=$ROOT/sources/critic-selected-a1229239-r1
test "$(git -C "$BABYSIT_CODE" rev-parse HEAD)" = "$BABYSIT_COMMIT"
test -z "$(git -C "$BABYSIT_CODE" status --porcelain)"
test "$(git -C "$RLINF_PATH" rev-parse HEAD)" = 8f9bfa7661928ef8c17494b04f4d7bf3125634e1
OUT=$ROOT/$RUN/gpu/$BABYSIT_ATTEMPT
test ! -e "$OUT"
mkdir -p "$(dirname "$OUT")"
mkdir -p "$ROOT/$RUN/logs"
exec > >(tee "$ROOT/$RUN/logs/$BABYSIT_ATTEMPT.log") 2>&1
START=$(date +%s)
finish() {
  rc=$?
  mkdir -p "$OUT"
  printf '{"started_unix":%s,"ended_unix":%s,"exit_code":%s,"hostname":"%s","gpus":1}\n' "$START" "$(date +%s)" "$rc" "$HOSTNAME" > "$OUT/CLOCK.json"
}
trap finish EXIT
export PYTHONPATH=$BABYSIT_CODE/src:$RLINF_PATH
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OPENPI_DATA_HOME=$ROOT/setups/robocasa365-native-20261101-r1/cache/.cache/openpi
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NCCL_P2P_DISABLE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl FLUFFY_GPU_EXECUTION=baige
if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
  UUIDS=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
  [[ $UUIDS == GPU-* && $UUIDS != *$'\n'* ]]
  export CUDA_VISIBLE_DEVICES=0
fi
PROBE=$ROOT/$RUN/probes/$BABYSIT_ATTEMPT.json
mkdir -p "$(dirname "$PROBE")"
"$PY" -B "$BABYSIT_CODE/scripts/[train]gpu_probe.py" --expected-count 1 --purpose evaluation --baige-container --output "$PROBE"
SECONDS_LIMIT=${BABYSIT_SECONDS_LIMIT:-3300}
timeout --kill-after=30s "$SECONDS_LIMIT" "$PY" -B "$BABYSIT_CODE/$RUN/READOUT.py" \
  --code "$BABYSIT_CODE" --source-commit "$BABYSIT_COMMIT" --method "$BABYSIT_METHOD" \
  --split "$BABYSIT_SPLIT" --stage "${BABYSIT_STAGE:-qualification}" --count "${BABYSIT_COUNT:-2}" \
  --output "$OUT" --probe "$PROBE" --expected-hostname "$HOSTNAME" --deadline "$((START+SECONDS_LIMIT))"
