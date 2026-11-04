#!/usr/bin/env bash
set -Eeuo pipefail
: "${BABYSIT_CODE:?}" "${BABYSIT_COMMIT:?}"
ROOT=/mowenjie/fluffy-bassoon-2
RUN=outputs/robocasa-eight-hour-babysit-20261104-r1
PY=$ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
export RLINF_PATH=$ROOT/sources/critic-selected-a1229239-r1
test "$(git -C "$BABYSIT_CODE" rev-parse HEAD)" = "$BABYSIT_COMMIT"
test -z "$(git -C "$BABYSIT_CODE" status --porcelain)"
OUT=$ROOT/$RUN/gpu/kv-native-qualification-01
test ! -e "$OUT"
mkdir -p "$ROOT/$RUN/logs" "$ROOT/$RUN/probes"
exec > >(tee "$ROOT/$RUN/logs/kv-native-qualification-01.log") 2>&1
START=$(date +%s)
finish() {
  rc=$?
  mkdir -p "$OUT"
  printf '{"started_unix":%s,"ended_unix":%s,"exit_code":%s,"gpus":1,"hostname":"%s"}\n' "$START" "$(date +%s)" "$rc" "$HOSTNAME" > "$OUT/CLOCK.json"
}
trap finish EXIT
export PYTHONPATH=$BABYSIT_CODE/src:$RLINF_PATH
export OPENPI_DATA_HOME=$ROOT/setups/robocasa365-native-20261101-r1/cache/.cache/openpi
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NCCL_P2P_DISABLE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl FLUFFY_GPU_EXECUTION=baige PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
  UUIDS=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
  [[ $UUIDS == GPU-* && $UUIDS != *$'\n'* ]]
  export CUDA_VISIBLE_DEVICES=0
fi
PROBE=$ROOT/$RUN/probes/kv-native-qualification-01.json
"$PY" -B "$BABYSIT_CODE/scripts/[train]gpu_probe.py" --expected-count 1 --purpose training --baige-container --baige-exclusive-training --output "$PROBE"
timeout --kill-after=30s 1650 "$PY" -B "$BABYSIT_CODE/$RUN/BRIDGE-QUALIFY.py" --code "$BABYSIT_CODE" --commit "$BABYSIT_COMMIT" --probe "$PROBE" --output "$OUT"
