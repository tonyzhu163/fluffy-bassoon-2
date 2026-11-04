#!/usr/bin/env bash
set -Eeuo pipefail
: "${BABYSIT_CODE:?}" "${BABYSIT_COMMIT:?}" "${BASELINE_MANIFEST_SHA:?}"
ROOT=/mowenjie/fluffy-bassoon-2
RUN=outputs/robocasa-eight-hour-babysit-20261104-r1
EVAL=$ROOT/sources/opendrawer-ablation-3m-eval-r3
ENGINE=$ROOT/sources/opendrawer-ablation-3m-engine-r2
VENDOR=$ROOT/sources/critic-selected-a1229239-r1
PY=$ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
test "$(git -C "$BABYSIT_CODE" rev-parse HEAD)" = "$BABYSIT_COMMIT"
test "$(git -C "$EVAL" rev-parse HEAD)" = 81caf87060c9d966e0256e94da2ae705366ec33f
test "$(git -C "$ENGINE" rev-parse HEAD)" = 02ce594015eaa8f972f875d7f37e6972f4ada495
test "$(git -C "$VENDOR" rev-parse HEAD)" = 8f9bfa7661928ef8c17494b04f4d7bf3125634e1
for CODE in "$BABYSIT_CODE" "$EVAL" "$ENGINE" "$VENDOR"; do test -z "$(git -C "$CODE" status --porcelain)"; done
if [[ $HOSTNAME == *-master-0 ]]; then INDEX=0; elif [[ $HOSTNAME == *-worker-0 ]]; then INDEX=1; else exit 2; fi
GROUP=$ROOT/$RUN/baseline-test512/attempt-01
OUT=$GROUP/$(printf 'node-%02d' "$INDEX")
mkdir -p "$GROUP"
mkdir "$OUT"
exec > >(tee "$OUT/console.log") 2>&1
START=$(date +%s)
finish() {
  rc=$?
  printf '{"started_unix":%s,"ended_unix":%s,"exit_code":%s,"hostname":"%s","gpus":1}\n' "$START" "$(date +%s)" "$rc" "$HOSTNAME" > "$OUT/CLOCK.json"
}
trap finish EXIT
echo "$BASELINE_MANIFEST_SHA  $ROOT/$RUN/BASELINE-MANIFEST.json" | sha256sum -c -
export TASK_CODE_PIN=02ce594015eaa8f972f875d7f37e6972f4ada495
export PYTHONPATH=$BABYSIT_CODE/$RUN:$EVAL/outputs/robocasa-opendrawer-ppo-ablation-3m-20261104-r1:$EVAL/outputs/robocasa-flow-sde-checkpoint-selection-20261104-r1:$EVAL/outputs/robocasa-flow-sde-endpoint-test512-20261104-r1:$ENGINE/src:$VENDOR
export OPENPI_DATA_HOME=$ROOT/setups/robocasa365-native-20261101-r1/cache/.cache/openpi
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NCCL_P2P_DISABLE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl FLUFFY_GPU_EXECUTION=baige PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
  UUIDS=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
  [[ $UUIDS == GPU-* && $UUIDS != *$'\n'* ]]
  export CUDA_VISIBLE_DEVICES=0
fi
"$PY" -B "$ENGINE/scripts/[train]gpu_probe.py" --expected-count 1 --purpose evaluation --baige-container --output "$OUT/GPU-PROBE.json"
DEADLINE=$((START+13200))
timeout --kill-after=30s 13200 "$PY" -B "$BABYSIT_CODE/$RUN/BASELINE.py" run --index "$INDEX" --output "$OUT" --deadline "$DEADLINE"
while (( $(date +%s) < DEADLINE )); do
  COUNT=$(find "$GROUP" -mindepth 2 -maxdepth 2 -name RESULT.json | wc -l)
  (( COUNT == 2 )) && exit 0
  sleep 10
done
exit 1
