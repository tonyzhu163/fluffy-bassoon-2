#!/usr/bin/env bash
set -Eeuo pipefail
: "${EVAL_CODE:?Pinned Git checkout required}" "${EVAL_COMMIT:?}" "${EVAL_MANIFEST_SHA:?}"
root=/mowenjie/fluffy-bassoon-2
run=outputs/robocasa-opendrawer-ppo-ablation-3m-20261104-r1
data=$root/$run
export TASK_CODE_PIN=02ce594015eaa8f972f875d7f37e6972f4ada495
engine=$root/sources/opendrawer-ablation-3m-engine-r2
vendor=$root/sources/critic-selected-a1229239-r1
py=$root/setups/robocasa365-native-20261101-r1/env/bin/python
test "$(git -C "$EVAL_CODE" rev-parse HEAD)" = "$EVAL_COMMIT"
test "$(git -C "$engine" rev-parse HEAD)" = "$TASK_CODE_PIN"
test "$(git -C "$vendor" rev-parse HEAD)" = 8f9bfa7661928ef8c17494b04f4d7bf3125634e1
test -z "$(git -C "$EVAL_CODE" status --porcelain)"
test -z "$(git -C "$engine" status --porcelain)"
test -z "$(git -C "$vendor" status --porcelain)"
if [[ ${HOSTNAME:-} == *-master-0 ]]; then
  index=0
elif [[ ${HOSTNAME:-} =~ -worker-([0-2])$ ]]; then
  index=$(("${BASH_REMATCH[1]}"+1))
else
  exit 2
fi
node=$(printf 'node-%02d' "$index")
group=$data/endpoint512/attempt-01
out=$group/$node
test ! -e "$out"
mkdir -p "$out"
exec > >(tee "$out/console.log") 2>&1
start=$(date +%s)
finish() {
  code=$?
  printf '{"started_unix":%s,"finished_unix":%s,"exit_code":%s,"gpus":1}\n' "$start" "$(date +%s)" "$code" > "$out/clock.json"
}
trap finish EXIT
manifest=$data/EVAL-MANIFEST.json
echo "$EVAL_MANIFEST_SHA  $manifest" | sha256sum -c -
# Scientific workers and baseline analysis come from the experiment's pinned checkout.
export PYTHONPATH=$EVAL_CODE/$run:$EVAL_CODE/outputs/robocasa-flow-sde-checkpoint-selection-20261104-r1:$EVAL_CODE/outputs/robocasa-flow-sde-endpoint-test512-20261104-r1:$engine/src:$vendor
export OPENPI_DATA_HOME=$root/setups/robocasa365-native-20261101-r1/cache/.cache/openpi
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
  device_uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
  [[ $device_uuid == GPU-* && $device_uuid != *$'\n'* ]]
  export CUDA_VISIBLE_DEVICES=0
fi
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 NCCL_P2P_DISABLE=1 FLUFFY_GPU_EXECUTION=baige
"$py" -B "$engine/scripts/[train]gpu_probe.py" --expected-count 1 --purpose evaluation --baige-container --output "$out/gpu-probe.json"
deadline=$((start+10140))
timeout --kill-after=30s 10140 "$py" -B "$EVAL_CODE/$run/matched_endpoint.py" run --manifest "$manifest" --index "$index" --workers 4 --output "$out" --deadline "$deadline"
# A Baige master returning early can tear down the other container replicas.
while (( $(date +%s) < deadline )); do
  count=$(find "$group" -mindepth 2 -maxdepth 2 -name RESULT.json | wc -l)
  (( count == 4 )) && exit 0
  sleep 10
done
exit 1
