#!/usr/bin/env bash
set -Eeuo pipefail
root=/mowenjie/fluffy-bassoon-2
run=outputs/robocasa-three-ppo-retry-20261110-r1
prev=outputs/robocasa-three-ppo-dpt-lora-20261110-r1
code=$1
export TASK_CODE_PIN=$2
test "$(git -C "$code" rev-parse HEAD)" = "$TASK_CODE_PIN"
test -z "$(git -C "$code" status --porcelain --untracked-files=no)"
export RLINF_PATH=$root/sources/critic-selected-a1229239-r1
test "$(git -C "$RLINF_PATH" rev-parse HEAD)" = 8f9bfa7661928ef8c17494b04f4d7bf3125634e1
if [[ $HOSTNAME =~ ^(job-[a-z0-9]+)-master-0$ ]]; then
    index=0
elif [[ $HOSTNAME =~ ^(job-[a-z0-9]+)-worker-([0-9]+)$ ]]; then
    index=$((BASH_REMATCH[2]+1))
else
    exit 2
fi
job=${BASH_REMATCH[1]}
tasks=(OpenDrawer CloseToasterOvenDoor TurnOnSinkFaucet)
test "$index" -lt 12
task=${tasks[index%3]}
bank=$((index / 3))
group=$root/$run/$3
mkdir -p "$group"
out=$group/$task-$bank
mkdir "$out"
exec > >(tee "$out/run.log") 2>&1
started=$(date +%s)
trap 'rc=$?; echo "{\"exit_code\":$rc,\"started\":$started,\"ended\":$(date +%s),\"job\":\"$job\"}" > "$out/clock.json"' EXIT
echo "$HOSTNAME $task $bank $TASK_CODE_PIN"
export PYTHONPATH=$code/src:$code/$prev:$RLINF_PATH
export ROBOCASA_PI05_TORCH_MODEL=$root/inputs/robocasa365-public-pi05-c484-torch-20261101-r2
export ROBOCASA_PI05_NATIVE_NORM=$root/inputs/robocasa365-public-pi05-c484-75000-20261101-r1/assets/norm_stats.json
export OPENPI_DATA_HOME=$root/setups/robocasa365-native-20261101-r1/cache/.cache/openpi
export FLUFFY_GPU_EXECUTION=baige MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NCCL_P2P_DISABLE=1
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export ROBOCASA_NATIVE_OUTPUT=$out
py=$root/setups/robocasa365-native-20261101-r1/env/bin/python
"$py" -B "$code/scripts/[train]gpu_probe.py" --expected-count 1 --output "$out/gpu.json" --purpose evaluation --baige-container
timeout --kill-after=20 4500 "$py" -Bu "$code/$run/retry.py" "$code" "$out" "$task" --first-bank "$bank" --end-bank "$((bank+1))" --reuse "$root/$run/attempt-01"
if [[ $index == 0 ]]; then
    while (( $(date +%s) - started < 4750 )); do
        if [[ $(find "$group" -mindepth 2 -maxdepth 2 -name complete.json | wc -l) == 12 ]]; then exit 0; fi
        if find "$group" -name clock.json -exec grep -lE '"exit_code":[1-9]' {} + | grep -q .; then exit 1; fi
        sleep 4
    done
    exit 1
fi
