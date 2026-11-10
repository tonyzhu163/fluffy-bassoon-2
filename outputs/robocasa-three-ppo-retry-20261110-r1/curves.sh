#!/usr/bin/env bash
set -Eeuo pipefail
root=/mowenjie/fluffy-bassoon-2
code=$1
export PILOT_COMMIT=$2
r=outputs/robocasa-three-ppo-retry-20261110-r1
p=outputs/robocasa-three-ppo-dpt-lora-20261110-r1
test "$(git -C "$code" rev-parse HEAD)" = "$PILOT_COMMIT"
test -z "$(git -C "$code" status --porcelain --untracked-files=no)"
if [[ $HOSTNAME =~ ^(job-[a-z0-9]+)-master-0$ ]]; then index=0
elif [[ $HOSTNAME =~ ^(job-[a-z0-9]+)-worker-([0-9]+)$ ]]; then index=$((BASH_REMATCH[2]+1))
else exit 2; fi
job=${BASH_REMATCH[1]}
seeds=(991 1481 2029)
seed=${seeds[index]}
group=$root/$r/$3
out=$group/seed-$seed
mkdir -p "$group"
mkdir "$out"
exec > >(tee "$out/fit.log") 2>&1
started=$(date +%s)
trap 'rc=$?; echo "{\"exit_code\":$rc,\"started\":$started,\"ended\":$(date +%s),\"job\":\"$job\"}" > "$out/clock.json"' EXIT
export RLINF_PATH=$root/sources/critic-selected-a1229239-r1
test "$(git -C "$RLINF_PATH" rev-parse HEAD)" = 8f9bfa7661928ef8c17494b04f4d7bf3125634e1
export PYTHONPATH=$code/src:$RLINF_PATH
export ROBOCASA_PI05_TORCH_MODEL=$root/inputs/robocasa365-public-pi05-c484-torch-20261101-r2
export ROBOCASA_PI05_NATIVE_NORM=$root/inputs/robocasa365-public-pi05-c484-75000-20261101-r1/assets/norm_stats.json
export OPENPI_DATA_HOME=$root/setups/robocasa365-native-20261101-r1/cache/.cache/openpi
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export FLUFFY_GPU_EXECUTION=baige MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NCCL_P2P_DISABLE=1
export ROBOCASA_NATIVE_OUTPUT=$out PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
py=$root/setups/robocasa365-native-20261101-r1/env/bin/python
"$py" -B "$code/scripts/[train]gpu_probe.py" --expected-count 1 --output "$out/gpu.json" --purpose training --baige-container --baige-exclusive-training
timeout --kill-after=20 3000 "$py" -Bu "$code/$p/fit.py" "$code" "$root/$p/data-01" "$out" --epochs 8 --fit-seed "$seed" --validation-steps 144 288 576 1152 2304
if [[ $index == 0 ]]; then
    while (( $(date +%s) - started < 3300 )); do
        if [[ $(find "$group" -mindepth 2 -maxdepth 2 -name held-out.json | wc -l) == 3 ]]; then exit 0; fi
        if find "$group" -name clock.json -exec grep -lE '"exit_code":[1-9]' {} + | grep -q .; then exit 1; fi
        sleep 4
    done
    exit 1
fi
