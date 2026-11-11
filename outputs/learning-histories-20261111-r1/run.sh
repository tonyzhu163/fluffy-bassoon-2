#!/usr/bin/env bash
set -Eeuo pipefail
root=/mowenjie/fluffy-bassoon-2
code=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
pin=${1:?commit} mode=${2:?qualify or grid} identity=${3:?fresh identity}
[[ $identity =~ ^[a-z0-9-]+$ && $HOSTNAME == job-* ]]
test "$(git -C "$code" rev-parse HEAD)" = "$pin"
git -C "$code" diff --quiet HEAD
if [[ $HOSTNAME =~ -master-0$ ]]; then
    seed=0
elif [[ $HOSTNAME =~ -worker-([0-9]+)$ ]]; then
    seed=$((BASH_REMATCH[1]+1))
else
    exit 2
fi
index=$seed
conditions=(demo blind attempts corrections linked)
if [[ $mode == grid ]]; then
    test "$index" -lt 15
    seed=$((index % 3))
    cond=${conditions[index/3]}
    member=$cond-s$seed
else
    test "$index" -eq 0
    member=seed-0
fi
folder=$root/outputs/learning-histories-20261111-r1
mkdir -p "$folder/$identity"
out=$folder/$identity/$member
mkdir "$out"
exec > >(tee "$out/run.log") 2>&1
started=$(date +%s)
trap 'rc=$?; echo "{\"exit_code\":$rc,\"started\":$started,\"ended\":$(date +%s),\"hostname\":\"$HOSTNAME\",\"gpus\":1}" > "$out/clock.json"' EXIT
echo "$pin $HOSTNAME" > "$out/identity.txt"
df -h "$out" | tee "$out/storage.txt"
if command -v quota >/dev/null; then quota -s >> "$out/storage.txt" 2>&1 || true; else echo 'quota tool unavailable' >> "$out/storage.txt"; fi
test -w "$out"
export FLUFFY_GPU_EXECUTION=baige NCCL_P2P_DISABLE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
export PYTHONPATH="$code/src"
if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
    uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
    [[ $uuid == GPU-* && $uuid != *$'\n'* && $uuid == "${NVIDIA_VISIBLE_DEVICES:?}" ]]
    export CUDA_VISIBLE_DEVICES=0
fi
py=$root/setups/robocasa365-native-20261101-r1/env/bin/python
"$py" -B "$code/scripts/[train]gpu_probe.py" --expected-count 1 --purpose training --baige-container --baige-exclusive-training --output "$out/gpu.json"
if [[ $mode == qualify ]]; then
    "$py" -Bu "$(dirname "$0")/qualify.py" "$code" "$out"
    "$py" -Bu "$code/exploration/learning-histories/train.py" --cond attempts --K 6 --steps 500 --eval-n 256 --out "$out/timing.json" --device cuda
elif [[ $mode == grid ]]; then
    test -f "$folder/${4:?qualification identity}/seed-0/qualification.json"
    "$py" -Bu "$code/exploration/learning-histories/train.py" --cond "$cond" --K 6 --seed "$seed" --out "$out/$cond.json" --device cuda > "$out/$cond.log" 2>&1
else
    exit 2
fi
touch "$out/complete"
if [[ $mode == grid && $index == 0 ]]; then
    until [[ $(find "$folder/$identity" -mindepth 2 -maxdepth 2 -name complete | wc -l) == 15 ]]; do
        if find "$folder/$identity" -name clock.json -exec grep -lE '"exit_code":[1-9]' {} + | grep -q .; then exit 1; fi
        sleep 10
    done
fi
