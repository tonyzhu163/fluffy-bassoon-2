#!/usr/bin/env bash
# One learner variant per one-GPU worker of a 15-worker Baige job.
#   run.sh <commit> <fresh identity>
set -Eeuo pipefail
root=/mowenjie/fluffy-bassoon-2
code=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
pin=${1:?commit} identity=${2:?fresh identity}
[[ $identity =~ ^[a-z0-9-]+$ && $HOSTNAME == job-* ]]
test "$(git -C "$code" rev-parse HEAD)" = "$pin"
git -C "$code" diff --quiet HEAD
if [[ $HOSTNAME =~ -master-0$ ]]; then
    index=0
elif [[ $HOSTNAME =~ -worker-([0-9]+)$ ]]; then
    index=$((BASH_REMATCH[1]+1))
else
    exit 2
fi
test "$index" -lt 15
names=(both base-lr both-lr wide attempts-both-lr)
conds=(corrections corrections corrections corrections attempts)
flags=("--dense --fourier 4" "--lr 1e-3" "--dense --fourier 4 --lr 1e-3" "--dense --fourier 4 --lr 5e-4 --d 256 --layers 6 --heads 8" "--dense --fourier 4 --lr 1e-3")
variant=$((index / 3)) seed=$((index % 3))
member=${names[variant]}-s$seed cond=${conds[variant]}
folder=$root/outputs/learning-histories-fix-20261111-r1
mkdir -p "$folder/$identity"
out=$folder/$identity/$member
mkdir "$out"
exec > >(tee "$out/run.log") 2>&1
started=$(date +%s)
trap 'rc=$?; echo "{\"exit_code\":$rc,\"started\":$started,\"ended\":$(date +%s),\"hostname\":\"$HOSTNAME\",\"gpus\":1}" > "$out/clock.json"' EXIT
echo "$pin $HOSTNAME" > "$out/identity.txt"
df -h "$out" | tee "$out/storage.txt"
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
# shellcheck disable=SC2086
"$py" -Bu "$code/exploration/learning-histories/train.py" --cond "$cond" ${flags[variant]} --K 6 --seed "$seed" --out "$out/result.json" --device cuda > "$out/fit.log" 2>&1
touch "$out/complete"
if [[ $index == 0 ]]; then
    until [[ $(find "$folder/$identity" -mindepth 2 -maxdepth 2 -name complete | wc -l) == 15 ]]; do
        if find "$folder/$identity" -name clock.json -exec grep -lE '"exit_code":[1-9]' {} + | grep -q .; then exit 1; fi
        sleep 10
    done
fi
