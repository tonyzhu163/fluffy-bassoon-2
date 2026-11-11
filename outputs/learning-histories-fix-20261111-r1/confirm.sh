#!/usr/bin/env bash
# One allocated GPU, 15-minute scheduler cap (0.25 GPU-hours).
set -Eeuo pipefail
root=/mowenjie/fluffy-bassoon-2
code=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
pin=${1:?analysis commit} identity=${2:?fresh analysis identity}
[[ $identity =~ ^[a-z0-9-]+$ && $HOSTNAME == job-*-master-0 ]]
test "$(git -C "$code" rev-parse HEAD)" = "$pin"
git -C "$code" diff --quiet HEAD
trainer=$root/sources/learning-histories-fix-4b06413e
test "$(git -C "$trainer" rev-parse HEAD)" = 4b06413e33d1e1279b2431f370ff9e638bb7cab9
git -C "$trainer" diff --quiet HEAD
folder=$root/outputs/learning-histories-fix-20261111-r1
out=$folder/$identity
mkdir "$out"
exec > >(tee "$out/run.log") 2>&1
started=$(date +%s)
trap 'rc=$?; echo "{\"exit_code\":$rc,\"started\":$started,\"ended\":$(date +%s),\"hostname\":\"$HOSTNAME\",\"gpus\":1}" > "$out/clock.json"' EXIT
echo "$pin $HOSTNAME" > "$out/identity.txt"
export FLUFFY_GPU_EXECUTION=baige NCCL_P2P_DISABLE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
export PYTHONPATH="$trainer/src"
if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
    uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
    [[ $uuid == GPU-* && $uuid != *$'\n'* && $uuid == "${NVIDIA_VISIBLE_DEVICES:?}" ]]
    export CUDA_VISIBLE_DEVICES=0
fi
py=$root/setups/robocasa365-native-20261101-r1/env/bin/python
"$py" -B "$trainer/scripts/[train]gpu_probe.py" --expected-count 1 --purpose evaluation --baige-container --output "$out/gpu.json"
old=$root/outputs/learning-histories-20261111-r1/grid01
"$py" -B "$(dirname "$0")/readout.py" "$folder/grid01" "$old" --out "$out/selection"
"$py" -Bu "$(dirname "$0")/confirm.py" --trainer-code "$trainer" --grid "$folder/grid01" --original-grid "$old" --selection "$out/selection/summary.json" --out "$out/results"
touch "$out/complete"
