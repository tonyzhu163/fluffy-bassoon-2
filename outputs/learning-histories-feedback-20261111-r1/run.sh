#!/usr/bin/env bash
set -Eeuo pipefail
root=/mowenjie/fluffy-bassoon-2
code=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
pin=${1:?commit} identity=${2:?fresh identity}
[[ $identity =~ ^[a-z0-9-]+$ && $HOSTNAME == job-* ]]
test "$(git -C "$code" rev-parse HEAD)" = "$pin"
git -C "$code" diff --quiet HEAD
if [[ $HOSTNAME =~ -master-0$ ]]; then seed=0
elif [[ $HOSTNAME =~ -worker-([0-9]+)$ ]]; then seed=$((BASH_REMATCH[1]+1))
else exit 2; fi
test "$seed" -lt 3
folder=$root/outputs/learning-histories-feedback-20261111-r1
mkdir -p "$folder/$identity"
out=$folder/$identity/seed-$seed
mkdir "$out"
exec > >(tee "$out/run.log") 2>&1
started=$(date +%s)
trap 'rc=$?; echo "{\"exit_code\":$rc,\"started\":$started,\"ended\":$(date +%s),\"hostname\":\"$HOSTNAME\",\"gpus\":1}" > "$out/clock.json"' EXIT
echo "$pin $HOSTNAME" > "$out/identity.txt"
df -h "$out" > "$out/storage.txt"
if command -v quota >/dev/null; then quota -s >> "$out/storage.txt" 2>&1 || true; else echo 'quota tool unavailable' >> "$out/storage.txt"; fi
printf 'write probe\n' > "$out/write-probe.txt"
export FLUFFY_GPU_EXECUTION=baige NCCL_P2P_DISABLE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
export PYTHONPATH="$code/src"
if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
 uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
 [[ $uuid == GPU-* && $uuid != *$'\n'* && $uuid == "${NVIDIA_VISIBLE_DEVICES:?}" ]]
 export CUDA_VISIBLE_DEVICES=0
fi
py=$root/setups/robocasa365-native-20261101-r1/env/bin/python
"$py" -B "$code/scripts/[train]gpu_probe.py" --expected-count 1 --purpose evaluation --baige-container --output "$out/gpu.json"
"$py" -Bu "$(dirname "$0")/diagnose.py" --code "$code" --grid "$root/outputs/learning-histories-20261111-r1/grid01" --seed "$seed" --out "$out" > "$out/evaluation.log" 2>&1
touch "$out/complete"
if [[ $seed == 0 ]]; then
 until [[ $(find "$folder/$identity" -mindepth 2 -maxdepth 2 -name complete | wc -l) == 3 ]]; do
  if find "$folder/$identity" -name clock.json -exec grep -lE '"exit_code":[1-9]' {} + | grep -q .; then exit 1; fi
  sleep 10
 done
fi
