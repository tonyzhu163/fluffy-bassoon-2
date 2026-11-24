#!/usr/bin/env bash
# The parent supplies the published data seal and submits this command once per fresh attempt.
set -euo pipefail
if [[ ${1:-} != --timed ]]; then
    exec timeout --kill-after=15s 7080s bash "$0" --timed "$@"
fi
shift
[[ $# == 3 || $# == 4 ]]
pin=$1
attempt=$2
seal=$3
[[ $pin =~ ^[0-9a-f]{40}$ && $attempt =~ ^gpu[0-9]+$ && $seal =~ ^[0-9a-f]{64}$ ]]
[[ $(uname -s) == Linux && $(hostname) == job-*-master-0 ]]
base=/mowenjie/fluffy-bassoon-2
legacy=/mowenjie/fluffy-bassoon
run=$base/outputs/frozen-vla-plate-transfer-20261124-r1
out=$run/fit/$attempt
here=$(cd "$(dirname "$0")" && pwd -P)
code=$(git -C "$here" rev-parse --show-toplevel)
mkdir "$out"
exec > >(tee "$out/console.log") 2>&1
start=$(date +%s)
trap 'rc=$?; echo "$rc $start $(date +%s)" > "$out/exit.txt"' EXIT
printf '%s\n' "$(hostname)" "$pin" "$attempt" "$seal" "$start" > "$out/launch.txt"
[[ $(git -C "$code" rev-parse HEAD) == "$pin" && -z $(git -C "$code" status --porcelain --untracked-files=no) ]]
data=$run/data/dataset-v1
[[ $(sha256sum "$data/seal.json" | cut -d' ' -f1) == "$seal" ]]
df -h "$out" > "$out/storage.txt"
quota -s >> "$out/storage.txt" 2>/dev/null || true
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 FLUFFY_GPU_EXECUTION=baige
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENPI_DATA_HOME=$legacy/cache/openpi-215abfb
export PYTHONPATH=$base/setups/xemb-icml-campaign-20261121-r1/cv2-headless-overlay:$code/src
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=2
export TMPDIR=$out/tmp XDG_CACHE_HOME=$out/runtime-cache
mkdir "$TMPDIR" "$XDG_CACHE_HOME"
IFS=: read -ra paths <<< "${LD_LIBRARY_PATH:-}"
clean=()
for p in "${paths[@]}"; do
    if [[ -n $p && ! -f $p/libtorch_python.so ]]; then clean+=("$p"); fi
done
export LD_LIBRARY_PATH=$(IFS=:; echo "${clean[*]}")
name=$(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ $name != *$'\n'* && $name == *4090* ]]
if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
    uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
    [[ $uuid == GPU-* && $uuid == "${NVIDIA_VISIBLE_DEVICES:?}" ]]
    export CUDA_VISIBLE_DEVICES=0
fi
cpus=$(python -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0))[:2])))')
probe=$base/setups/robocasa365-native-20261101-r1/env/bin/python
taskset -c "$cpus" "$probe" -Bu "$code/scripts/[train]gpu_probe.py" --expected-count 1 \
    --baige-container --baige-exclusive-training --purpose training --output "$out/gpu.json"
args=(--data "$data" --seal "$seal" --out "$out" --pin "$pin")
if [[ $# == 4 ]]; then args+=(--prior "$4"); fi
taskset -c "$cpus" "$legacy/envs/openpi-215abfb/bin/python" -Bu "$here/fit_plate.py" "${args[@]}"
