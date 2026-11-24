#!/usr/bin/env bash
# Run inside the exclusive single-GPU Baige Distributed Training job step.
set -euo pipefail
if [[ ${1:-} != --timed ]]; then
    exec timeout --kill-after=10s 5340s bash "$0" --timed "$@"
fi
shift
[[ $# == 4 || $# == 5 ]]
pin=${1:?full Git revision}
attempt=${2:?fresh attempt name}
data=${3:?absolute staged source directory}
manifest=${4:?expected source manifest SHA-256}
prior=${5:-}
[[ $pin =~ ^[a-f0-9]{40}$ && $attempt =~ ^gpu[[:alnum:]_-]+$ && $manifest =~ ^[a-f0-9]{64}$ ]]
[[ $(uname -s) == Linux && $(hostname) == job-*-master-0 ]]
here=$(cd "$(dirname "$0")" && pwd -P)
code=$(git -C "$here" rev-parse --show-toplevel)
[[ $(git -C "$code" rev-parse HEAD) == "$pin" && -z $(git -C "$code" status --porcelain --untracked-files=no) ]]
git -C "$code" ls-files --error-unmatch -- "outputs/frozen-vla-cotraining-20261124-r1/data/source_data.py" >/dev/null
[[ $(sha256sum "$data/manifest.json" | cut -d' ' -f1) == "$manifest" ]]
root=/mowenjie/fluffy-bassoon-2
legacy=/mowenjie/fluffy-bassoon
run=$root/outputs/frozen-vla-cotraining-20261124-r1/model
out=$run/$attempt
mkdir "$out"
exec > >(tee "$out/console.log") 2>&1
start=$(date +%s)
trap 'rc=$?; echo "$rc $start $(date +%s)" > "$out/exit.txt"' EXIT
printf '%s\n' "$(hostname)" "$pin" "$data" "$manifest" "$start" > "$out/launch.txt"
df -h "$out" > "$out/storage.txt"
quota -s >> "$out/storage.txt" 2>/dev/null || true
export OPENPI_DATA_HOME=$legacy/cache/openpi-215abfb
export PYTHONPATH=$root/setups/xemb-icml-campaign-20261121-r1/cv2-headless-overlay:$code/src:$code/outputs/libero-xemb-ur5e-sysid-20261120-r1:$here/../data
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 FLUFFY_GPU_EXECUTION=baige
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=2
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="$out/tmp" XDG_CACHE_HOME="$out/cache-runtime"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
# Inherit the scheduler's assigned UUID and drivers. Drop image-local libtorch that
# would shadow the PyTorch ABI of the pinned probe and OpenPI environments.
IFS=: read -ra paths <<< "${LD_LIBRARY_PATH:-}"
clean=()
for p in "${paths[@]}"; do
    if [[ -n $p && ! -f $p/libtorch_python.so ]]; then clean+=("$p"); fi
done
export LD_LIBRARY_PATH=$(IFS=:; echo "${clean[*]}")
name=$(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ $name != *$'\n'* ]]
case "$name" in
    *6000D*|*RPBZZZ6*) export NCCL_P2P_DISABLE=1 ;;
    *4090*) ;;
    *) echo "Unexpected allocated accelerator $name"; exit 2 ;;
esac
if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
    uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
    [[ $uuid == GPU-* && $uuid == "${NVIDIA_VISIBLE_DEVICES:?}" ]]
    export CUDA_VISIBLE_DEVICES=0
fi
probe=$root/setups/robocasa365-native-20261101-r1/env/bin/python
PYTHONPATH="$code/src" "$probe" -Bu "$code/scripts/[train]gpu_probe.py" --expected-count 1 \
    --baige-container --baige-exclusive-training --purpose training --output "$out/gpu.json"
python=$legacy/envs/openpi-215abfb/bin/python
args=(--data "$data" --manifest "$manifest" --out "$out" --pin "$pin")
if [[ -n $prior ]]; then args+=(--prior "$prior"); fi
"$python" -Bu "$here/train.py" "${args[@]}"
