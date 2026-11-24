#!/usr/bin/env bash
# One caller owns each fresh attempt directory. CPU modes run on Baige Linux.
set -euo pipefail
mode=${1:?data, qualify, refit, or predict}
pin=${2:?full deployed Git revision}
attempt=${3:?fresh attempt name}
data_attempt=${4:-}
checkpoint_attempt=${5:-}
[[ $(uname -s) == Linux && $pin =~ ^[a-f0-9]{40}$ && $attempt =~ ^[[:alnum:]][[:alnum:]_-]*$ ]]
case "$mode" in
    data) [[ $# == 3 ]] ;;
    qualify|refit) [[ $# == 4 ]] ;;
    predict) [[ $# == 5 ]] ;;
    *) exit 2 ;;
esac
for src in "$data_attempt" "$checkpoint_attempt"; do
    [[ -z $src || $src =~ ^[[:alnum:]][[:alnum:]_-]*$ ]]
done
here=$(cd "$(dirname "$0")" && pwd -P)
code=$(git -C "$here" rev-parse --show-toplevel)
[[ $(git -C "$code" rev-parse HEAD) == "$pin" && -z $(git -C "$code" status --porcelain) ]]
if git -C "$code" symbolic-ref -q HEAD >/dev/null; then
    echo "Use a detached pin" >&2
    exit 2
fi
root=/mowenjie/fluffy-bassoon-2
legacy=/mowenjie/fluffy-bassoon
run="$root/outputs/$(basename "$here")"
py="$legacy/envs/openpi-215abfb/bin/python"
host=$(hostname)
if [[ $mode == refit && $host != job-* ]]; then
    echo "Use an exclusive Baige Distributed Training allocation for reader refits" >&2
    exit 2
fi
mkdir -p "$run"
out="$run/$attempt"
mkdir "$out"
start=$(date +%s)
trap 'rc=$?; printf "%s %s %s\n" "$rc" "$(date +%s)" "$start" > "$out/exit.txt"' EXIT
printf '%s %s %s %s %s %s\n' "$pin" "$host" "$mode" "$attempt" "$data_attempt" "$checkpoint_attempt" > "$out/launch.txt"
df -h "$out" > "$out/storage.txt"
sim="$root/setups/libero-meta-learning-20261117-r1"
release="$root/sources/libero-pro2-e5500c69"
overlay="$root/setups/xemb-icml-campaign-20261121-r1/cv2-headless-overlay"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export LIBERO_PRO_ROOT="$sim/simulator-source" LIBERO_CONFIG_PATH="$sim/libero-config"
export CROSSARM_RELEASE="$release" OPENPI_DATA_HOME="$legacy/cache/openpi-215abfb"
export PYTHONPATH="$overlay:$code/src:$release:$LIBERO_PRO_ROOT:$code/outputs/libero-crossarm-20261120-r1:$code/outputs/libero-xemb-ur5e-sysid-20261120-r1:$code/outputs/libero-xemb-ur5e-scale-20261120-r1:$code/outputs/xemb-icml-campaign-20261121-r1:$code/outputs/symmeta-wider-family-20261123-r1"
if [[ $mode == refit ]]; then
    export JAX_PLATFORMS=cuda FLUFFY_GPU_EXECUTION=baige NCCL_P2P_DISABLE=1
    export CUDA_VISIBLE_DEVICES=0
    libs="$root/setups/libero90-full-action-20261119-r1/desktop-libs02"
    IFS=: read -ra paths <<< "${LD_LIBRARY_PATH:-}"
    drivers=""
    for p in "${paths[@]}"; do
        if [[ -n $p && ! -f $p/libtorch_python.so ]]; then drivers+=":$p"; fi
    done
    export LD_LIBRARY_PATH="$libs/prefix/usr/lib/x86_64-linux-gnu$drivers"
    date +%s > "$out/gpu-start.txt"
    "$root/setups/robocasa365-native-20261101-r1/env/bin/python" -Bu "$code/scripts/[train]gpu_probe.py" \
        --expected-count 1 --purpose training --baige-container --baige-exclusive-training \
        --output "$out/gpu.json" > "$out/probe.log" 2>&1
    "$py" -Bu "$here/readers.py" refit "$run/$data_attempt" "$out" "$pin" > "$out/run.log" 2>&1
    date +%s > "$out/gpu-end.txt"
else
    export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void
    # Respect the process cpuset. Limit JAX and OpenCV to two allowed CPU cores.
    cores=$("$py" -B -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0))[:2])))')
    prefix=(taskset -c "$cores" "$py" -Bu)
    case "$mode" in
        data) "${prefix[@]}" "$here/source.py" "$out" "$pin" > "$out/run.log" 2>&1 ;;
        qualify) "${prefix[@]}" "$here/readers.py" qualify "$run/$data_attempt" "$out" "$pin" > "$out/run.log" 2>&1 ;;
        predict) "${prefix[@]}" "$here/readers.py" predict "$run/$data_attempt" "$out" "$pin" \
            --checkpoints "$run/$checkpoint_attempt" > "$out/run.log" 2>&1 ;;
    esac
fi
