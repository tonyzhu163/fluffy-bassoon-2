#!/usr/bin/env bash
set -euo pipefail
mode=${1:?qualify, prepare, fit, physical or physical-gpu}
pin=${2:?full git pin}
attempt=${3:?fresh attempt directory}
here=$(cd "$(dirname "$0")" && pwd -P)
code=$(git -C "$here" rev-parse --show-toplevel)
[[ $(uname -s) == Linux && $pin =~ ^[a-f0-9]{40}$ && $attempt =~ ^[a-zA-Z0-9_-]+$ ]]
[[ $(git -C "$code" rev-parse HEAD) == "$pin" && -z $(git -C "$code" status --porcelain --untracked-files=no) ]]
root=/mowenjie/fluffy-bassoon-2
legacy=/mowenjie/fluffy-bassoon
run="$root/outputs/$(basename "$here")"
out="$run/$attempt"
py="$legacy/envs/openpi-215abfb/bin/python"
sim="$root/setups/libero-meta-learning-20261117-r1"
release="$root/sources/libero-pro2-e5500c69"
overlay="$root/setups/xemb-icml-campaign-20261121-r1/cv2-headless-overlay"
export PYTHONPATH="$overlay:$code/src:$release:$sim/simulator-source:$code/outputs/libero-crossarm-20261120-r1:$code/outputs/libero-xemb-ur5e-sysid-20261120-r1:$code/outputs/libero-xemb-ur5e-scale-20261120-r1:$code/outputs/xemb-icml-campaign-20261121-r1:$code/outputs/symmeta-wider-family-20261123-r1:$code/outputs/symmeta-camera-command-20261123-r1"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export LIBERO_PRO_ROOT="$sim/simulator-source" LIBERO_CONFIG_PATH="$sim/libero-config"
export CROSSARM_RELEASE="$release" OPENPI_DATA_HOME="$legacy/cache/openpi-215abfb"
mkdir -p "$run"
mkdir "$out"
start=$(date +%s)
trap 'rc=$?; echo "$rc $(date +%s) $start" > "$out/exit.txt"' EXIT
echo "$pin $(hostname) $mode $attempt" > "$out/launch.txt"
df -h "$out" > "$out/storage.txt"
if [[ $mode == fit || $mode == physical-gpu || $mode == blind ]]; then
    [[ $(hostname) == job-* ]]
    if [[ $mode == fit || $mode == blind ]]; then
        [[ -s "$run/${4:?prepared bank}/bank/ready.json" ]]
    else
        [[ -s "$run/${4:?sealed fit}/sealed.json" && -s "${5:?physical bank}" ]]
    fi
    export JAX_PLATFORMS=cuda FLUFFY_GPU_EXECUTION=baige NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0
    libs="$root/setups/libero90-full-action-20261119-r1/desktop-libs02"
    IFS=: read -ra paths <<< "${LD_LIBRARY_PATH:-}"
    drivers=""
    for p in "${paths[@]}"; do
        if [[ -n $p && ! -f $p/libtorch_python.so ]]; then drivers+=":$p"; fi
    done
    export LD_LIBRARY_PATH="$libs/prefix/usr/lib/x86_64-linux-gnu$drivers"
    date +%s > "$out/gpu-start.txt"
    "$root/setups/robocasa365-native-20261101-r1/env/bin/python" -Bu "$code/scripts/[train]gpu_probe.py" \
       --expected-count 1 --purpose training --baige-container --baige-exclusive-training --output "$out/gpu.json" > "$out/probe.log" 2>&1
    case "$mode" in
        blind)
            "$py" -Bu "$here/augmentation.py" fit "$out" "$run/$4/bank" "$run/$5/bank" "$run/${6:?primary fit}" > "$out/train.log" 2>&1
            "$py" -Bu "$here/augmentation.py" digital "$out" "$out" "$run/$6/digital-predictions.json.gz" > "$out/digital.log" 2>&1
            "$py" -Bu "$here/augmentation.py" physical "$out" "$out" "$run/${7:?primary physical}/physical-predictions.json.gz" "${8:?physical bank}" > "$out/physical.log" 2>&1
            ;;
        fit)
            "$py" -Bu "$here/train.py" train "$run/$4/bank" "$out" "$run/${5:?off-axis prep}/bank" > "$out/train.log" 2>&1
            "$py" -Bu "$here/evaluate.py" digital "$out" "$out" > "$out/digital.log" 2>&1
            ;;
        physical-gpu)
            "$py" -c 'import jax; assert jax.default_backend() == "gpu" and len(jax.devices()) == 1'
            "$py" -Bu "$here/evaluate.py" physical "$run/$4" "$out" "$5" > "$out/run.log" 2>&1
            ;;
    esac
    date +%s > "$out/gpu-end.txt"
else
    export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void
    cmd=(taskset -c 34,35 "$py" -Bu)
    case "$mode" in
        qualify) "${cmd[@]}" "$here/qualify.py" "$out" > "$out/run.log" 2>&1 ;;
        prepare) "${cmd[@]}" "$here/train.py" prepare "$out/bank" > "$out/run.log" 2>&1 ;;
        extra) "${cmd[@]}" "$here/train.py" extra "$out/bank" "$run/${4:?base prep}/bank" > "$out/run.log" 2>&1 ;;
        physical) "${cmd[@]}" "$here/evaluate.py" physical "$run/${4:?sealed fit}" "$out" "${5:?qualified bank.json}" > "$out/run.log" 2>&1 ;;
        *) exit 2 ;;
    esac
fi
