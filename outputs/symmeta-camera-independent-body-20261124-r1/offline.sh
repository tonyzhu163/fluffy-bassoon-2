#!/usr/bin/env bash
# fixture PIN ATTEMPT BANK SHA 1, or release PIN ATTEMPT BANK SHA N FIT SEAL_SHA.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
code=$(git -C "$here" rev-parse --show-toplevel)
mode=${1:?fixture or release}
pin=${2:?deployed git SHA}
attempt=${3:?fresh attempt}
bank=${4:?absolute manifest path}
bank_sha=${5:?sealed bank digest}
size=${6:?prespecified panel size}
[[ $(uname -s) == Linux && "$pin" =~ ^[0-9a-f]{40}$ && $(git -C "$code" rev-parse HEAD) == "$pin" ]]
[[ -z $(git --no-optional-locks -C "$code" status --porcelain --untracked-files=no) && "$attempt" =~ ^[a-z0-9_-]+$ ]]
root=/mowenjie/fluffy-bassoon-2
run="$root/outputs/$(basename "$here")"
[[ $bank == "$run"/*/bank"$size".json && $bank_sha =~ ^[a-f0-9]{64}$ ]]
args=()
if [[ $mode == fixture ]]; then
    [[ $# == 6 && $size == 1 ]]
else
    [[ $mode == release && $# == 8 && ( $size == 32 || $size == 64 ) ]]
    [[ $7 == "$root/outputs/symmeta-camera-frame-reader-20261124-r1/"* && $8 =~ ^[0-9a-f]{64}$ ]]
    args=(--fit "$7" --seal-sha "$8")
fi
out="$run/$attempt"
if [[ $(hostname) == job-* ]]; then
    [[ $mode == release ]]
    remaining=1800
else
    exec 9>"$run/qualification.lock"
    flock -n 9
    used=$(awk '{s+=$1} END {print int(s)}' "$run"/*/wall.seconds)
    remaining=$((1800-used))
    ((remaining > 5))
fi
mkdir "$out"
start=$(date +%s)
trap 'rc=$?; end=$(date +%s); echo $((end-start)) > "$out/wall.seconds"; echo "$rc $start $end" > "$out/exit.txt"' EXIT
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
sim="$root/setups/libero-meta-learning-20261117-r1"
export LIBERO_PRO_ROOT="$sim/simulator-source" LIBERO_CONFIG_PATH="$sim/libero-config" CROSSARM_RELEASE="$root/sources/libero-pro2-e5500c69"
export PYTHONPATH="$root/setups/xemb-icml-campaign-20261121-r1/cv2-headless-overlay:$code/src:$CROSSARM_RELEASE:$LIBERO_PRO_ROOT:$code/outputs/libero-crossarm-20261120-r1:$code/outputs/libero-xemb-ur5e-sysid-20261120-r1:$code/outputs/libero-xemb-ur5e-scale-20261120-r1:$code/outputs/xemb-icml-campaign-20261121-r1:$code/outputs/symmeta-wider-family-20261123-r1:$code/outputs/symmeta-camera-frame-reader-20261124-r1:$code/outputs/symmeta-camera-frozen-preprocess-20261124-r1:$code/outputs/symmeta-camera-command-20261123-r1"
export TMPDIR="$out/tmp" XDG_CACHE_HOME="$out/cache" MPLCONFIGDIR="$out/cache/mpl"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
echo "$pin $(hostname) $mode $* $remaining" > "$out/launch.txt"
py=/mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python
if [[ $(hostname) == job-* ]]; then
    export FLUFFY_GPU_EXECUTION=baige NCCL_P2P_DISABLE=1 JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0
    libs="$root/setups/libero90-full-action-20261119-r1/desktop-libs02/prefix/usr/lib/x86_64-linux-gnu"
    IFS=: read -ra paths <<< "${LD_LIBRARY_PATH:-}"
    for path in "${paths[@]}"; do
        if [[ -n $path && ! -f $path/libtorch_python.so ]]; then libs+=":$path"; fi
    done
    export LD_LIBRARY_PATH="$libs"
    "$root/setups/robocasa365-native-20261101-r1/env/bin/python" -Bu "$code/scripts/[train]gpu_probe.py" --expected-count 1 --purpose evaluation --baige-container --output "$out/gpu.json" > "$out/probe.log" 2>&1
    "$py" -c 'import jax; assert jax.default_backend() == "gpu" and len(jax.devices()) == 1'
    "$py" -Bu "$here/score_frozen.py" "$mode" "$bank" "$bank_sha" "$size" "$out" "${args[@]}" > "$out/run.log" 2>&1
    exit 0
fi
export CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void JAX_PLATFORMS=cpu
# Verify availability again at the handover from renderer to inference.
taskset -c 36,37 python3 - <<'PY' > "$out/idle.json"
import json,time
def sample():
    with open('/proc/stat') as f:
        return {v[0]:list(map(int,v[1:9])) for line in f if (v:=line.split())[0] in ('cpu36','cpu37')}
a=sample();time.sleep(2);b=sample()
delta={k:[y-x for x,y in zip(v,b[k])] for k,v in a.items()}
print(json.dumps(delta))
assert len(delta)==2 and all(v[3]>.9*sum(v) for v in delta.values()),delta
PY
timeout -k 2s "$((remaining-4))s" taskset -c 36,37 /mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python -Bu \
    "$here/score_frozen.py" "$mode" "$bank" "$bank_sha" "$size" "$out" "${args[@]}" > "$out/run.log" 2>&1
