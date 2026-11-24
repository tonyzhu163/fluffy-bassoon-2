#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
code=$(git -C "$here" rev-parse --show-toplevel)
pin=${1:?deployed Git pin}
attempt=${2:?fresh attempt directory name}
mode=${3:?proof, digital, physical or join}
[[ $pin =~ ^[0-9a-f]{40}$ && $(git -C "$code" rev-parse HEAD) == "$pin" ]]
[[ $(uname -s) == Linux && $attempt =~ ^[a-z0-9_-]+$ ]]
root=/mowenjie/fluffy-bassoon-2
run="$root/outputs/$(basename "$here")"
out="$run/$attempt"
mkdir -p "$run"
exec 9>"$run/cpu-budget.lock"
flock -n 9
mkdir "$out"
df -h "$out" > "$out/storage.txt"
date +%s > "$out/start.txt"
trap 'rc=$?; echo "$rc $(date +%s)" > "$out/exit.txt"' EXIT
export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
sim="$root/setups/libero-meta-learning-20261117-r1"
release="$root/sources/libero-pro2-e5500c69"
export LIBERO_PRO_ROOT="$sim/simulator-source" LIBERO_CONFIG_PATH="$sim/libero-config" CROSSARM_RELEASE="$release"
export PYTHONPATH="$root/setups/xemb-icml-campaign-20261121-r1/cv2-headless-overlay:$code/src:$release:$sim/simulator-source:$code/outputs/libero-crossarm-20261120-r1:$code/outputs/libero-xemb-ur5e-sysid-20261120-r1:$code/outputs/libero-xemb-ur5e-scale-20261120-r1:$code/outputs/xemb-icml-campaign-20261121-r1:$code/outputs/symmeta-wider-family-20261123-r1:$code/outputs/symmeta-camera-command-20261123-r1:$code/outputs/symmeta-camera-frame-reader-20261124-r1"
py=/mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python
echo "$pin $(hostname) $mode $attempt $*" > "$out/launch.txt"
# Sum every earlier subprocess's user+system time and bound the next multithreaded process.
bounded() {
    local label=$1
    shift
    local remaining
    remaining=$(find "$run" -name '*.cpu' -type f -exec cat {} + | awk 'NF==2 {used+=$1+$2} END {print int(890-used)}')
    ((remaining > 0))
    (ulimit -t "$remaining"; TIMEFORMAT='%U %S'; { time taskset -c 50,51 "$py" -Bu "$@" >> "$out/run.log" 2>&1; } 2> "$out/$label.cpu")
}
case "$mode" in
    proof|digital|physical)
        bounded replay "$here/replay.py" "$mode" "$out" "${@:4}"
        for panel in digital physical; do
            if [[ -f "$out/$panel-predictions.json.gz" ]]; then
                bounded "aggregate-$panel" "$here/aggregate.py" "$out/$panel-predictions.json.gz" "$out/$panel-summary.json"
            fi
        done
        ;;
    join)
        bounded aggregate "$here/aggregate.py" "${4:?frozen predictions}" "$out/joined-summary.json" --refit "${5:?refit predictions}"
        ;;
    *) exit 2 ;;
esac
