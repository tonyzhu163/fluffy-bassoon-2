#!/usr/bin/env bash
set -euo pipefail
pin=$1
stage=$2
attempt=$3
code=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
[[ "$attempt" =~ ^[a-z0-9]+$ && $(git -C "$code" rev-parse HEAD) == "$pin" ]]
[[ -z $(git -C "$code" status --porcelain) ]]
root=/mowenjie/fluffy-bassoon-2
release="$root/sources/libero-pro2-e5500c69"
runtime="$root/setups/libero-meta-learning-20261117-r1"
archive="$root/outputs/libero-decision-pretraining-20261119-r1/menu02"
[[ $(git -C "$release" rev-parse HEAD) == e5500c690c1498eca1b7050350654964190cdf70 ]]
[[ $(cut -d' ' -f1 "$archive/launch.txt") == b90ee5a0c049d301529d0bf661d0a682b858ed78 ]]
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$release:$runtime/simulator-source:$code/src"
export LIBERO_PRO_ROOT="$runtime/simulator-source" LIBERO_CONFIG_PATH="$runtime/libero-config"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
py="$runtime/env/bin/python"
case "$stage" in
    factor)
        run=libero-axis-factor-20261119-r1
        py="$root/setups/robocasa365-native-20261101-r1/env/bin/python"
        cmd=(factor.py --episodes "$archive/episodes")
        cap=10m ;;
    native)
        run=libero-axis-native-20261119-r1
        cmd=(native.py --episodes "$archive/episodes" --release "$release")
        cap=30m ;;
    s18)
        run=libero-s18-probe-20261119-r1
        cmd=(probe.py --release "$release")
        cap=5m ;;
    *) exit 2 ;;
esac
out="$root/outputs/$run/$attempt"
mkdir "$out"
start=$(date +%s)
trap 'rc=$?; echo "$rc $(date +%s) $start" > "$out/exit.txt"' EXIT
df -h "$out" > "$out/storage.txt"
printf '%s %s %s\n' "$pin" "$(hostname)" "$$" > "$out/launch.txt"
cd "$code/outputs/$run"
timeout --kill-after=10s "$cap" "$py" -Bu "${cmd[@]}" --output "$out/results" > "$out/run.log" 2>&1
