#!/usr/bin/env bash
# Run from parent-deployed pinned checkout. Fresh destination per attempt.
set -euo pipefail
code=$(git rev-parse --show-toplevel)
run=outputs/symmeta-camera-pose-pairs-20261124-r1
root=/mowenjie/fluffy-bassoon-2
dst=$root/$run/${1:?fresh attempt suffix}
mkdir -p "$root/$run"
mkdir "$dst"
git rev-parse HEAD > "$dst/pin.txt"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1
sim=$root/setups/libero-meta-learning-20261117-r1/env/bin/python
reader=/mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python
taskset -c 44 "$sim" "$code/$run/original.py" "$dst/original-bank.json"
taskset -c 44,45 "$sim" -u "$code/$run/render.py" target "$dst/original" 0 6
taskset -c 44 "$sim" -u "$code/$run/render.py" target "$dst/even" 2 4 7 > "$dst/even.log" 2>&1 &
even=$!
taskset -c 45 "$sim" -u "$code/$run/render.py" target "$dst/odd" 1 3 5 > "$dst/odd.log" 2>&1 &
odd=$!
wait "$even"
wait "$odd"
taskset -c 44,45 "$sim" -u "$code/$run/render.py" source "$dst/source"
export PYTHONPATH="$root/setups/xemb-icml-campaign-20261121-r1/cv2-headless-overlay:$code/src"
taskset -c 44,45 "$reader" -u "$code/$run/measure.py" "$dst/release" "$dst/original" "$dst/even" "$dst/odd" "$dst/source"
