#!/usr/bin/env bash
set -euo pipefail
root=/mowenjie/fluffy-bassoon-2
sibling="$root/outputs/frozen-vla-motion-models-20261123-r1/source"
[[ $(git -C "$sibling" rev-parse HEAD) == 761f4209f4838099e4f80fad2937ad1e0f0cfcc5 ]]
here=$(cd "$(dirname "$0")" && pwd -P)
export PYTHONPATH="$sibling" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
exec taskset -c 0 /mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python -Bu "$here/compare.py" --root "$root/outputs" --splits "$root/setups/xemb-icml-campaign-20261121-r1/source-ff2ddf0a/outputs/libero-meta-learning-20261117-r1/SPLITS.json" --out "${1:?fresh output directory}"
