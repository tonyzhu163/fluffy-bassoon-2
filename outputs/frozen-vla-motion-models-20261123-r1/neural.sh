#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
root=/mowenjie/fluffy-bassoon-2
runtime=/mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python
splits="$root/setups/xemb-icml-campaign-20261121-r1/source-ff2ddf0a/outputs/libero-meta-learning-20261117-r1/SPLITS.json"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
exec "$runtime" -Bu "$here/neural.py" --data-root "$root/outputs" --splits "$splits" --linear "$root/outputs/frozen-vla-motion-models-20261123-r1/cpu01/source-fit.json" --out "${1:?fresh absolute output directory}"
