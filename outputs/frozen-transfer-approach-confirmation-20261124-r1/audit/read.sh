#!/usr/bin/env bash
# Stream the independent reader onto two idle cores. Every remote path is read-only.
set -eu -o pipefail
cd "$(dirname "$0")"
pin=$1
attempt=$2
job=${3:-}
[[ "$pin" =~ ^[a-f0-9]{40}$ ]]
[[ "$attempt" =~ ^(gpu|cpu)[[:digit:]]+$ ]]
[[ -z "$job" || "$job" =~ ^job-[a-z0-9]+$ ]]
ssh -o BatchMode=yes baige-dev 'python3 -B -' < idle.py > "$attempt-idle.json"
python3 -c 'import json,sys; x=json.load(open(sys.argv[1])); assert not x["narrow_affinities"] and all(v[3] > sum(v[:3])+sum(v[4:]) for v in x["cpu_stat_deltas"].values()), x' "$attempt-idle.json"
cat references.py readback.py | ssh -o BatchMode=yes baige-dev "env CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1 taskset -c 46,47 /mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python -B - attempt $pin $attempt $job" > "$attempt.partial"
mv "$attempt.partial" "$attempt.json"
