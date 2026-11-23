#!/usr/bin/env bash
# Run from the local partition/audit directory. Only reads remote files and writes to stdout.
set -eu -o pipefail
receipt=${1:?usage: bash inspect.sh RECEIPTS.json}
root=/mowenjie/fluffy-bassoon-2/outputs/reground-rotation-reference-20261123-r1
{
  wc -c < inspect_survivors.py
  cat inspect_survivors.py
  jq -sc '{receipts:.[0], local_snapshot:.[1]}' "$receipt" ../../speed/active-snapshot.json
} | ssh -T -o BatchMode=yes baige-dev "cd '$root' && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 env/bin/python -IB -c 'import sys; b=sys.stdin.buffer; exec(b.read(int(b.readline())))'"
