#!/usr/bin/env bash
# No allocation or reservation: owner supplies an already funded exclusive job.
set -euo pipefail
TASK_CODE=$(cd -- "$(dirname -- "$0")/../.." && pwd)
TASK_PYTHON=${TASK_PYTHON:?Set the qualified allocated Linux Python executable}
TASK_PIN=${1:?full clean source pin}
TASK_RUN=${2:?unique libero-cached-scalar run id}
TASK_FUNDING=${3:?ROOT-funded record path}
TASK_FUNDING_SHA=${4:?funding SHA256}
TASK_INPUTS=${5:?hash-bound input index path}
TASK_INPUTS_SHA=${6:?input index SHA256}
TASK_OPENPI=${7:?qualified pinned OpenPI root}
TASK_CHECKPOINT=${8:?official pi05_libero checkpoint root}
TASK_OUTPUT=${9:?fresh external output directory named exactly as run id}
[[ "$(uname -s)" == Linux && "$TASK_PIN" =~ ^[0-9a-f]{40}$ ]]
[[ "$(hostname)" == job-* ]]
export PYTHONPATH="$TASK_CODE/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 FLUFFY_GPU_EXECUTION=baige
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
# Owner hard-caps the actual scheduler allocation at600seconds, including setup.
# Inner540s, outer570s TERM +20s KILL leave10s for scheduler/launcher overhead.
exec timeout --signal=TERM --kill-after=20s 570 "$TASK_PYTHON" \
  -m fluffy_bassoon.meta_rl.vla_cached_scalar_qualification \
  --source-commit "$TASK_PIN" --run-id "$TASK_RUN" --expected-hostname "$(hostname)" \
  --funding "$TASK_FUNDING" --funding-sha256 "$TASK_FUNDING_SHA" \
  --inputs "$TASK_INPUTS" --inputs-sha256 "$TASK_INPUTS_SHA" \
  --openpi-root "$TASK_OPENPI" --checkpoint "$TASK_CHECKPOINT" --output "$TASK_OUTPUT"
