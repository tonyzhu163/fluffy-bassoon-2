#!/usr/bin/env bash
# Targeted phase payload only, under the sole owner's finite actual0GPU
# admission/proof/guardian. No prior54, model construction or physics.
set -euo pipefail
: "${READOUT_SOURCE:?exact clean provenance successor worktree}"
: "${READOUT_PYTHON:?existing qualified allocated-Linux Python}"
: "${READOUT_CPU_OUTPUT:?fresh small proof directory}"
: "${OWNER_CPU_RESOURCE_RECEIPT:?owner-validated actual allocated0GPU proof}"
[[ "$(uname -s)" == Linux ]]
[[ -s "$OWNER_CPU_RESOURCE_RECEIPT" ]]
[[ ! -e "$READOUT_CPU_OUTPUT" ]]
mkdir "$READOUT_CPU_OUTPUT"
cd "$READOUT_SOURCE"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1
export PYTHONPATH="$READOUT_SOURCE/src${PYTHONPATH:+:$PYTHONPATH}"
set +e
"$READOUT_PYTHON" -m pytest -q tests/test_robotwin_readout_provenance.py \
  --junitxml="$READOUT_CPU_OUTPUT/junit.xml" >"$READOUT_CPU_OUTPUT/pytest.log" 2>&1
code=$?
set -e
printf '%s\n' "$code" >"$READOUT_CPU_OUTPUT/exit-code"
exit "$code"
