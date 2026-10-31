#!/usr/bin/env bash
# Phase payload only; sole owner supplies the fresh finite0GPU180 admission,
# actual receipt, existing process guardian and complete terminal accounting.
set -euo pipefail
: "${READOUT_SOURCE:?exact clean successor worktree}"
: "${READOUT_PYTHON:?existing qualified allocated-Linux Python}"
: "${READOUT_CPU_OUTPUT:?fresh small evidence directory}"
: "${OWNER_CPU_RESOURCE_RECEIPT:?actual allocated0GPU proof already validated by owner}"
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
"$READOUT_PYTHON" -m pytest -q tests/test_robotwin_command_readout_resources.py \
  --junitxml="$READOUT_CPU_OUTPUT/junit.xml" >"$READOUT_CPU_OUTPUT/pytest.log" 2>&1
code=$?
set -e
printf '%s\n' "$code" >"$READOUT_CPU_OUTPUT/exit-code"
exit "$code"
