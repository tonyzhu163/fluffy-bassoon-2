#!/usr/bin/env bash
# Direct admitted phase payload. Parent retains existing actual probe, process
# guardian, fresh output, exclusive action ownership and clock begin/finish.
set -euo pipefail
: "${READOUT_SOURCE:?exact clean deployed source}"
: "${READOUT_PYTHON:?actual allocated Linux Python}"
: "${READOUT_REQUEST:?materialized actual request JSON}"
: "${READOUT_REQUEST_SHA256:?bound request SHA256}"
: "${READOUT_TICKET:?actual admitted owner ticket JSON}"
: "${READOUT_TICKET_SHA256:?actual ticket SHA256}"
: "${READOUT_GPU_PROBE:?actual oneGPU probe JSON}"
: "${READOUT_MODEL:?unchanged Motus model path}"
: "${READOUT_NORM:?unchanged normalization file}"
: "${READOUT_ROBOTWIN:?qualified native RoboTwin source}"
: "${READOUT_RLINF:?qualified RLinf source}"
: "${READOUT_OUTPUT:?fresh runtime output}"
[[ "$(uname -s)" == Linux ]]
cd "$READOUT_SOURCE"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$READOUT_SOURCE/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$READOUT_PYTHON" scripts/diagnostics/robotwin_command_diagnostic_readout.py \
  --request "$READOUT_REQUEST" --request-sha256 "$READOUT_REQUEST_SHA256" \
  --ticket "$READOUT_TICKET" --ticket-sha256 "$READOUT_TICKET_SHA256" \
  --gpu-probe "$READOUT_GPU_PROBE" --model-path "$READOUT_MODEL" --norm-stats "$READOUT_NORM" \
  --robotwin-root "$READOUT_ROBOTWIN" --rlinf-root "$READOUT_RLINF" --output "$READOUT_OUTPUT"
