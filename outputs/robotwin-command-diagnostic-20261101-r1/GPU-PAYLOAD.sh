#!/usr/bin/env bash
# Existing-owner registered inclusive GPU attempt; no begin/finish/submission.
set -euo pipefail
: "${OPERATION:?cache04 or fit}"
: "${KIND:?cache04 or action or kv}"
: "${ACTUAL_REQUEST:?hash-bound real bank/cache/profile request}"
: "${REQUEST_SHA:?exact SHA256}"
: "${ACTUAL_TICKET:?actual admitted source/host/clock ticket}"
: "${TICKET_SHA:?exact SHA256}"
: "${ACTUAL_PROBE:?actual selected-GPU CUDA proof}"
: "${MOTOR:?original motor path}"
: "${NORM:?original norm file}"
: "${ROBOTWIN:?qualified runtime}"
: "${RLINF:?qualified runtime}"
: "${FRESH_OUTPUT:?phase/reactive-curve-v1/KIND/fresh-child}"
if [[ "$OPERATION" == cache04 && "$KIND" == cache04 ]]; then
  entry=scripts/diagnostics/robotwin_command_diagnostic_cache04.py
elif [[ "$OPERATION" == fit && ( "$KIND" == action || "$KIND" == kv ) ]]; then
  entry=scripts/diagnostics/robotwin_command_diagnostic_curve.py
else
  exit 2
fi
python "$entry" --kind "$KIND" \
  --request "$ACTUAL_REQUEST" --request-sha256 "$REQUEST_SHA" \
  --ticket "$ACTUAL_TICKET" --ticket-sha256 "$TICKET_SHA" \
  --gpu-probe "$ACTUAL_PROBE" --model-path "$MOTOR" --norm-stats "$NORM" \
  --robotwin-root "$ROBOTWIN" --rlinf-root "$RLINF" --output "$FRESH_OUTPUT"
