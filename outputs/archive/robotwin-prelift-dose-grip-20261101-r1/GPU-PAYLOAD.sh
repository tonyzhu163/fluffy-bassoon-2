#!/usr/bin/env bash
# Existing owner wrapper ONLY; never allocates. Require qualified6397 begin/finish,
# NEW slice840/reserved900/start-derived deadline780, allow_short_slice=True.
# Actual allocation: mode robotwin_phone04_prelift_dose_grip_factorial_v1,
# capacity_action_bound1600, prior_a1/phase2988, retained_previous_pair_gpu_seconds336.
# Parent verifies closed E head, CPU composite PASS/preflight, renderer and real oneGPU probe.
set -Eeuo pipefail
: "${TASK_CODE:?}" "${TASK_PYTHON:?}" "${TASK_SOURCE:?}" "${TASK_RUN_ID:?}" "${TASK_OUTPUT:?}"
: "${TASK_PROFILE:?}" "${TASK_PROFILE_SHA256:?}" "${TASK_CONTEXT:?}" "${TASK_CONTEXT_SHA256:?}"
: "${TASK_SCOPE:?}" "${TASK_SCOPE_SHA256:?}" "${TASK_ALLOCATION:?}" "${TASK_ALLOCATION_SHA256:?}"
: "${TASK_INTERFACE:?}" "${TASK_INTERFACE_SHA256:?}" "${TASK_JOINTS:?}" "${TASK_JOINTS_SHA256:?}"
: "${TASK_DEMOS:?}" "${TASK_DEMOS_SHA256:?}" "${TASK_MODEL:?}" "${TASK_NORM:?}"
: "${TASK_ROBOTWIN:?}" "${TASK_RLINF:?}" "${TASK_GPU_PROBE:?}"
cd "$TASK_CODE"
exec "$TASK_PYTHON" scripts/diagnostics/robotwin_prelift_probe.py \
  --profile "$TASK_PROFILE" --profile-sha256 "$TASK_PROFILE_SHA256" \
  --execution-context "$TASK_CONTEXT" --execution-context-sha256 "$TASK_CONTEXT_SHA256" \
  --scope "$TASK_SCOPE" --scope-sha256 "$TASK_SCOPE_SHA256" \
  --allocation "$TASK_ALLOCATION" --allocation-sha256 "$TASK_ALLOCATION_SHA256" \
  --interface "$TASK_INTERFACE" --interface-sha256 "$TASK_INTERFACE_SHA256" \
  --joint-conventions "$TASK_JOINTS" --joint-conventions-sha256 "$TASK_JOINTS_SHA256" \
  --demos "$TASK_DEMOS" --demos-sha256 "$TASK_DEMOS_SHA256" \
  --source-commit "$TASK_SOURCE" --run-id "$TASK_RUN_ID" --output "$TASK_OUTPUT" \
  --model-path "$TASK_MODEL" --norm-stats "$TASK_NORM" \
  --robotwin-root "$TASK_ROBOTWIN" --rlinf-root "$TASK_RLINF" --gpu-probe "$TASK_GPU_PROBE"
