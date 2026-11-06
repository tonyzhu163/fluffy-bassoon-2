# Stopping-rule and steering-gate review

Reviewed `bc56b849` and diagnosis `933716fb` on 2026-11-06 Asia/Shanghai.
The main checkout was fast-forwarded from `7794733e` to `933716fb`; all
13 pre-existing dirty files were preserved, with only the upstream diagnosis
row added to the existing coordination edits.

The relative stopping threshold is corrected: at reference loss 0.13 and
relative delta 0.001, a material improvement is approximately 0.00013,
rather than 0.001. The recorded declining KV-AD tail and a flat tail are
covered by the regression tests. This does not establish the cause of the
robot results or change the contracts of the completed commit-pinned fits.

## Findings

1. **P1 — Gate enforcement is missing.** Repository-wide call-site inspection
   finds `steering_effect` and `steering_gate` called only by their tests.
   The existing frozen-bridge readout checks `converged` and
   `stop_reason == validation_plateau`, then loads the candidate; it does
   not require a gate result. No production thresholds are bound in the
   run contract. The helpers are useful diagnostics, but the advertised
   pre-readout admission is not implemented. A new run needs a caller,
   prospective thresholds and a candidate-bound result checked at dispatch.
   Preserve the completed runs' pinned scripts and contracts.

2. **P2 — The gate's reset input retains previous-episode feedback.**
   `steering_effect` passes the same full-history token batch to
   `query_context` with `mode=episode`. `encode_sequence` clears the cache
   at episode starts, but does not clear token action/reward/done fields.
   Real episode-reset deployment and `trial_tokens(mode=episode)` clear
   those fields. The resulting `history_fraction` and reset loss therefore
   compare against a different control. Clear boundary feedback in a copy
   of the reset tokens while retaining the full-history input.

3. **P2 — The diagnosis states a causal conclusion before measuring it.**
   Similar marginal success counts do not establish that the learned and
   native policies produce identical actions, nor that negligible steering
   explains the full/reset comparison. The diagnosis itself lists measuring
   the eight actual candidates as future work. Keep negligible steering and
   the stopping rule's contribution as hypotheses until direct measurements
   and a controlled continuation/refit support them.

## Verification

On the existing Baige Linux development container
`aibox-rffe453e3884-5f96c78b88-rtvhb`, the source was deployed through an
incremental Git bundle and extracted from exact commit
`933716fb499e8d76b46bedc3b179c791d0ac536e` into a fresh isolated test snapshot.
The shared checkout and active run code were unchanged. Shared PFS had
362.16 GiB free and passed a write probe; no quota utility was available.
The existing Python environment was reused without dependency changes.

With CUDA hidden and CPU thread counts bounded to two,
`tests/test_flow_bridge_train.py`, `tests/test_residual_history.py` and
`tests/test_flow_history.py` passed **31 tests in 31.10 seconds**.
The requested [Linux log](../steering-fit-diagnosis-20261106-r1/LINUX-TESTS.log)
is saved alongside the original diagnosis.

A separate CPU [reset reproduction](RESET-REPRO.json) fixes the query at an
episode start, changes only previous-action/reward/done token fields and
compares episode-mode contexts. Maximum differences were 3.6439743 for KV
and 2.0232000 for velocity. Clearing those boundary fields produced exactly
equal contexts for both interfaces. This verifies the input-control mismatch;
it does not measure the eight trained candidates or robot performance.

No GPU allocation, fitting, simulator episode, or held readout was launched.
