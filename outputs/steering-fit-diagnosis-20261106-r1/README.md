# Stage 1 steering-fit diagnosis and fix (2026-11-06)

Claude Code session (tonyzhu163). Analysis of
[STAGE1-RESULT](../robocasa-three-task-kv-velocity-meta-20261105-r1/STAGE1-RESULT.md);
code fix `bc56b849` on `main`. No GPU used.

## Observation and hypothesis

Observed: AD KV/velocity query-0 success counts (of 50) on
Drawer/Toaster/Coffee/Fridge/Microwave are close to the native SFT panel:
native SFT 35/15/21/33/6, KV-AD 34/14/20/31/10, velocity-AD 35/15/21/33/8
([native panel](../../docs/FINDINGS.md)), and full-history and reset curves
are similar.

Hypothesis, not yet measured: the fitted steering is too small to change
behaviour, so these readouts largely re-measure the native policy. Similar
marginal counts do not show identical actions or per-case outcomes. Test this
with `steering_effect` on the eight actual candidates and a controlled
continuation/refit before treating it as the explanation.

Verified separately (from logs): `advance_plateau` used
`relative_delta * max(1., abs(reference))`. With flow losses near 0.13 this is
an absolute 1e-3 (about 0.8%) per patience window, so the fits stopped early.
From [VALIDATION-FINAL08.json](../robocasa-three-task-kv-velocity-meta-20261105-r1/monitor/VALIDATION-FINAL08.json):

- KV-AD fell monotonically 0.130915 → 0.129984 over its last 10 checks
  (0.00093 total) and was declared converged at update 2,900. KV DPT/DIT/IC-IQL
  show the same tail.
- All four velocity fits stopped at exactly 1,900 = minimum 1,000 + 10 checks;
  their reference never moved from the first check.
- Fits saw 0.03–0.15 equivalent epochs and 0.5–11% held-out loss reduction.

Contributing: zero-initialized adapters at scale 0.1 / rank 4, logical batch 8,
gradient norms ~1e-3, and a velocity correction that is one bounded vector
broadcast across all 50 chunk steps. Separately, the source corpus may carry
little in-window improvement (fixed130 teachers are +2 to +8 pp over SFT over
the whole PPO run), which limits AD even when fitted.

## Fix (`bc56b849`)

- `advance_plateau` threshold is now `relative_delta * |reference|`. The
  recorded KV-AD tail is no longer converged; a 0.05%-per-check flat tail is.
  Behaviour for |loss| ≥ 1 is unchanged.
- `flow_bridge_train.steering_effect` measures, on held-out batches at
  identical noisy actions: native vs steered loss, `loss_reduction`,
  `relative_steering` (RMS steered−native over RMS native velocity) and
  `history_fraction` (full−reset over total steering). `steering_gate`
  admits a candidate only above contract thresholds fixed before readout.

Existing runs are unaffected: they are pinned to their own commits.

## Review follow-up (2026-11-06)

[Review](../steering-fit-review-20261106-r1/README.md) of `933716fb`:

- Reset input (fixed): the gate's episode-reset pass kept the previous
  episode's action/reward/done in start tokens. `episode_reset_batch` now
  zeroes them as deployment and `trial_tokens(mode="episode")` do, with a
  regression test.
- Enforcement (open, by design for the next run): the helpers have no
  production caller. The next refit contract must bind thresholds before
  readout and require a candidate-bound gate result at dispatch. Completed
  runs keep their pinned scripts and contracts.
- Causal wording (fixed above): negligible steering is a hypothesis until measured.

## Linux checks requested on Baige (CPU only, no GPU)

macOS checks: 25/25 `tests/test_flow_bridge_train.py` +
`tests/test_flow_history.py` pass; both plateau tests pass in isolation
(`tests/test_residual_history.py` cannot import locally without
`gymnasium`). Pending on a Baige CPU checkout at `bc56b849` or later:

```bash
uv run pytest -q tests/test_flow_bridge_train.py tests/test_residual_history.py tests/test_flow_history.py
```

Record the log here as `LINUX-TESTS.log`.

## Next steps (not launched)

1. Keep the six held Stage 1 readouts held until the candidates are measured.
2. Run `steering_effect` on the eight existing candidates (CPU-feasible on a
   subset, or one short GPU slice) to test the negligible-steering hypothesis.
3. Check source streams for success improvement within 12-episode windows.
4. Refit KV with the corrected plateau rule, gate it, then read out.
