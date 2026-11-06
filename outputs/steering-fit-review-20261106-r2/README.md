# Review response and Linux verification

Pulled through `cd6d6e33f93eb8e47d875ccc65d33db381dd5404` on
2026-11-06 Asia/Shanghai, preserving all 13 existing dirty files.
The response to the [initial review](../steering-fit-review-20261106-r1/README.md)
addresses its findings as follows:

- **Reset leakage fixed in `aa74162d`.** `episode_reset_batch` clones the
  full-history tokens and zeros previous-action/reward/done fields only at
  episode starts. `steering_effect` uses this copy for its reset comparison;
  the full-history input stays unchanged. The new two-interface regression
  checks boundary clearing and independence from previous-episode feedback.
- **Causal wording corrected.** The diagnosis now distinguishes observed
  success counts from the unmeasured negligible-steering hypothesis and
  requests direct measurements of the eight candidates.
- **Enforcement explicitly remains open.** The helpers still have no
  production caller. The next refit contract must bind prospective thresholds
  and require a candidate-bound gate result at dispatch. This remains a
  prerequisite to a new readout; completed runs keep their pinned contracts.

On Baige Linux development container `aibox-rffe453e3884-5f96c78b88-rtvhb`,
an incremental Git bundle supplied the exact `cd6d6e33` source to a fresh
isolated snapshot. The existing Python environment was reused unchanged,
CUDA was hidden, and CPU thread counts were bounded to two. PFS had
360.94 GiB free and passed a write probe; the quota utility was unavailable.

The requested three files (`test_flow_bridge_train.py`,
`test_residual_history.py`, `test_flow_history.py`) passed **33 tests in
76.05 seconds**. See the complete [Linux log](LINUX-TESTS.log).

The independent [original reproduction repeated with the fix](RESET-FIX-VERIFICATION.json)
now gives exactly equal reset contexts for both interfaces when only the
previous episode's feedback changes (maximum difference 0). The full-history
tokens remain byte-for-byte unchanged. Without boundary clearing, the same
fixture still reproduces the earlier context differences of 3.6439743 for
KV and 2.0232000 for velocity.

The reset patch addresses the reported input-control mismatch. It does not
establish material steering, a causal explanation of the robot results, or
readout admission. No GPU allocation, fitting or simulator episode was
launched; the six held readouts remain held.
