# Flow-SDE PPO bug audit

## Confirmed: rollout and replay use different critic predictions (P1)

All five completed 10M-control sources used `value_after_vlm: false`, ten
denoising steps and the single-step likelihood path (`joint_logprob` defaults
to false). The deployed OpenPI code computes:

- **Rollout:** the mean of ten suffix critic predictions, one from every
  denoising state. This becomes `prev_values` for GAE and PPO value clipping.
- **PPO replay:** the critic prediction from only the selected denoising index.

These are different functions even at identical weights. PPO's value clip is
therefore centered on a different prediction from the one being trained.
The pre-SGD qualification checks likelihood parity but does not check critic
parity, so it cannot detect this discrepancy.

An allocated-Linux CPU reproduction executes the unmodified ASTs of the
deployed `_predict_train` and `default_forward` methods, plus their real sampler
and value helpers. It replaces expensive neural-network features with a
deterministic time feature and an identity linear value head. With those
features, rollout always returns 0.55; replay returns 1.00 at index 0 and 0.10
at index 9. All ten indices have **exactly zero log-probability difference**
and nonzero value differences. The largest value gap is 0.45, and 12/20
synthetic samples exceed the configured 0.2 value clip before any SGD. A
prefix-critic control has zero value gap.

The synthetic gaps demonstrate the implementation defect; they are not
measurements of production-model value errors or production clipping frequency.
The bug affects the critic contract in all five configurations, but this audit
does not establish that it caused their learning failures or the Drawer
regression.

**Repair before further PPO:** make rollout and replay evaluate the same
critic definition, and enforce unchanged-weight value parity as well as
likelihood parity. Selecting the same suffix value during rollout is one
candidate; consistently replaying the ten-value average is another. A prefix
critic is a separate architectural choice with a different head shape, not a
safe flag change when resuming these checkpoints. Existing runs were not
modified and no replacement training was launched.

Evidence: [deployed source snapshot](deployed-openpi-rl.py) lines 268–296 and
451–507; [source/configuration hashes for all five tasks](SOURCE.json);
[executable reproduction](REPRODUCE.py); [CPU reproduction results](REPRODUCTION.json).

## Diagnostic interpretation and reporting issues (P2)

The original one-cohort-per-task drift comparison does not match denoising
indices: Drawer uses 9, OpenDrawer 0, Coffee 1, and Toaster/Stove 4. The
corresponding transition standard deviations are approximately 0.03162,
0.30000, 0.28460 and 0.11619. This makes raw cross-task Gaussian log-density
drift incomparable as a measure of equal parameter movement or behavior.
Drawer's inverse variance is 90 times OpenDrawer's; this does not by itself
predict a 90-fold difference in observed drift because residuals and states
also differ. The existing dimension-drift readout already records this
limitation. No evidence of an invalid or off-by-one denoising index was found.

The active pinned dimension diagnostic also calls `describe` on flattened
`action` [16, 600] and `model_action` [16, 1600] arrays. Its legacy
`cohort.action` / `cohort.model_action` last-axis statistics therefore describe
flattened time-coordinate positions, not twelve physical action dimensions.
The separate [cohort readout](../robocasa-flow-sde-dimension-drift-20261104-r1/COHORT-READOUT.json)
reshapes them correctly. This reporting issue does not change the saved
[16, 50, 12] log-probability tensors or the verified dimension decomposition.

## Other checks

The Linux CPU check agrees with an independent scalar noise-scale formula at
all ten valid indices. Native action mapping thresholds indices 6 and 11 at
0.5, preserves all continuous coordinates and does not mutate recorded PPO
actions. Reward inspection confirms the first-success reward reads the
previous latch before the environment updates it; the existing transient
success regression passes. No additional bug was established in these paths.

The existing Flow-SDE, native-PPO and physical-pairing suites pass **37 tests**
on the allocated Linux runtime. They do not contain the suffix critic parity
case. [Test receipt](TESTS.json). This audit used CPU only: no model weights,
simulator episodes, SGD, new GPU allocation or active worktree edits.
