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

## Effect on the loss and consequences for existing runs

The follow-up [actual loss-registry probe](CRITIC-LOSS-PROBE.json) runs the
deployed `policy_loss` entry point, including embodied preprocessing and the
registered actor/critic loss, on Linux CPU. With a selected-step prediction
0.9, rollout reference 0.55, target 1.0 and clip 0.2, the critic gradient is
zero because the clipped-reference branch dominates. Anchoring the same
prediction to its own unchanged-policy value gives a nonzero gradient toward
the target. A second case with prediction 0.1 and target zero shows the same
mechanism in the opposite direction. Thus preprocessing does not remove this
value inconsistency, and it can suppress useful critic learning. These
synthetic cases do not measure the trained models' gaps or counterfactual
learning outcomes. [Executable probe](CRITIC-LOSS-PROBE.py).

Update-level metrics cannot attribute clipping to this issue: they combine
the initial value-definition gap with actual parameter updates over four
epochs. For context, Drawer's logged value-clip ratio averages 2.83% across
the campaign and 2.36% over updates 121–130; its logged explained variance
averages 0.250 overall and 0.280 over that last window. The critic was not
uniformly failing to fit returns, but these metrics neither qualify value
parity nor rule out a material learning penalty.

There is no evidence-based requirement to restart the entire campaign now.
The retained observations, actions, videos, rewards, accounting and measured
checkpoint success remain observations of the implementation that actually
ran. They cannot be relabeled as outcomes of a corrected PPO learner. A
corrected-method training claim requires repeating the relevant affected
training condition; model conversion, simulator setup and all historical
artifact collection do not need to be redone merely because of this defect.

The smallest useful next validation is an unchanged-policy replay on saved
production batches to measure actual value gaps, baseline offsets and their
effect on critic gradients, followed by one short controlled old-versus-fixed
training comparison if needed. Broader training reruns should follow that
evidence. No such model replay, comparison or replacement training was
launched by this audit; previous synthetic numbers are not production error
estimates.

## Upstream source and reporting status

Checked Nov4: the deployed file is byte-identical to official RLinf's file,
Git blob `9dafb7f1ced52ce3bd18b2f432ec710e61e5e5c4`, and remains unchanged at
current upstream main `c716f625f8475f9be9fbc98941812bf95270ac08`.
The mismatch was already present in the Sep14
[OpenPI refactor PR #1512](https://github.com/RLinf/RLinf/pull/1512) under
`openpi_rlinf/tasks/rl.py`; the Sep22 rename
[PR #1580](https://github.com/RLinf/RLinf/pull/1580) moved it to the current path.
The three local runtime commits after the official Oct30 base did not modify
this file; the deployed vendor checkout has no tracked changes.

Primary evidence is the official
[rollout average](https://github.com/RLinf/RLinf/blob/c716f625f8475f9be9fbc98941812bf95270ac08/rlinf/models/embodiment/openpi/tasks/rl.py#L268-L296)
and [single-step replay](https://github.com/RLinf/RLinf/blob/c716f625f8475f9be9fbc98941812bf95270ac08/rlinf/models/embodiment/openpi/tasks/rl.py#L450-L507),
together with our unchanged-weight reproduction. No matching public issue or
maintainer confirmation was found in the searches recorded in
[UPSTREAM.json](UPSTREAM.json). We have not filed an external issue.

[Issue #1568](https://github.com/RLinf/RLinf/issues/1568) is a different critic
report about `[B]` versus `[B, 1]` broadcasting. It was closed after a
[reproduction comment](https://github.com/RLinf/RLinf/issues/1568#issuecomment-5780797446)
showed that loss preprocessing aligns those shapes. It does not report or
confirm this ten-values-versus-one-value discrepancy.

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
