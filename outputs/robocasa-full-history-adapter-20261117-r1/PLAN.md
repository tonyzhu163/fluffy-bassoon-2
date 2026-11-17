# Full history adapter: first native meta-learning test

Written November 17, 2026. **Plan only; no jobs submitted.** The user requested
the plan before execution. Code reviewed at main `6f4b5ce0`; actual implementation,
validation, source pin, data manifests and allocations must be recorded before
launch. This is a new phase, separate from the closed 8 GPU-hour pilot and
230 GPU-hour campaign.

## Question and scope

Can a trained history transformer steer the **frozen pi0.5 action expert through
its KV adapter** so that a second real attempt succeeds more often because of
experience from the first attempt, with no test-time weight updates?

Start on RoboCasa OpenDrawer. The [small native pilot](../robocasa-feedback-advantage-20261117-r1/RESULT.md)
established an information advantage in a deliberately controlled calibration
problem. Its readers selected among recorded correction outcomes; it did not
establish that the full VLA adapter can produce useful adapted actions. The
[earlier multi-task comparison](../robocasa-six-task-scale-20261113-r1/RESULT.md)
remains negative in aggregate.

The immediate experiment trains **DPT and an equally trained current-only
adapter**, then tests them in fresh closed-loop rollouts. DPT supplies a strong
supervised positive control before introducing value-estimation uncertainty.
Broad PPO collection and another four-method sweep remain deferred. A positive
result here establishes a mechanism in one task family, not representative
meta-learning or held-out-task generalization.

## 1. Preserve the working adaptation problem

Use the pilot's **uncorrected first attempt with motion feedback** condition.
Both small DPT and DIT readers reached 100% on its small recorded held-out panel.
This is a choice of the easier information condition, not a claim that reward
alone is sufficient. The action/reward-only condition can follow if the full
adapter works.

- Frozen public Human300 pi0.5 motor, native RoboCasa rewards, OpenDrawer,
  600 physical controls per episode, predict 50 controls and execute 5 before
  replanning, 10 denoising steps.
- Hidden lateral OSC calibration offset `z = -0.04 m` or `+0.04 m`, constant
  across the two attempts. Retain the pilot's approach trigger and 100-control
  window, ending at first grasp. The trigger uses simulator geometry; this
  remains a disclosed privileged experimental scaffold.
- First attempt: the same uncorrected frozen SFT policy for every comparison.
  Second attempt: reset the scene, preserve the hidden calibration and supply
  the recorded first attempt to the history-conditioned learner.
- Current inputs are cameras, public robot state and the original instruction.
  History contains prior public observations, actually executed controls,
  native rewards and episode boundaries. No hidden calibration, simulator
  geometry, teacher identity, current target or future outcome enters the
  learner. Public motion may reveal calibration during either attempt.
- The student supplies robot commands through the frozen action expert. It
  receives no analytic correction after inference. Analytic compensation is
  allowed only for the explicitly named privileged teacher/reference.

This first test is **same-scene retry**. Changing scenes between support and
query is a subsequent transfer test, not mixed into the primary estimate.

## 2. Qualify the full interface and collect suitable data

Reuse [the KV bridge](../../src/fluffy_bassoon/meta_rl/flow_history.py) and
[its frozen-expert training path](../../src/fluffy_bassoon/meta_rl/flow_bridge_train.py),
with a new run-specific collector and trial sampler. The compact pilot archives
lack the saved camera inputs needed for these fits; they are development
evidence, not a substitute visual training corpus.

Before bulk collection, use development scenes to resolve these concrete risks:

1. **Runtime and actuator parity.** Verify the actual allocated GPU, native
   episode, zero-initialized adapter parity and deterministic reset pairing.
   Use one qualified accelerator/runtime family across collection, fitting and
   evaluation; do not repeat the pilot's hardware/split confound.
2. **Teacher target correctness.** Query the frozen motor at the exact saved
   query observation. The privileged teacher compensates the known OSC offset
   in native control space, respecting the controller gain, clipping and active
   window. Record the mapping from model space to issued and executed controls.
   Demonstrate real corrected rollouts and quantify clipping; an inverse-sign
   label alone is not evidence that the motor can execute it.
3. **Executed action semantics.** Save the actual five-control execution and
   its mask separately from the proposed 50-control chunk. Define a causal,
   masked representation for history. Supervise only validated teacher command
   coordinates and timesteps; do not invent targets for an unexecuted suffix.
   The existing bridge needs an explicit loss-mask extension for this protocol.
   Resolve changes of the calibration window within a chunk in the teacher
   execution/label contract before fitting. Preserve native 32-dimensional
   normalization, physical 12-dimensional controls and binary action mapping.
4. **Learning path.** Demonstrate finite gradients reaching the history model
   and KV adapter through the frozen expert, unchanged motor tensors, and a
   small development fit that reproduces distinct teacher corrections. Compare
   decoded commands and native behavior, not training loss alone.

These are validity checks, not a new success-percentage admission rule. If the
teacher cannot correct the perturbation, or the interface cannot represent the
targets, diagnose that before spending the main training budget. Run applicable
existing core tests and targeted numerical checks on allocated Linux resources;
add a core regression test only if a substantive implementation fix needs it.

### Trial split and collection

Target **32 training, 8 development and 32 fresh held-out scenes**, with both
calibration signs in each scene. A scene, including all its attempts and signs,
belongs to exactly one split. Old pilot scenes, including its old held-out
scenes, are development-only and cannot enter the new confirmatory holdout.

Before collection, publish ordered, disjoint candidate seed lists and their
hashes. Screen at most 128 training, 64 development and 128 test candidates.
Eligibility is nominal SFT success and a valid approach trigger, as in the
pilot; never select using perturbed, oracle or learned-adapter outcomes. Keep
all screening counts and all failed qualified trials. Report that the result
is conditional on this nominal-solvable subset. If a quota cannot be filled,
report the smaller fixed set rather than replacing cases after reading results.

For each training/development scene and sign, collect one uncorrected support
episode and one fresh privileged-teacher query episode from the same reset.
This gives up to **160 support/query episodes** across the 40 scenes, plus
nominal screening. Preserve exact query images, public states, teacher labels,
executed control masks, terminal feedback and reset/noise identities. A failed
teacher episode remains included. Stop each episode at native success or the
declared horizon; there are no post-success training queries.

## 3. Train the actual adapter, with a matched control

| Setting | Planned value |
| --- | --- |
| Interface | History decoder to learned KV residuals over the frozen VLA prefix; no direct correction selector or velocity bypass |
| Decoder | Width 256, 6 layers, 8 heads |
| KV adapter | Rank 4, scale 0.1, all qualified prefix layers, zero output initialization |
| Context | 256 query tokens: two episodes of at most 120 decisions plus boundaries; verify terminal support feedback is retained |
| Training | DPT teacher-action flow matching, logical batch 8, AdamW learning rate 0.0003, weight decay 0 |
| Fits | Full-history DPT and current-only DPT, each with seeds 0, 1, 2: six fits |
| Endpoint | Fixed 4,096 updates; development curves at 1,024/2,048/4,096, final checkpoint used for confirmatory evaluation |

The current-only control has the same trainable architecture, labels, query
draws, initialization seeds and optimizer budget, but receives no previous
observations/actions/rewards at training or evaluation. Preserve the complete
support episode in every full-history query window; the old generic sampler's
optional one-episode window is unsuitable for this primary comparison.

Sample query positions without replacement within scene/sign strata before
reshuffling; record coverage and multiplicities. At most 7,680 query positions
come from 32 training scenes, two signs and 120 decisions. The 32,768 query
draws allow repeated coverage of this small corpus. This endpoint is a bounded
first fit, not a guarantee of convergence. Normalization uses training only.
Measure execution batch capacity and update time without changing the logical
batch or objective. If development shows unresolved underfitting, report that
and amend the plan before opening confirmatory outcomes.

## 4. Evaluate fresh, paired native attempts

The target panel has **64 calibration cases in 32 independent scenes**. Every
case shares a real first-attempt support trajectory across arms. All second
attempts start from the same verified scene state and use matched inference
noise where supported. Run each policy closed loop; its own subsequent
observations/actions feed its own query-episode memory. No recorded outcome
lookup scores the learned adapter.

| Second-attempt arm | Purpose |
| --- | --- |
| Full DPT, three fit seeds | Does the full adapter turn support experience into successful action? |
| Same DPT with support cleared, three seeds | Causal effect of previous-attempt history; retain normal memory within the query episode |
| Separately trained current-only DPT, three seeds | Does history beat equal supervised training and model capacity? |
| Frozen SFT and existing OpenDrawer PPO30 | Absolute comparison with the two previously requested policy stages; no new PPO training |
| Privileged inverse-calibration teacher | Can this motor/controller solve the qualified case? |
| Compact motion-history reader | Does the simpler history mechanism reproduce in fresh closed-loop execution? |

Retrain the compact reference on the same new training split with its previously
fixed architecture/endpoint; it is a diagnostic reference, not an extra VLA
method. Verify the archived PPO30 identity against the existing
[baseline contract](../robocasa-six-task-scale-20261113-r1/BASELINE-CONTROLS.json).
The complete primary panel costs up to **832 query episodes plus 64 shared
support episodes**, excluding nominal screening. Profile this before reserving
the full panel. If it cannot fit the phase budget, change the case count using
timing alone before inspecting held-out policy outcomes, preserving all arms
and three fit seeds.

On a predeclared 16-scene subset, budget permitting, add opposite-calibration
support from the same scene for each DPT fit (96 query episodes). This tests
whether the content of experience changes behavior. It is deliberately
contradictory history, not an unrelated random donor. Reward masking alone
cannot establish a negative result here because motion also carries information.

Primary outcomes are paired second-attempt success differences: full DPT minus
its support-cleared version, and full DPT minus separately trained current-only
DPT. Also report SFT/PPO30 differences, first-attempt success, cumulative
first-or-second success, physical controls consumed, clipping and behavior at
the first adapted decision. Separate within-query recovery from cross-attempt
adaptation. Average fit seeds within scene, report each seed, and estimate
paired uncertainty by resampling whole scenes; signs, repeated fits and shared
support are not independent extra samples. Include success discordances and
intervals even when the result is inconclusive. Thirty-two scenes target the
large pilot-sized effect; they cannot rule out small useful benefits.

## 5. Budget, storage and delivery

New phase ceiling: **48 GPU-hours**, with cumulative usage recorded separately
from closed campaigns. Stage amounts are planning reservations, not guaranteed
runtime: qualification 4, collection/labels 6, fits 20, evaluation 14, recovery
reserve 4 GPU-hours. Stop at the total ceiling with resumable artifacts; no
automatic budget extension. The reserve covers failed attempts as well as
retries. Dependent stages follow qualification, not a blanket concurrent launch.

Baige is the target. At launch inspect live queue capacity, limits and actual
accelerator type; prefer up to four independent one-GPU workers on qualified
RTX 6000-class allocations for fits and evaluation shards. Do not require a
distributed fit. Use the in-job GPU probe and bind inference/rendering to the
allocation. Review latest main, commit the implementation and validate it,
then deploy immutable commit-pinned worktrees through Git.

Check project NVMe capacity/quota and a write probe first. Save lossless policy
input images only at decision points, not video at every simulator step; share
the corpus and frozen weights across fits. Profile bytes per episode before
bulk collection and stop safely if storage cannot accommodate the remaining
contract. Retain initial/final/evaluation checkpoints plus two complete recovery
boundaries with optimizer, sampler and RNG state. Archive and hash-readback
completed raw shards before removing their hot copies; protect all active
dependencies. Small manifests/results enter Git, large payloads do not.

Target a first decision within **three days of allocated resources becoming
available**: day 1 interface/teacher qualification and collection; day 2 fits;
day 3 paired evaluation and report. This is a target, not an assumption of
queue availability. Verify advancing updates after launch and monitor every
20 minutes, recording job IDs, source, progress, last check and next action in
the single coordination board.

## Decision and path to the full meta-learning idea

| Observation | Next decision |
| --- | --- |
| Teacher or simple native reference fails | Repair the experiment/runtime; this does not test the large adapter hypothesis |
| Simple reference works, full adapter cannot fit/execute teacher actions | Diagnose labels, normalization, KV influence and optimization; do not collect more PPO |
| Full adapter imitates training targets but lacks held-out history benefit | Examine closed-loop shift and history use; stop scaling this recipe until the cause is resolved |
| Held-out gains over both controls recur across fits, with useful history interventions | Proceed to a representative adaptation distribution; quantify uncertainty rather than declaring success from one scene or a pooled percentage |
| Intervals still admit both useful improvement and no benefit | Call the result inconclusive; propose additional paired scenes rather than a bigger teacher campaign |

A positive bridge result is followed by a separate, costed distribution plan:

1. Choose task families for coverage of articulated opening/closing, grasp and
   placement, spatial targets and contact conditions—not because jobs already
   exist. For each, identify persistent uncertainty for which previous attempts
   can actually help. Qualify native information/reference controls first.
2. Freeze training families and distinguish **ID new scenes**, **OOD condition
   values/combinations**, and **OOD held-out task families**. Hold all histories
   from a task/scene/condition unit in the same split. Record the frozen base
   motor's pretraining exposure separately from meta-training exposure.
3. Collect a shared chronological corpus, then compare **AD, DPT, DIT and
   IC-IQL** with equal architecture, data access, fit budgets and paired native
   evaluations, alongside SFT and PPO references. DPT needs query-aligned
   teacher labels; AD needs a genuinely changing learning trajectory; DIT and
   IC-IQL need separately validated Q/V and effective action weighting.
4. Add PPO teachers only where that corpus requires better actions or genuine
   learning trajectories. Teacher quality and same-state action ranking should
   justify the collection before scaling it.

Those later phases are **not included in this 48 GPU-hour reservation**. Their
task list, budget and launch decision depend on the bridge result. The present
deliverable is this plan; execution starts in a subsequent step.
