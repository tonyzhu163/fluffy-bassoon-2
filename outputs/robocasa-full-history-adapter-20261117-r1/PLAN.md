# Full history adapter: in-context recalibration of a frozen VLA

Written November 17, 2026. Revised the same day to merge the duplicate
closed-loop bridge plan and the agreed scope changes. **Execution authorized:**
the user approved up to **200 GPU-hours** for this campaign on November 17.
Implementation is starting; jobs and usage are recorded in EXECUTION.md.
Code was reviewed at main `dae30718`. The actual implementation,
validation, source pin, data manifests and allocations must be recorded before
launch. This is a new phase, separate from the closed 8 GPU-hour pilot and the
230 GPU-hour campaign.

Record this phase lightly, following AGENTS.md: a plan, the runs and a result
note. Full audits are reserved for the confirmatory panel in §4.

## Method

**Problem.** A frozen VLA meets a hidden, persistent condition that the
current observation does not reveal. Here that condition is a miscalibration
of the commanded end-effector position. The condition shows itself only
through what happens during an attempt.

**Method: privileged-teacher distillation into a history adapter.**

- **Training.** A teacher that knows the hidden condition produces the
  corrected action at each saved query state.
- **Student.** A history-conditioned KV adapter on the frozen π0.5 learns to
  reproduce the teacher's action from public experience only: earlier
  observations, executed controls, native rewards and episode boundaries.
- **Deployment.** No weight updates and no privileged input.

The training distribution is part of the method. Its conditions vary across
trials. Earlier attempts can reveal it before the next attempt; observations
within the current attempt may also reveal it, which the cleared-history
control measures.
Test conditions include magnitudes and directions never seen in training.
That separates in-context inference from memorizing a few fixed corrections.

Objective ablations (AD, DIT, IC-IQL) and interface ablations (velocity bridge)
come after the primary result. They are not part of this phase.

## Question and scope

Does the trained adapter, run closed loop, make later attempts succeed more
often *because of* earlier experience? Is the effect still present for
held-out condition values, and how does it compare with online RL on the same
condition?

The [small native pilot](../robocasa-feedback-advantage-20261117-r1/RESULT.md)
established an information advantage in one lateral calibration problem. Its
readers selected among recorded correction outcomes. The
[earlier multi-task comparison](../robocasa-six-task-scale-20261113-r1/RESULT.md)
remains negative in aggregate.

This phase covers OpenDrawer only. If it is positive, the same pipeline
extends to two or three more RoboCasa tasks that the nominal SFT policy can
solve. That extension is costed separately (see the last section). The
campaign hard cap is 200 GPU-hours, including qualification, failures, retries,
PPO comparison and any contingent task/objective extension. Closed campaigns
are not reopened. This is a ceiling, not a spending target.

## 1. The adaptation problem

- **Motor and episodes.**
  - Frozen public Human300 π0.5 motor, native RoboCasa rewards, OpenDrawer.
  - 600 physical controls per episode.
  - Predict 50 controls, execute 5 before replanning; 10 denoising steps.
- **Hidden condition.** An end-effector goal offset `z`, constant across all
  attempts of one trial, applied over the pilot's 100-control approach window
  and ending at the first decision boundary after grasp. Window flags update
  at five-control boundaries so teacher labels depend only on the current
  decision state. This is at most four controls later than the old pilot;
  qualification checks this explicit timing migration.
- **Training conditions.** Lateral and depth axes separately, magnitudes 2 cm
  and 4 cm, both signs: eight conditions. Qualify the depth axis in §2 before
  using it.
- **Held-out test conditions** (never trained):
  - 3 cm on either axis (interpolation).
  - 6 cm on either axis (extrapolation).
  - Diagonal lateral + depth offsets of 4 cm norm (a held-out direction).
- **No privileged trigger for the student.** The adapter acts at every chunk
  and must learn from its normal inputs when to correct. Simulator handle
  geometry may define the teacher's correction window. It never enters the
  student or the evaluation policy.
- **Attempt 1** is the same uncorrected frozen SFT policy for every arm. Before
  each later attempt, reset the scene and keep the hidden condition. The
  learner receives its recorded earlier attempts.
- **Student inputs.**
  - Current: cameras, public robot state and the original instruction.
  - History: earlier public observations, actually executed controls, native
    rewards and episode boundaries.
  - Never: the hidden offset, simulator geometry, teacher identity, current
    target or future outcome. Public motion may reveal the offset, and that is
    the intended channel.
- **Analytic compensation** is allowed only in the named privileged teacher.

This is **same-scene retry**. Changing the scene between support and query is a
later transfer test.

## 2. Qualify the interface and collect data

Reuse [the KV bridge](../../src/fluffy_bassoon/meta_rl/flow_history.py) and
[its frozen-expert training path](../../src/fluffy_bassoon/meta_rl/flow_bridge_train.py),
with a new collector and trial sampler for this run. The compact pilot
archives lack camera inputs. They are development evidence, not a visual
training corpus.

Engineering checks before bulk collection, on development scenes:

1. **Runtime parity.**
   - Actual allocated GPU and a native episode.
   - Zero-initialized adapter parity and deterministic reset pairing.
   - One accelerator and runtime family across collection, fitting and
     evaluation.
2. **Teacher correctness and authority, per axis.**
   - The teacher compensates the known offset in native control space,
     respecting controller gain, clipping and the active window.
   - Show real corrected rollouts for every training and held-out condition,
     and quantify clipping.
   - Fit one bridge that is given the offset as a privileged input. Confirm it
     beats the best fixed correction on development resets.
   - Failure can reflect labels, optimization or limited correction authority.
     Diagnose these separately; fix `scale` from measured command authority,
     not held-out success. Fit this privileged bridge on training conditions.
   - Qualify condition feasibility on development scenes only, then freeze
     the condition set before confirmatory collection. Never drop individual
     held-out failures based on teacher or learner outcomes.
3. **Executed-action semantics.**
   - Save the five executed controls and their mask separately from the
     proposed 50-control chunk.
   - Supervise only validated teacher command coordinates and timesteps, with
     native action targets outside the correction window.
   - Extend the bridge loss with the needed mask. Resolve window changes
     within a chunk in the label contract.
   - Preserve native 32-dimensional normalization, 12-dimensional physical
     controls and binary action mapping.
4. **Learning path.**
   - Finite gradients reach the history model and KV adapter through the
     frozen expert; motor tensors are unchanged.
   - A small development fit reproduces distinct teacher corrections in
     decoded commands and native behavior.

These are validity checks, not success-percentage admission rules. Run the
applicable core tests on allocated Linux.

### Splits and collection

**Scenes:** 32 training, 8 development and 32 fresh held-out. A scene and all
of its attempts and conditions belong to exactly one split. Old pilot scenes
are development-only.

**Screening:**
- Publish ordered, disjoint candidate seed lists with hashes before
  collection.
- Screen at most 128 training, 64 development and 128 test candidates.
- Eligibility is nominal SFT success and a valid approach. Never select on
  perturbed, teacher or adapter outcomes.
- Keep all screening counts and failed trials. Report the result as
  conditional on the nominally solvable subset.

**Per training or development scene,** draw two training conditions without
replacement, balanced across the eight. For each, collect from the same reset:

1. An uncorrected attempt.
2. A partial-correction attempt: the teacher at a random 0–100% of the needed
   compensation.
3. A full teacher query attempt.

At every query state in attempts 2 and 3, also obtain the full teacher label
from that exact observation, independently of the partial behavior action.
Attempt-2 targets use only attempt 1 and the causal attempt-2 prefix; attempt-3
targets use attempts 1–2 and its causal prefix. No third-attempt experience
enters an attempt-2 input.

Training contexts hold one or two earlier attempts, so the student sees both
first-attempt and second-attempt histories. That gives up to **240 episodes**
across 40 scenes, plus nominal screening.

Preserve query images, public states, teacher labels, executed-control masks,
terminal feedback and reset and noise identities. Failed teacher episodes stay
in. There are no queries after success.

## 3. Train the adapter and a matched control

| Setting | Planned value |
| --- | --- |
| Interface | History decoder to learned KV residuals over the frozen VLA prefix; no correction selector, no velocity bypass, no trigger input |
| Decoder | Width 256, 6 layers, 8 heads |
| KV adapter | Rank 4, `scale` fixed in §2, all qualified prefix layers, zero output initialization |
| Context | Up to three episodes of at most 120 decisions, plus boundaries; verify terminal feedback is retained |
| Training | DPT teacher-action flow matching, logical batch 8, AdamW learning rate 0.0003, weight decay 0 |
| Fits | Full-history DPT and current-only DPT, seeds 0, 1, 2: six fits |
| Endpoint | Fixed 4,096 updates; development curves at 1,024/2,048/4,096; final checkpoint used for evaluation |

**The current-only control** has the same architecture, labels, query draws,
initialization seeds and optimizer budget, but no earlier
observations, actions or rewards. Preserve complete earlier episodes in every
full-history window.

Sample query positions without replacement within scene and condition strata.
Record coverage and multiplicities. Normalization uses training data only.
Measure execution batch capacity without changing the logical batch. If
development shows underfitting, report it and amend the plan before opening
held-out outcomes.

## 4. Confirmatory evaluation: fresh paired native attempts

**64 cases in 32 held-out scenes, two cases per scene.** Assign conditions in a
fixed balanced design before collection:

| Condition group | Cases |
| --- | ---: |
| In-distribution conditions (training magnitudes and axes) | 16 |
| Held-out 3 cm (interpolation) | 16 |
| Held-out 6 cm (extrapolation) | 16 |
| Held-out diagonal direction | 16 |

**Each case runs three attempts.**
- Attempt 1 is shared across arms.
- Attempts 2 and 3 start from the same verified scene state, with matched
  inference noise.
- Each policy runs closed loop, and its own attempt 2 feeds its attempt-3
  history.
- No recorded outcome lookup scores a learned policy.

| Arm | Attempts | Purpose |
| --- | --- | --- |
| Full DPT, three fit seeds | 2 and 3 | Primary: does experience turn into successful action? |
| Same DPT, earlier attempts cleared, three seeds | 2 and 3 | Causal effect of history; normal within-episode memory kept |
| Current-only DPT, three seeds | 2 and 3 | History versus equal supervision and capacity |
| Full DPT, random-donor history, three seeds | 2 | Same task and condition group, random condition: does the *particular* experience matter? |
| Frozen SFT | 2 and 3 | Native retry |
| Archived OpenDrawer PPO30 | 2 | Measured zero-shot policy-stage reference; no new training |
| Fixed correction selected on training/development data | 2 | Constant-correction baseline; no test-condition axis input or test selection |
| Privileged teacher | 2 | Upper bound and solvability |

On a predeclared 16-scene subset, add opposite-condition support for each DPT
fit (96 episodes). This deliberately contradictory history should reverse the
commanded correction. Reward masking alone is not a useful control here,
because motion also carries the information.

The compact reader is deferred. PPO30 is retained: lack of offset exposure
does not imply that it behaves like SFT. The online PPO comparison below asks
a separate adaptation-cost question.

**Primary outcomes.** Paired success at attempt 2:
- full DPT minus its history-cleared version;
- full DPT minus current-only DPT.

Report both for in-distribution and held-out groups separately.

**Secondary outcomes.**
- The attempt 1–3 success curve with history kept versus cleared.
- Cumulative success.
- The random-donor contrast.
- Signed commanded displacement at the first grasp, compared with the true
  offset.
- Physical controls consumed and clipping.

**Uncertainty.**
- Average fit seeds within scene and report each seed.
- Resample whole scenes. Conditions, repeated fits and shared support are not
  independent samples.
- Report discordances and intervals even when inconclusive.
- Sixteen cases per held-out group gives coarse per-group estimates. The
  pooled held-out contrast is the main generalization result.

**Episode count.** 64 shared supports + 1,344 DPT queries + 320 reference
episodes = **1,728 episodes**, plus 96 optional opposite-history episodes.
Profile throughput first. If the initial allocation cannot fit, defer optional
controls and attempt-3 endpoints before reducing independent scenes. Any
change is fixed from timing before inspecting confirmatory outcomes.

## 5. Sample efficiency against online RL (conditional)

Run this only if §4 shows a history benefit. Start online PPO from Human300
SFT, using the four-epoch recipe, under one fixed in-distribution condition on
the training scenes. Evaluate after 2, 5 and 10 updates, at 128 episodes per
update, on the same held-out cases for that condition.

Report deployment adaptation episodes to reach the adapter's attempt-2
success, with the same observation/reward access. Separately report all offline
student data, privileged teacher labels and training compute; one support
episode does not mean one total training episode. Use a dedicated sufficiently
sized fixed paired panel for this condition, not the two cases per condition
in the small ID panel. PPO fits a separate condition-specific policy; the
adapter has fixed weights across conditions.

## 6. Budget, storage and delivery

**Campaign hard cap: 200 GPU-hours, explicitly approved by the user.**
Initially reserve 48 for §§2–4, 10 for conditional §5, 90 for contingent
task/objective extension and 52 uncommitted for justified profiling, recovery
or precision work. Reassign these reservations from evidence while keeping
all work within 200. Record cumulative usage separately from closed campaigns. The
stage amounts below are planning reservations, not guaranteed runtimes:

| Stage | GPU-hours |
| --- | ---: |
| Qualification, including the privileged-input bridge | 4 |
| Collection and labels | 7 |
| Fits | 18 |
| Evaluation | 16 |
| Recovery reserve | 3 |

Stop at the 200 GPU-hour hard cap with resumable artifacts. No automatic
extension beyond the user-approved cap.

**Compute.** Baige is the target. At launch, inspect live queue capacity,
limits and the actual accelerator type. Prefer up to four independent one-GPU
workers on qualified RTX 6000-class allocations, using the in-job GPU probe
and binding inference and rendering to the allocation. Review the latest main,
commit and validate the implementation, then deploy commit-pinned worktrees
through Git.

**Storage.**
- Check NVMe capacity, quota and a write probe first.
- Save lossless policy-input images at decision points only, and profile bytes
  per episode before bulk collection.
- Keep the initial, final and evaluation checkpoints, plus two complete
  recovery boundaries with optimizer, sampler and RNG state.
- Archive completed raw shards before removing hot copies. Small manifests and
  results go into Git; large payloads do not.

**Timeline target:** about four days once resources are allocated:

| Day | Work |
| --- | --- |
| 1 | Qualification and collection |
| 2 | Fits |
| 3 | Paired evaluation and result note |
| 4 | §5 if it runs |

Monitor every 20 minutes after launch on the coordination board.

## Decision and path to the paper

| Observation | Next decision |
| --- | --- |
| Teacher or privileged-input bridge fails | Fix the interface or runtime; this does not test the history hypothesis |
| Bridge fits teacher actions but shows no held-out history benefit | Examine closed-loop shift and history use; do not scale this recipe |
| Benefit in distribution only | Report it; held-out conditions become the next method question |
| Benefit in held-out groups over both controls, across fits | Run §5, then extend to two or three more RoboCasa tasks |
| Intervals admit both benefit and none | Inconclusive; add paired scenes rather than a teacher campaign |

**Extension after a positive result.** Run the same condition family on two
or three more RoboCasa tasks that the nominal SFT policy can solve, with one
held-out task. Then run the objective ablations (AD, DIT, IC-IQL) on the
shared corpus, with equal architecture and budgets. That extension and its
budget, roughly 60–90 GPU-hours, need their own short plan and must remain
within the approved 200 GPU-hour total. Launch only if the preceding evidence
supports the extension.
