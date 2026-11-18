# Plan: does the best execution cadence depend on hidden conditions?

Exploratory. Planned November 18; nothing collected or launched. This is a
scientific-scope expansion (a cadence interface beside the action adapter), so
it needs the user's go-ahead before any GPU use.

## Question

"Learning when to replan" is worth pursuing only if the best execution cadence
of the frozen π0.5 motor changes with conditions the current observation does
not show. This run measures that directly, with no learner. It answers three
things:

1. **Oracle gap.** How much native success does choosing the cadence per hidden
   condition gain over the single best fixed cadence?
2. **Observability.** Does a selector on the current observation already close
   that gap? If so, history is unnecessary.
3. **Bandit ceiling.** How close does a simple cross-attempt bandit over
   cadences get? A history learner must beat this to be a contribution.

## Setup

Reuse the calibration-pilot harness
([collect_panel.py](../robocasa-feedback-advantage-20261117-r1/collect_panel.py)):
public Human300 π0.5 (`pi05_pretrain_human300/multitask_learning/75000`,
revision `c484448a`), predict 50, 10 denoising steps, H = 600 controls, native
RoboCasa `OpenDrawer`, `pretrain` split, native success predicate,
`matched_start` pairing so every branch of a scene shares an identical start.

**Cadence K** (controls executed per inference): K ∈ {2, 5, 10, 25, 50}. The
current loop hard-codes 5 (`t % 5`, `noise[t // 5]`). Generalise it to a replan
counter, and draw noise per replan from one per-scene stream, so a branch's
noise depends only on scene and replan index.

**Hidden conditions** (none visible in the first observation):

| Condition | Change |
| --- | --- |
| `nominal` | none |
| `lat5`, `lat15` | inference delay d = 5 or 15 controls (0.25 s / 0.75 s at 20 Hz) |
| `off+`, `off-` | the pilot's persistent ±4 cm lateral approach offset, uncorrected |
| `heavy` | drawer slide joint damping and friction loss scaled up (factor fixed at Gate 0) |

Delay uses naive asynchronous execution in a synchronous simulator: at a
replan at t, the policy sees obs_t; the previous chunk keeps running for the d
controls of inference; the new chunk starts at its index d. The first chunk
holds a zero-delta command for d controls. A chunk must cover K + d steps, so
under delay K = 50 becomes K = 50 − d; record the executed K. This matches the
naive baseline of real-time-chunking work, not a tuned asynchronous method.

## Stages

**Gate 0** (2 scenes, seeds `982101`, `982103`; all 6 conditions × K ∈ {5, 50}).
Check that tape replay of `nominal` is bit-identical, delay branches match the
nominal prefix up to the first replan, and the first observation hash is equal
across conditions. Set the `heavy` factor as the smallest tried (2, 4, 8) at
which nominal K = 5 still opens the drawer on at least one Gate-0 scene but
takes visibly longer; record what was tried. Stop and report if no factor
behaves sensibly, or if delay breaks determinism.

**Panel** (16 scenes, admitted by the pilot's rule from nominal K = 5
behaviour only; first 8 for fitting the selectors, next 8 held out). Every
scene runs all 6 conditions × 5 cadences × 2 noise seeds = 60 episodes,
960 in total. Store each episode's outcome, controls to success, number of
inferences, end-effector tracking error, and the state at the first replan
after control 50.

## Analysis (offline over the exhaustive outcome table)

Selectors fitted on the 8 training scenes, scored on the 8 held-out scenes:

- `oracle_per_condition`: best K for each condition, chosen on train.
- `best_fixed`: one K for all conditions.
- `current_obs_ref`: ridge classifier predicting the best K from the first
  observation (expected near `best_fixed`, since conditions are hidden there).
- `early_obs_ref`: the same from state at control 50, which tests whether the
  condition is identifiable within one episode without earlier attempts.
- `bandit_ucb`, `bandit_thompson`: over K across 4 simulated attempts per
  trial, drawing outcomes from the measured table, with repeated-noise draws as
  attempt variation.

Report per-condition success-by-K curves, the oracle-minus-best-fixed gap
and each selector's gap with 10,000-draw scene-stratified paired bootstrap
intervals, and the success/inference-count Pareto front per condition.

## Decision

- **Continue to a cadence learner** if the best K differs across conditions,
  the held-out oracle-minus-best-fixed interval excludes zero, and
  `current_obs_ref` leaves most of that gap. Whether `early_obs_ref` closes it
  decides the framing: within-episode adaptation vs. history across attempts.
- **Stop** if one fixed K is near-best everywhere: there is nothing for
  in-context learning to find on this task. Record the result and close.
- If the bandit reaches the oracle within 2–3 attempts, the learner must
  instead target state-dependent cadence within an episode. Write that up
  before any training.

One task and six handpicked conditions cannot establish generality. This is
a go/no-go for the idea, not evidence for a paper.

## Resources

Baige Pro6K, 4 GPUs as in the pilot, with `scripts/[train]gpu_probe.py
--expected-count 4` in the job step. The pilot ran about 0.011 worker
GPU-hours per K = 5 episode. Inference count scales roughly as 5/K, which puts
the panel near 10–12 worker GPU-hours plus Gate 0 and reservation overhead.
Phase ceiling: **20 GPU-hours**. Pin the latest `main` commit at launch and
record it in `EXECUTION.md`. Results go in `RESULT.md` here; promote to
`docs/FINDINGS.md` only if they decide the idea.
