# Findings to date

Consolidated 2026-11-01. Only the Meta-World ML10 comparison and the DSRL runs
are well-controlled enough to build on. The other threads are summarized here
so they are not repeated. Their run folders were deleted, except RoboTwin's
(restored under `outputs/archive/`); other links point to GitHub permalinks at
the pre-cleanup commit. Single-seed results are
descriptive, and none establishes a CTM, memory or in-context-RL advantage.

## Headline

- **No experiment so far shows in-context adaptation or a memory benefit**, on
  Meta-World, RoboMimic, LIBERO or RoboTwin. Reset and reward-blind controls
  match or beat full-history controllers wherever they were run.
- **Ordinary DSRL works when there is headroom.** Original Pi0 on task 57 goes
  4/10 → 10/10, and OpenPI40 pi0.5 on task 44 goes about 5/20 → 20/20. Both
  are single-seed. On saturated tasks (pi0.5 task 57, already 10/10) it only
  shortens episodes.
- The main blockers were **headroom and source learning**, not freezing the
  motor. Weak source learners (~11–17% success, flat curves) gave the
  meta-learners nothing useful to imitate.

## Meta-World ML10 (CTM question)

10M training transitions per run, seed 1000.
[Evaluation](../outputs/metta-common-ppo-chart-20261029-r1/ML10-EVALUATION.md),
[common-PPO bundle](../outputs/metta-common-ppo-chart-20261029-r1/README.md),
[native-method bundle](../outputs/metaworld-ml10-matched-baige-20261028-r1/README.md).

| Core (common PPO, ≤250k params) | Train success | Held-out success |
| --- | ---: | ---: |
| GRU | 20.8% | 0.8% |
| Transformer-XL | 20.0% | 8.8% |
| CTM | 16.0% | 0.0% |
| CMM | 10.0% | 22.4% (reset 26.4%) |

The CMM lead is not a memory effect: reset scores higher. The native-method
comparison (CTM, RL²-PPO, ECET, AMAGO1) shares the CTM checkpoint and also
shows no CTM advantage. CMM few-shot calibration failed (4.55% vs paper
30.63%) from optimization instability.

## RoboMimic Can

Two seeds × 8 held-out scenes × 4 attempts, 28,800 training steps per arm.
RL² scores 41/64 and frozen DSRL-NA 31/64. RL² with memory reset scores 40/64,
and an episode-reset-trained controller 39/64, so the gain is not from learning
across attempts.
[Evaluation](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/741064b4d8df00555f92a7a8449055c0edb50707/docs/robomimic-can-dsrl/EVALUATION.md).

## RoboCasa365

The original Human30075000 pi0.5 completed the predeclared OpenDrawer headroom
bank with **3/8 terminal successes**, full native750-step episodes. The first
episode verified all15 chunk seeds against exact zero-residual action replay
before executing each chunk once. The task was frozen and the fallback was
not run. On exactly those eight native videos, official Flora expected scores
ranked10of15 success/failure pairs correctly: **AUROC0.6667**, below the
predeclared0.70 gate. Both integrated simulator- and Flora-reward residual PPO
pipelines therefore stopped cleanly before optimizer updates or shared before
panels. That original attempt produced no comparative learning or final trained checkpoint.

The user subsequently authorized continuation below the original AUROC gate.
The unchanged initializer scored **4/8 on the fixed held-out ID panel** and
**6/8 on the predeclared CloseFridge OOD panel**, before PPO; initial DEV was
7/8. These eight-case panels are descriptive and remain separate from TRAIN.
A common 64-episode residual-PPO batch now collects through eight allocated
GPU workers; the numerical checks preserve the serial draw stream and next
Adam update. The baseline has completed three optimizer updates and saved
full recovery checkpoints; its first fixed DEV evaluation was6/8, versus7/8
before training. This small panel does not establish an improvement or a
simulator-versus-Flora result. The original failed calibration remains
unchanged. [Before panels and qualification](../outputs/robocasa-eight-gpu-speed-20261102-r1/RUNTIME-STATUS.json).

The persistent official scorer loaded once and completed all eight labels;
two affected core numerical/reuse/cleanup cases passed on allocated Linux.
This qualifies scorer execution, while Flora-reward training-batch/optimizer integration
remains unexercised. The eight-case result is descriptive development evidence,
not a population estimate or a conclusion that Flora is generally unsuitable.
Original weights, complete native videos, raw probabilities and costs are
retained. No task or score was adjusted after observing this failed gate.
[Native result](../outputs/robocasa-native-residual-headroom-q4-20261101-r1/terminal/summary.json),
[scorer gate and terminal audit](../outputs/robocasa-residual-integrated-20261101-r3/TERMINAL-REVIEW.json).

## LIBERO

**Flora versus simulator PPO stopped before training (2026-11-01).** The
published few-shot PyTorch LIBERO40 SFT checkpoint succeeded on all64 baseline
episodes on task44, covering40 nominal TRAIN placements with24 second draws.
Each episode ran400 steps plus15 settling actions, with no training updates.
This exceeds the predeclared70% native-success limit, so neither PPO arm is
admitted. AUROC is undefined because the bank has no simulator failures. The
already-running Flora smoke-video qualification remains separate; no comparative
learning result exists. Historical C565 headroom does not establish headroom
for this different initializer.
[Result and limits](../outputs/flora-ppo-comparison-20261101-r1/RESULT.md).

**Frozen-motor DSRL.**

- *Original Pi0, task 57* (Baige `job-psgxs9n5b08w`): native 4/10 → learned
  10/10 at 500k updates. 10/10 is first reached at 40k updates (~50k training
  actions). This qualitatively reproduces the DSRL paper; its three-seed
  claims are not verified.
  [Evaluation](../outputs/dsrl-pi0-task57-20261101-r1/EVALUATION.md).
- *OpenPI40 pi0.5, task 57*: native and learned are both 10/10 (ceiling).
  Learned uses 14% fewer primitives; one regression panel at 390k. 28.7k
  GPU-s, 333k actions.
  [Result](../outputs/dsrl-pi05-task57-20261031-r1/RESULT.md).
- *OpenPI40 pi0.5, task 44 ("turn on the stove")*: screening 5/20 and
  confirmation 5/20. Fresh controls: native 6/20, untrained tied noise 13/20,
  initialized SAC 8/20. Learned panels: 1/20 at 20.6k updates, then 20/20 at
  every panel from 40k through 400k. Paused by the human at 406,780 updates,
  with full recovery, for the Flora comparison. The fixed 500k endpoint and the
  final native panel were never run.
  [Pause record](../outputs/owned-jobs-pause-20261101-r1/RESULT.md).

**Headroom screen** (384 episodes, 12 LIBERO90 candidates, states 1–8 × 2
draws): selected task 59 (native 3/16, tied 9/16), runner-up task 64. Task 44
was native 1/16, tied 10/16. Untrained tied noise often beats native sampling,
so any learned gain must be compared against it.
[Results](../outputs/libero90-pi05-headroom-screen-20261101-r1/RESULTS.md).

**Source40 → target90 meta-learning** (AD/DPT history controllers trained on
LIBERO40 source histories):

- Fixed100 phase: native 28/90; AD full/static 29/90 each; DPT full/static
  28/90 each. At K1, AD full, reset and native all score 6/9 on the same cases,
  with AD spending 4,014 actions vs native's 1,961. No memory, topline or
  sample-efficiency gain.
  [Result](../outputs/libero40-fixed100-target-plan-20261031-r1/RESULT.md).
- First full90 comparison: supervised and direct-PPO controllers fall below
  contemporaneous native. PPO source validation succeeds as execution but shows
  little adaptation headroom.
- All-40 existing-teacher assessment (319 valid source episodes): native
  78/80, final TRAIN SAC 78/80 in both mean and sample modes, tied warmup 70/79.
  The trained source actors add nothing over the native motor, so they cannot
  serve as teachers.
  [Run](../outputs/libero40-teacher-assessment-20261101-r1/).

## RoboTwin (Motus / VLRLA-style)

- FULL and trained-RESET learners each used ~1M source actions. Source success
  stays near 17% with no upward trend; source SAC reaches 277/2400 (11.5%).
  Two-goal targets: FULL 8/16 and RESET 8/16; cache-reset 7/16 and reward-blind
  9/16; saved native 6/16. No consistent memory or reward-input benefit.
- Steering-interface redesign: the physical teacher bank stays at 11/12, and
  fitted command/KV heads score 0/5 where native scores 0/5. No physical
  improvement; no teacher or student admitted.
  [Evaluation](robotwin/coefficient-steering/EVALUATION.md).

## ManiSkill / native pi0.5 bridge

Native pi0.5 CTM/XL PPO, the support/probe gate and the privileged tracker
completed and were audited on 2026-10-28. Cached CTM/XL/GTrXL screens and DPT
warm starts ran with one seed. None shows ICRL or a CTM advantage.
[Snapshot](../outputs/archive/coordination/coordination-20261031.md).

## Lessons carried forward

1. Establish headroom first: native success that is neither 0 nor ceiling.
   Compare against untrained tied noise as well as native sampling.
2. Get a positive control before testing a new interface or reward. A failure
   is uninterpretable without one; the Flora test uses simulator-reward PPO as
   arm S for this reason.
3. Source learners must actually improve before their histories can teach a
   meta-learner.
4. Report reset and reward-blind controls with every memory claim.