# Residual PPO improvement gate (proposal, not admitted)

2026-11-02. Status: **PROPOSED; needs human approval before any GPU launch.**
Motivation: [free checks](../robocasa-ppo-free-checks-20261102-r1/RESULT.md).
Neither OpenDrawer PPO arm improved TRAIN success over six updates (S 72→67,
F 71→71 of 128). AD needs histories that improve, so FLORA-PPO gate 2
("positive control") has not been passed yet.

## Diagnosis from the retained runs

The failed runs only show that this short run was too small. They do not
show that the residual interface can't help.

- **Very little optimization.** Each arm ran 6 updates × 64 episodes
  (384 episodes, about 5,760 chunk decisions) with 4 epochs at LR 3e-4, for 360
  Adam steps. The DSRL run that did improve here (LIBERO task44, 5/20 →
  20/20) used about 2,400 TRAIN episodes.
- **The policy barely moved.** On fixed original-policy features, the final
  residual means are 4.6–7.1% of the bound
  ([investigation](../robocasa-ppo-reward-gap-investigation-20261102-r1/RESULT.md)).
- **The signal is sparse and noisy.** There is one terminal reward per
  12-chunk episode at about 50% success, and entropy coefficient 0. Many
  held-out cases flip in both directions under motor noise alone (16–18 of 50
  discordant).
- **Exploration noise swamps the learned mean.** Pre-tanh std stays at
  0.49–0.50, its initial value, through all six updates, and entropy only
  drifts from 4.35 to 4.28. The mean moved only to about 7% of the bound.
  TRAIN success therefore measures a mostly random residual, and it cannot
  show a better mean unless std shrinks.
- **KL is not the limiter.** Per-update approximate KL is 0.007–0.016 and
  the clip fraction is 0.08–0.20. A target_kl of 0.02 would rarely fire, and
  it can only make updates smaller.
- **Headroom is unverified.** It is not known whether any residual within
  0.25× norm-std can turn a failure into a success, as opposed to the motor
  noise succeeding by chance.

**Risk to the planned joint run.** The joint five-task template is fixed at
6 iterations × 80 episodes, which is 96 episodes per task. That is less
per-task data than the flat OpenDrawer run, so it will probably be flat too.
Fix this before joint SGD.

## Stage 0: interface headroom screen (no SGD)

Question: can a bounded residual recover native failures beyond what motor
resampling alone recovers?

- Tasks: the four candidates with gates passed: OpenDrawer (58%),
  DrawerToCounter (46%), CoffeeSetupMug (34%) and TurnOffStove (14%).
  CounterToCabinet stays out pending its keep-or-replace decision.
- Cases: 20 TRAIN cases per task that failed natively at the 600-control
  cutoff.
- Per case, 8 draws each of: (a) zero residual with fresh motor noise
  (control); (b) a random residual from the initial actor at bound 0.25;
  (c) the same at bound 0.5, the declared fallback. Arms (b) and (c) use the
  same motor-noise seeds as (a).
- Readout: per-arm success rate and best-of-8 recovery per case, compared
  with (a). Use simulator success only; Flora is not needed here.
- Cost estimate: 4 tasks × 20 × 24 × 600 ≈ 1.15M controls. At about 87
  controls/s/GPU (8 lanes, no scorer), that is about 3.7 GPU-h.
- Decision:
  - If (b) beats (a), pick the task with the largest margin and bound 0.25.
  - If only (c) beats (a), use bound 0.5 (the declared fallback).
  - If neither beats (a), the residual can't steer at this scale. Escalate
    to the noise interface (DSRL), the only interface that has worked here,
    before spending on PPO.

## Stage 1: simulator-reward positive control (single task)

- Use one task from Stage 0, with the simulator terminal reward, the 600
  cutoff and the same residual architecture and bound.
- Change only the optimization budget and settings, all declared up front:
  - up to 30 updates × 64 episodes (1,920 TRAIN episodes);
  - 10 epochs × minibatch 32 (the human-chosen joint setting);
  - target_kl 0.02 kept;
  - LR 3e-4 kept for the first run.
- Log TRAIN success per update, plus a deterministic-mean success panel on
  16 fixed TRAIN cases every 5 updates, and the policy std. Run DEV every 5 updates for monitoring only;
  it never stops the run.
- **Pass:** the last three updates beat the first three by ≥ 15 points
  (192 vs 192 episodes, about 2.4 SE at 50%). The paired held-out 50-case ID
  test also favors the final policy over the original at p < 0.05 (exact
  McNemar).
- Stop early at a plateau: no gain in the 3-update moving average for 10
  updates after update 10. Save full recoveries at every update.
- Cost estimate: ≤ 1.15M training controls plus 2 × 50 evaluation episodes,
  about 4 GPU-h without the scorer.
- If it fails, retry in order with one variable at a time:
  1. Initial std 0.2, so a moved mean dominates the noise.
  2. LR 1e-3.
  3. 128 episodes per update.
  4. Residual SAC (Policy Decorator style) as an off-policy learner on the
     same interface.

## Stage 2: Flora-reward arm

Only after Stage 1 passes: run the identical configuration with Flora
reward. Use the 600-control calibration, and the new rubric only if it
qualifies separately. Apply the same pass test. If S passes and F fails, the
reward is the bottleneck, so take up the rubric and aggregation work in the
[rubric audit](../robocasa-flora-rubric-audit-20261102-r1/CANDIDATE-RUBRIC.md).
If both pass, the per-update Flora histories are the first usable AD/DPT
source.

## Stage 3: multi-task histories (separate approval)

Apply the passing per-task budget, about 1–2k episodes per task rather than
96, to the joint template or to per-task learners. Hold out tasks for
meta-test. Size and budget come from Stage 1's measured improvement rate.

## Not changed

Active runs, the AD/DPT readout jobs and existing calibrations are not
changed. Stages 0–2 each get a fresh run ID and a commit pinned from `main`.
