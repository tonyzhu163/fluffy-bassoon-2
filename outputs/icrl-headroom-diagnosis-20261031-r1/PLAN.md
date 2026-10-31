# Why no adaptation result yet, and a headroom-first redesign

Written 2026-10-31 Asia/Shanghai from saved results on `main` (`f7dde2c`) and
on the unmerged branches linked below. No job was queried or launched. This is
a proposal; GPU stages need the user's approval of scope and budget.

## Diagnosis

Every robot comparison so far reports no memory, adaptation or
sample-efficiency advantage. The evidence says the setups could not have shown
one, independent of the CTM or memory architecture. Five causes, in order of
how much they constrain the result:

### 1. The motor decides almost every outcome

- LIBERO fixed100: 81 of 90 tasks give the same outcome under native, AD
  full/static and DPT full/static (57 fail under all, 24 succeed under all).
  Totals 28/29/29/28/28. On the K1 panel the support outcome equals the
  query outcome in nearly every case; only tasks 0, 40 and 80 fail support,
  and every method fails them again.
  [Result](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/codex/libero-fixed100-closure-20261031/outputs/libero40-fixed100-target-plan-20261031-r1/RESULT.md),
  [per-case CSV](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/codex/libero-fixed100-closure-20261031/outputs/libero40-fixed100-target-plan-20261031-r1/case-results.csv).
- Privileged knowledge barely helps. The tracker gate's external oracle
  scores 4/6 against zero 3/6, and 1/2 at −45°
  ([tracker](../pi05-tracker-gate-baige-20261028-r2/RESULTS.md),
  [support/probe](../pi05-support-probe-baige-20261028-r1/AUDIT-20261028-2148.md)).
  In Meta-World reach no noise arm or native sample ever reaches goals 1
  or 3; the best fixed arm adds about 6 points
  ([partial results](../metaworld-pi05-noise-baige-20261029-r2/PARTIAL-RESULTS.md)).
  In ManiSkill S0, fixed arms are at ceiling (59/64), and the first chunk
  differs between arms (0.052) about as much as between native repeats
  (0.039) ([S0 results](../noise-mode-icrl-20261029-s0-r1/RESULTS.md)).

Steering through noise or a bounded residual re-selects behaviour the motor
already has. Where the motor lacks a skill there is nothing to select, and
where it has it, the native sample usually finds it. The ceiling for any
history-conditioned learner is the oracle, and the oracle sits about one case
above native.

### 2. The training histories contain no learning to distil

The LIBERO source corpus has 94 failures in 2,108 episodes; 454 of 527 K4
trials are four successes, and 10 show a failure followed by a success
([evaluation](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/codex/libero-fixed100-closure-20261031/docs/frozen-vla-libero/EVALUATION.md#2026-10-30-broad-controls-and-baseline-readout)).
Algorithm distillation imitates the improvement present in its histories.
These histories show a competent SAC repeating success, so the target is
competence, not recovery.

### 3. The learners ignore reward

- With the same recorded observations and actions, erasing support moves the
  LIBERO controller's first-query mean by L2 1.289; masking rewards moves it
  by 0.066 (same evaluation document, offline input-sensitivity probe).
- RoboTwin bottle: reward-blind 8/8, normal 7/8, cache-reset 6/8
  ([final campaign](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/codex/robotwin-native-wholetrials/outputs/robotwin-trained-continuation-20261031-r1/FINAL-CAMPAIGN-RESULTS.md)).
- Native CTM/XL PPO normal/wiped/blind: CTM 75.0/50.0/62.5%, XL
  50.0/62.5/75.0% ([final audit](../pi05-live-bridge-poc-baige-20261028-r1/FINAL-AUDIT-20261028.md)).

Where history matters, the controllers use it to identify the task from
observations. That is context-conditioned imitation, not in-context RL.

### 4. The noise floor equals the effect size

- With bitwise-identical inputs, cross-process decoding flips 4 of 90 native
  LIBERO outcomes; fixed100 has 9 discordant tasks across methods.
- ManiSkill S0: 4 of 32 repeated fixed-arm trajectories are identical.
- Sample sizes: one initial state per LIBERO task; RoboTwin cells are one K8
  trial each, effectively n = 1; probe and tracker gates have 6 lanes.

### 5. The learners are weak even on their source tasks

RoboTwin PPO reaches 2–3 of 24 source-validation cases. LIBERO AD's
validation loss rose while training loss fell on 742 chunks from one goal;
DPT validation NLL rose from fit update 25 to 100.

### Separate: the CMM port is unstable

Calibration ended at 4.55% query accuracy against the paper's 30.63%.
Clipped updates grew to 72% before the collapse, amplification runs mainly
through the short-term recurrence, and least squares on the same data scores
32% ([collapse report](../cmm-collapse-debug-20261028-r1/REPORT.md)). Changing
the decay start did not help. No CMM comparison is interpretable until this
is fixed.

## Steering interface

Noise is a reasonable way to improve the motor, as DSRL does, but the
evidence above makes it a poor set of options for in-context RL:

- It cannot add skills the motor lacks. Meta-World arms differ by first-chunk
  L2 0.43 against 0.03 between repeats, yet none reaches goals 1 or 3.
- Where the motor is competent it adds little. S0 arms differ about as much
  as native resampling; the LIBERO all-90 noise controls score native 29,
  tied Gaussian 28, zero 25.
- One noise draw does not mean the same behaviour across scenes. Hidden-goal
  rank consistency is 0.70, over only two informative goals. A learner that
  generalises evidence across attempts needs options whose effect transfers.
- Learned per-chunk noise was also weak in the bounded LIBERO DSRL run:
  72/240 training successes on task 0, 0/240 on task 40, both final queries
  failed. The fixed-noise screens did not test this per-chunk form.
- Noise steering is DSRL's ground, and VLRLA already covers recurrent
  adaptation. Choosing among meaningful options in context is a clearer
  contribution.

Proposal: make semantic options (instruction or target object, subgoal,
skill primitive) the primary interface, with an oracle equal to the correct
option, random at about 1/k and a bandit baseline. Keep noise as the DSRL
baseline and, optionally, as refinement within a chosen option. Decide by
running H1 on both interfaces, each with its own oracle, on the same panel;
the one that clears the oracle-gap and retry gates is used for H2–H4.

## Which cells of the design matrix to test

The [three-axis design](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/codex/frozen-vla-poc/docs/frozen-vla-icrl-design.md)
crosses training (RL², AD, DPT), architecture (GRU, Transformer-XL, CTM) and
interface (noise, residual, features/KV; semantic options added above). Do not
run the full factorial. Every cell needs a task whose history carries
information that reward reveals, and an interface with oracle headroom;
neither axis supplies these. On the current tasks all cells would be null and
would say nothing about the axes.

| Axis | What to test | Why |
| --- | --- | --- |
| Interface | Every option, oracle only, no learner (H1) | Cheap, and it is where the current nulls come from. Semantic options: the correct option. Noise: per-condition DSRL, already a required baseline. Residual: the privileged tracker, already about one case of headroom. Features/KV: no cheap oracle; defer |
| Training | RL² and DPT first; AD after an H2 corpus exists | RL² needs no teacher data. DPT needs per-condition expert actions, which the H1 oracle supplies. AD needs histories that improve, which the current corpus lacks |
| Architecture | Last (H4) | It matters only once some cell adapts |

From the first working cell, change one axis at a time: training method,
then architecture. This is about four interface screens and five learner
cells instead of 27 learner cells.

## Redesign

Order the work so that each stage can fail cheaply before the next one
spends budget. A stage's gate is fixed before its data are seen.

### H0. Measure detectable effect on existing panels (CPU, no new actions)

For each completed panel (LIBERO fixed100 and K1, RoboTwin final100,
Meta-World reach, S0, tracker), compute from saved outcomes:

- informative cases: native or support fails and some arm, oracle or other
  method succeeds;
- the oracle-minus-native gap over those cases;
- the repeat flip rate where repeats exist;
- the smallest paired effect detectable at α = 0.05, power 0.8.

Output: one table that states, per panel, the largest effect it could have
detected. This turns "no advantage" into "no advantage larger than X", and
shows which panels are uninformative.

### H1. Find a condition with headroom (small GPU screen)

Pass all four before any learner training:

| Gate | Requirement |
| --- | --- |
| Oracle gap | A policy given the hidden variable beats native by ≥ 25 points, over ≥ 20 initial states per task |
| Retry solvability | After a failed attempt, some available behaviour succeeds on the same case in ≥ 50% of failed cases |
| Hidden from observation | The variable is not recoverable from the first observation (otherwise this is observation-conditioned DSRL) but is identifiable from one or two outcomes |
| Repeatability | Three repeats of identical runs flip ≤ 5% of outcomes, or the flip rate is measured and the panel sized for it |

Screen every cheaply oracled interface on the same panel: semantic options,
learned per-chunk noise (per-condition DSRL) and the residual (see
[steering interface](#steering-interface) and
[which cells to test](#which-cells-of-the-design-matrix-to-test)).
Candidate tasks, cheapest first:

1. **Prompt arms with a hidden target.** The motor follows instructions
   for some objects: ManiSkill explicit-minus-vague gaps of +17, +31 and +10
   points for carrot, banana and hamburger (toy bear −2)
   ([instruction swap](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/instruction-swap-maniskill/outputs/pi05-instruction-swap-thu2-20261029-r1/RESULTS.md)).
   Hide which of k reachable objects is rewarded; the arms are instructions.
   The oracle is the correct instruction, random scores about 1/k, and
   reward identifies the target after a few attempts.
2. **Meta-World hidden goal restricted to reachable goals.** Goals 0 and 2
   are reachable now. Add goals only after the planned goal-hidden motor
   fine-tune reaches them.
3. **LIBERO multi-target scenes** with the target object withheld from the
   instruction, using the same arm logic as candidate 1.

Size the panel with H0's numbers. For a paired binary outcome with about 20%
discordant pairs, detecting a 15-point difference needs about 70 paired
cases.

### H2. Build histories that contain learning (CPU first)

Collect source trials from an exploring teacher over the H1 arms (Thompson
sampling or UCB, plus a from-scratch learner), not from a converged policy.
Gate: at least 30% of K-attempt trials contain a failure followed by a
success, and attempt-k success exceeds attempt-1 success across the corpus.
Report this statistic for every corpus before fitting AD or DPT.

### H3. Test in-context RL with controls built in

Start with one cheap learner on the interface that passed H1: GRU trained by
RL², plus GRU trained by DPT on H1 oracle labels. Add AD on the H2 corpus
only if one of these adapts.

Primary metric: paired within-trial gain (last attempt minus first). Every
run includes:

- reward-shuffle and reward-blind (the RL test; section 3 shows these are
  where current learners fail);
- memory reset and wrong-task history;
- the oracle (ceiling), random arm choice and the H2 bandit teacher at equal
  attempt budget (an in-context learner must beat the bandit to matter);
- static multitask DSRL as the observation-only reference.

Gate for any adaptation claim: normal beats reward-shuffle and memory reset
on the paired metric at the H1 panel size.

### H4. Architecture comparison

Only after H3 passes: hold the working training method and interface fixed
and compare CTM against Transformer-XL and GRU at matched parameters and
interaction budget, on the same H1 panel and controls. Features/KV steering
is considered only if no cheaper interface passes H1.

### CMM track (runs in parallel; needs approval)

Before any new full run: replay the epoch-450 checkpoint on fixed batches
with lower learning rate, warmup and STM-path gradient normalization; keep
milestone checkpoints. Gate: stable ≥ 25% query accuracy through the full
budget. Until then, CMM stays out of H3 and H4.

## What this changes in current work (for the user to decide)

- Pause further scaling of LIBERO AD/DPT on the current source corpus and
  the RoboTwin final100 export until H1 passes. Both reuse histories that
  fail section 2.
- Treat native VLRLA PPO as a motor-competence intervention, not an
  in-context RL test, unless it is evaluated under the H3 controls.
- H0 needs no approval and can start now. H1 needs a GPU budget; a first
  screen of candidate 1 at about 70 cases per arm set is sized from H0 and
  measured episode cost before asking.
