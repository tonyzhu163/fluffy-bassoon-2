# After the replication: targeted steering-interface experiments

Drafted 2026-11-01 as the follow-on to [PLAN.md](PLAN.md). **Proposal only.**
It adds no grant and doesn't change the authorized RoboTwin
[targeted scope](../frozen-pi05-endtoend-20261031-r1/SCOPE.json), which already
adopts the A1/A2/A3 criteria
([proposal](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/741064b4d8df00555f92a7a8449055c0edb50707/outputs/steering-interface-criteria-20261031-r1/PROPOSAL.md),
[review](../archive/frozen-pi05-steering-screen-20261031-r1/CRITERIA-REVIEW.md)).
This document applies those criteria to LIBERO once a replication exists.

## Why the replication comes first

Every interface comparison so far has failed without a reference point. Did
the interface fail, or the task, recipe, data or runtime? Nothing has had a
known positive result to test against. The RoboTwin pilot stopped at its
teacher-executability gate before any interface was tested
([result](../frozen-pi05-endtoend-20261031-r1/EVENT-CAPACITY-TERMINAL-RESULT.md)).

A Stage B pass gives a **positive control**: one frozen motor, task, case
bank, reward and recipe where tied 32D noise plus SAC produces a measured
gain. Each later experiment keeps all of those fixed and changes **one**
interface factor. A failure then points at that factor, and DSRL noise is
rerun as the reference on the same cases.

If Stage B fails on a task that passed the screen, that is itself the next
experiment: the gap between reachable noise successes and SAC learning is a
learner/recipe question, not an interface one.

## Factors to vary, one at a time

| Factor | Levels (DSRL reference first) | Why it matters |
| --- | --- | --- |
| Where the controller acts | initial flow noise; native KV-cache residual (VLRLA-bridge-style); flow-velocity bridge ([existing](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/741064b4d8df00555f92a7a8449055c0edb50707/docs/pi05-feature-bridge.md)); decoded-action residual | Different expressiveness and different risk of leaving the motor's competent manifold |
| When it is chosen | per chunk (DSRL); **one setting held for a whole episode** | A per-episode choice is a bandit arm across attempts, the unit that in-context RL selects between |
| Dimension | 32D tied (DSRL); 4D coefficients (RoboTwin pilot); full 10×32 | Search/learning cost versus reach |
| Conditioning | observation-conditioned (DSRL); context-blind constant | Separates reactive control from what history could add |

Don't cross factors until each single-factor contrast is read.

## Three measurements per interface, cheapest first

**E1 reachability (A1).** Use the selected task's native-failure development
cases from Stage A states 1–8, never query states 40–49. Run bounded
derivative-free search over the interface, three repeats per candidate, with
the budget charged as actions. Report cases with a candidate succeeding in
≥2/3 repeats. DSRL tied noise is searched the same way as the reference row.
An interface with no reached success is dropped for this task. This
is a lower bound, not an impossibility proof.

**E2 learnability.** Use the same SAC learner on the same training/query banks
at equal simulator actions, with the replication's learning curve as the
reference. Report actions to first reach the replication's final gain, or the
gap at equal cost. Only E1 passers enter E2. Where an interface needs a different
learner (e.g. supervised KV fit), name that as a recipe change.

**E3 condition dependence (A2).** Hold the task fixed and add a declared hidden
condition that the initial observation does not reveal. Measure whether the
controller matched to each condition beats the best mismatched controller and a
single context-blind controller on paired cases. Only an interface passing E3
can show in-context RL gains, because otherwise a static controller matches any
history learner. Candidate conditions, chosen on evidence before any data:
- hidden object physics (mass/friction), which needs a simulator change and its own qualification;
- reward-defined goal choice in a scene that supports two goals;
- a LIBERO-Plus/Pro perturbation, if it is invisible in the first frame.

Only interface × condition pairs passing E1–E3 collect A3 per-condition
lineages for AD/DPT, with full-history, trained-static and episode-reset
controls on locked fresh cases.

## Suggested order

1. **Per-episode constant noise** (32D, then 4D): it reuses the qualified
   noise path and needs no new hooks, and it is the bandit form ICRL needs.
   E1 shows directly whether a fixed noise per attempt can rescue failures.
2. **4D noise coefficients**: the same parameterization as the RoboTwin pilot,
   so the two benchmarks become comparable.
3. **KV-cache residual**: the strongest published frozen-motor interface
   ([VLRLA protocol](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/741064b4d8df00555f92a7a8449055c0edb50707/docs/vlrla-protocol.md)), at the cost of new hooks.
4. **Decoded-action residual**: highest reach but leaves the motor's
   manifold, so treat as an upper reference, not the target interface.

Then E3 on the best one or two E1/E2 survivors.

## Cost envelope (to be fixed per experiment)

- E1: ≤3 failure cases × bounded candidates × 3 repeats. Order of 10–30k
  actions and ≤2 GPU-hours per interface.
- E2: the replication used 332,977 actions and 8 GPU-hours to 500k updates.
  Compare at the cost where the reference gain first appears, not the full
  500k, and fix that point after Stage B.
- E3: two conditions × per-condition controllers; size it after E1/E2.

Each is a separate grant under the standing ≤10 GPU-hour instruction, with
no borrowing between interfaces. Anything larger, a motor change or a new
benchmark needs a new decision.

## What would change this plan

- Stage A finds no eligible task: replace the replication with a
  reachability-only E1 on fixed100 0/5 tasks, and accept that there is no
  positive control on LIBERO.
- Stage B fails: diagnose the SAC/warmup recipe on that task first.
- E1 shows every interface reaches the same cases: interface choice matters
  less than learner and condition design, so prioritize E3.
