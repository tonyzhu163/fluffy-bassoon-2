# Stage 0 handoff to the Baige owner

2026-11-02. **Human-approved: launch Stage 0** of the
[improvement gate](PLAN.md). This is a no-SGD screen. It was handed to the
Baige owner (chat `01a199ef`) because the proposing session has no Baige
route.

## Question

Can a bounded action residual recover native failures more often than plain
resampling of π₀.₅'s motor noise? The answer chooses the interface and bound
for Stage 1. Background: both OpenDrawer PPO arms stayed flat on TRAIN over
six updates ([free checks](../robocasa-ppo-free-checks-20261102-r1/RESULT.md)).

## Design

- Tasks: OpenDrawer, PickPlaceDrawerToCounter, CoffeeSetupMug and
  TurnOffStove. CounterToCabinet is excluded pending its keep-or-replace
  decision.
- Cases: up to 20 per task that failed natively at the 600-control cutoff.
  Take them from the existing five-task 50-case native banks where per-case
  outcomes at 600 controls are retained. If a task has fewer than 20
  failures at 600, use all of them and record the count. Fix the case list
  before any Stage 0 outcome.
- Arms, 8 draws per case each, simulator success only (no Flora):
  - **(a) Zero residual** with 8 fresh motor-noise seeds. This is the
    control, and it also measures noise-steering and sample-and-select
    headroom.
  - **(b) The initial (untrained) residual actor**, sampled stochastically at
    bound 0.25× norm std. Use the same 8 motor seeds as (a).
  - **(c) The same actor at bound 0.5×** (the declared fallback), with the
    same motor seeds as (a).
- Use the existing `scripts/diagnostics/robocasa_residual_rollout.py`
  requests, which already take `bound`, `policy`, `deterministic`, `seeds`
  and `noise`. Use the 600-control horizon and the four-lane, spawned-reset
  runtime from current `main`, with no scorer colocated. Use the same
  batching for all arms.
- Size: at most 4 × 20 × 24 = 1,920 episodes, about 1.15M controls, about
  3.7 GPU-h at the measured 8-lane native rate. Four lanes without a scorer
  may run somewhat slower. Use the usual output, accounting and GPU-probe
  checks.

## Readout

Per task and arm: success rate, and best-of-8 recovered cases out of N.
Compare each arm with (a) per case, using an exact paired test over
recovered or not, and also in total. Report the case count used per task.

## Decision

The rules below were declared before any outcome:

- If (b) clearly beats (a), keep the residual at 0.25 and run Stage 1 on the
  task with the largest margin.
- If only (c) beats (a), run Stage 1 at bound 0.5.
- If neither beats (a) but (a) recovers many failures, the useful variation
  is in the motor noise. Take noise steering (DSRL) or sample-and-select to
  the human; the 2026-11-01 interface decision is theirs.
- If nothing recovers failures, change the task before the interface.

Stage 1 and anything else beyond this screen need separate human approval.
Write results to `outputs/<fresh stage0 run-id>/` and update the gate row in
`outputs/coordination.md`.
