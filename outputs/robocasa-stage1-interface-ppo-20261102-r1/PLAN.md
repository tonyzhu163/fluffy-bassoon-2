# Stage 1: velocity and all-layer KV steering PPO (simulator reward)

2026-11-02. Status: **approved by the human: two zero-init arms, 20 GPU-h
cap. Interface code on `main` (`1c4c7039`), 34/34 thu2 CPU tests (job 50005,
two envs; [log](CPU-TESTS-thu2-50005.txt)). Not launched; this session has
no Baige route, so a Baige owner launches from [HANDOFF.md](HANDOFF.md).**
Supersedes the residual Stage 1 in the
[improvement gate](../robocasa-ppo-improvement-gate-20261102-r1/PLAN.md#stage-1-simulator-reward-positive-control-single-task);
its pass test, logging and stop rule carry over unchanged.

## Human decisions (2026-11-02)

- Interfaces: **velocity residual** and **KV conditioning**, two separate
  arms. The action residual is retired for Stage 1. A combined arm was not
  screened and is not in scope.
- **KV conditioning steers every layer of the prefix KV the action expert
  reads**, not only the final layer as in Stage 0. Expert weights stay
  frozen. Velocity already acts at every Euler step and is unchanged.
- **No warm start.** Both arms start at native identity (actor output layer
  zero, so the mean code is 0 and the deterministic policy is native π₀.₅).
  The warm start was dropped after the Stage 0 codes showed no success
  signal (below).
- Reward: first success (+1 at the first success chunk, γ=0.99, λ=0.95,
  terminate on success).

## Arms (OpenDrawer, simulator reward)

| Arm | `InterfaceSpec` | Initialization |
| --- | --- | --- |
| V0 | `kind="velocity", steering_dim=32` | zero (native identity) |
| K0 | `kind="conditioning_all", steering_dim=32` | zero (native identity) |

Learner: the existing `InterfaceLearner`; the policy emits a per-chunk,
observation-conditioned 32-D code plus native iid motor noise as a scored
action (std 1, native law). Code std starts at 0.5, the Stage 0 law.

## All-layer KV interface

`conditioning_all` adds the same kind of fixed ±1/√32 projection value
residual to every prefix KV layer, scaled by that layer's RMS and one shared
fraction. Layer seeds count from the top, so the final layer keeps the
Stage 0 map exactly; zero code is bitwise native. The fraction is calibrated
on GPU by the Stage 0 rule (action-space RMS matched to the action-residual
0.25 target) before the first update and frozen. The all-layer magnitude
was not screened in Stage 0.

## Why no warm start

Stage 0 drew an independent code ~N(0, 0.5²) per chunk (12 per episode),
observation-independent. Codes rebuilt from the Stage 0 seed rule do not
predict case-centred success (permutation p: OpenDrawer velocity .56,
conditioning .86; other tasks .09–.63). The reconstruction was not checked
against Baige `PLAN.pt`.

## Budget

Each learner: up to 30 updates × 64 episodes (≤1.15M controls), ~4 GPU-h
plus evaluation. Two learners ≈ 8 GPU-h; approved ceiling 20 GPU-h
including failures and allocation lifetime.
