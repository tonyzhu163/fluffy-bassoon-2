# Stage 1: velocity and all-layer KV steering PPO (simulator reward)

2026-11-02. Status: **Stage 1 and a 20 GPU-h cap approved by the human.
All-layer KV interface on `main` (`1c4c7039`), 34/34 thu2 CPU tests (job
50005, two envs). Warm-start design reopened (below). No GPU launch; this
session has no Baige route, so a Baige owner must launch.** Supersedes the residual Stage 1 in the
[improvement gate](../robocasa-ppo-improvement-gate-20261102-r1/PLAN.md#stage-1-simulator-reward-positive-control-single-task);
its pass test, logging and stop rule carry over unchanged.

## Human decisions (2026-11-02)

- Interfaces: **velocity residual** and **KV conditioning**, two separate
  arms. The action residual is retired for Stage 1. A combined arm was not
  screened and is not in scope.
- **KV conditioning steers every action-expert layer.** Stage 0 nudged only
  the final layer's values through a fixed random projection. Stage 1 uses
  a learned adapter that writes a value residual into each expert layer's
  KV cache. The expert weights stay frozen, and the adapter's final layer is
  zero-initialized so zero output reproduces native π₀.₅ exactly. Velocity
  is already applied at every Euler step and is unchanged in structure.
- **Warm-start from the recorded Stage 0 data, then PPO.** Each warm-started
  arm is compared against a zero-init arm of the same interface.
- Reward: first success (+1 at the first success chunk, γ=0.99, λ=0.95,
  terminate on success), per the same-day human decision.

## Arms (OpenDrawer, simulator reward)

| Arm | Interface | Initialization |
| --- | --- | --- |
| V0 | velocity, 32-D per chunk | zero (native identity) |
| VW | velocity | warm-started from Stage 0 velocity bank |
| K0 | all-layer KV adapter | zero (native identity) |
| KW | all-layer KV adapter | warm-started (see limits) |

Learner: the existing `InterfaceLearner` (`kind="velocity"` /
`"conditioning"`), native iid motor noise kept as a scored action. The
policy outputs a per-chunk, observation-conditioned code.

## Warm-start and its limits

- **Correction:** Stage 0 drew an independent code ~N(0, 0.5²) per **chunk**
  (12 per episode, `WORKLOAD.py` `plan`), observation-independent. That is
  the zero-init PPO policy's steering law at the Stage 0 fraction.
- **The planned code-mean warm start has no signal.** Codes rebuilt from the
  Stage 0 seed rule, episode-mean code against case-centred success,
  permutation p: OpenDrawer velocity .56, conditioning .86 (other tasks
  .09–.63). A reward-weighted mean would move the start in a random
  direction. Reconstruction is not yet checked against Baige `PLAN.pt`.
- Usable alternative for VW only: the velocity bank (160 episodes on selected
  failure cases) is an on-policy sample of the zero-init velocity policy, so
  it can pretrain the critic or serve as a first PPO batch after per-chunk
  features are recomputed on Baige. The KV bank used the final-layer map
  and does not match `conditioning_all`.
- Pending human choice: drop warm-start (V0, K0 only) or use the velocity
  bank as above.

## All-layer KV interface

`InterfaceSpec(kind="conditioning_all")`: the same fixed ±1/√32 projection
value residual on every prefix KV layer, scaled by that layer's RMS and one
fraction; layer seeds count from the top so the final layer keeps the Stage 0
map exactly. Zero code is bitwise native. The fraction is calibrated on GPU by
the Stage 0 rule (action-space RMS matched to the action-residual 0.25
target); the all-layer magnitude itself was not screened.

## Budget

Each learner: up to 30 updates × 64 episodes (≤1.15M controls), ~4 GPU-h
plus evaluation, per the gate plan. Four learners ≈ 16 GPU-h, cap 20 GPU-h
including failures. The earlier ~1M-action approval was for one learner;
Cap approved by the human (2026-11-02).

## Before launch

1. ~~All-layer KV adapter~~ done (`1c4c7039`).
2. Implement the warm-start fit and critic pretraining (CPU).
3. Run core tests on allocated Linux, pin a `main` commit, fresh run IDs.
