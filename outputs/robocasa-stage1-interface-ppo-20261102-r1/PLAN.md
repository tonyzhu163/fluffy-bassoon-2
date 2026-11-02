# Stage 1: velocity and all-layer KV steering PPO (simulator reward)

2026-11-02. Status: **Stage 1 approved by the human; scope below recorded,
code not yet written, no GPU launch.** Supersedes the residual Stage 1 in the
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

- Stage 0 drew one code ~N(0, 0.5²) per **episode**, held fixed across all
  12 chunks, independent of observation; 160 episodes/arm/task with success.
  The warm start is a reward-weighted fit of the initial code mean and std
  (CPU only), plus critic pretraining on those returns. It cannot be used as
  on-policy PPO data: a per-chunk policy does not generate episode-constant
  codes.
- No Stage 0 gain reached p<.05, so most of the success variation is motor
  noise. Warm-starting may just fit noise; the zero-init arms show this.
- KW: the Stage 0 KV data belong to the final-layer fixed-projection map,
  not the new adapter. KW initializes the adapter to reproduce that map on
  the final layer (other layers zero) and warm-starts the code distribution;
  this must be checked on CPU to match Stage 0 outputs before launch.
- Critic pretraining needs per-chunk features. The 46,980 recorded chunks
  are actual chunk records; features are recomputed on the frozen motor if
  not stored.

## Budget

Each learner: up to 30 updates × 64 episodes (≤1.15M controls), ~4 GPU-h
plus evaluation, per the gate plan. Four learners ≈ 16 GPU-h, cap 20 GPU-h
including failures. The earlier ~1M-action approval was for one learner;
**four learners need the human to confirm this cap.**

## Before launch

1. Implement the all-layer KV adapter in `robocasa_interface`, with a
   zero-output identity test and an all-layer causality/shape test.
2. Implement the warm-start fit and critic pretraining (CPU).
3. Run core tests on allocated Linux, pin a `main` commit, fresh run IDs.
