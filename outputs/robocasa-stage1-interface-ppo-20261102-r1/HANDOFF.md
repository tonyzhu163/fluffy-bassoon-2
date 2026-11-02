# Stage 1 launch handoff (needs a Baige owner)

Human-approved 2026-11-02: two learners, V0 (velocity) and K0 (all-layer
KV), OpenDrawer, simulator first-success reward, ≤20 GPU-h in total.
Contract: [PLAN.md](PLAN.md). Use the latest `main` at launch (at least
`1c4c7039`), recorded with fresh run IDs `…-V0-r1` and `…-K0-r1`.

## What exists

- `InterfaceSpec(kind="velocity" | "conditioning_all", steering_dim=32)` and
  `InterfaceLearner` in `meta_rl/interface_ppo.py`: γ .99, λ .95, 10 epochs,
  minibatch 32, target_kl .02, LR 3e-4, full resumable checkpoints.
- `FrozenRoboCasaInterface.prepare_batch/execute_batch` and
  `RoboCasaInterfaceEnvironment(reward_arm="S", interface_horizon=600)` in
  `meta_rl/robocasa_interface.py`.

## What the owner writes

No driver joins these yet; `robocasa_residual_parallel.py` is the
action-residual lane driver to follow. The driver needs:

1. Calibrate `spec.fraction` per arm once on GPU, before update 1, with the
   Stage 0 `calibrate` rule (public initial observations only, action RMS
   matched to the 0.25 action-residual target). Velocity may reuse the Stage
   0 selected value. Check zero code is bitwise native on that GPU.
2. Fit feature normalization on the first rollout, then 30 updates × 64
   TRAIN episodes, early stop by the gate plan's plateau rule.
3. Log per update: TRAIN success, policy code std, KL, clip fraction; every 5
   updates a deterministic 16-case TRAIN panel and DEV (monitoring only).
4. Final: paired held-out 50-case ID test, final vs native (exact McNemar).
   Pass: last-3 vs first-3 updates ≥ 15 points and p < .05.
5. Save a full checkpoint (policy, optimizer, sampler, environment, RNG) at
   every update.

Probe GPUs inside the job step with `scripts/[train]gpu_probe.py
--expected-count`, run core tests on the allocation first, and update the
Stage 1 coordination row with job ID, progress and last check every 20 min.
