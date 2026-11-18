# OpenDrawer RLinf flow_sde PPO, simulator first-success reward

2026-11-02. **Human-approved simulator-only run; execution delegated to parent
chat `01a19b74` / subagent `flow_sde_ppo`.** The [execution binding](EXECUTION.md)
records the qualified runtime and recovery policy; current jobs are tracked in
[coordination](../coordination.md).

## Why

Residual PPO stayed flat on OpenDrawer TRAIN (S 72→67, F 71→71 of 128;
[free checks](../robocasa-ppo-free-checks-20261102-r1/RESULT.md)). Against
RLinf's `robocasa365_opendrawer_ppo_openpi` recipe and VLRLA's teacher PPO
([protocol](../../docs/literature/vlrla-protocol.md#32-reward-and-ppo-settings)),
our learner (1) left pi0.5's sampling noise outside the policy, (2) used a
final-control reward with γ=1, and (3) ran about 1% of the data. This run is
FLORA-PPO gate 2's positive control with the published recipe instead of our
interface. Full-weight PPO is a reward check, not the frozen-motor baseline
([FLORA-PPO](../../docs/FLORA-PPO.md#gates-on-each-benchmark)).

## Human decisions (2026-11-02)

- Simulator reward only for now; the Flora arm waits for this to pass.
- First-success reward, as RLinf and VLRLA.
- Budget about 1M source actions per per-task learner. Spend it on this one
  task first; extend to other tasks only if it passes. This run replaces the
  joint five-task 6×80 template as the next PPO step.

## Contract

Config: [`configs/native_ppo/robocasa365_opendrawer_pi05_sim_ppo.yaml`](../../configs/native_ppo/robocasa365_opendrawer_pi05_sim_ppo.yaml).

| Item | Setting | Source |
| --- | --- | --- |
| Motor | public pi0.5 Human300 c484/75000 torch conversion, H50, 10 flow steps, 3 views | same inputs as `robocasa365_pi05_eval.yaml` |
| Trained | action expert + value head; VLM frozen (`train_expert_only`) | VLRLA |
| Exploration | `flow_sde`, noise 0.3, one denoise step per chunk | VLRLA (RLinf recipe uses 0.5) |
| Value | action-expert features, detached input | VLRLA |
| Task | OpenDrawer, `atomic_seen`, split `target` | our earlier native panels |
| Reward | +1 at first success, episode terminates, later chunks masked | RLinf / VLRLA |
| Discount | γ 0.99, λ 0.95, chunk-level | RLinf / VLRLA |
| Horizon | 400 train and eval (first-success maximum279 in the limited timing panel, not a population p99) | RLinf recipe; [timing](../robocasa-pi05-success-timing-20261102-r3/RESULT.md) |
| Rollout | 32 envs × 4 epochs = 128 episodes, 1,024 chunk samples per iteration | RLinf recipe |
| Update | 4 epochs, global batch 256 (16 optimizer steps), micro 16 | RLinf recipe, batch scaled for H50 |
| Optimizer | AdamW, actor 5e-6 / value 1e-4, betas (0.9, 0.95), wd 0.01, clip 1.0 | RLinf / VLRLA |
| Loss | clip 0.2, dual clip 3.0, value clip 0.2, Huber 10, no entropy or KL | RLinf / VLRLA |
| Iterations | 18 | budget |
| Eval | 64 fixed-reset cases, flow ODE, full 400 steps, before (only_eval) and at iteration 18 | budget |

Budget: 18 × 51,200 train + 2 × 25,600 eval = **972,800 actions**. All
environments step the full 400 controls even after success (checked in
`robocasa365_env.py`), so nominal equals executed.
The one-iteration qualification counts as the first of these18 iterations;
never add18 new iterations after it. All failed/diagnostic calls count against
the shared1M escrow, and valid completed evaluation panels are reused on an
explicitly bound recovery rather than collected again.

## Before launch (owner)

1. Use the patched RLinf copy the native eval runs use
   (`sources/robocasa365-public-pi05-nnx-leaves-20261101-r1/RLinf`); the
   `action_space.gripper_index` and `binarize_base_mode_control` keys exist
   only there. Record its commit and this repository's commit.
2. Confirm the converted checkpoint loads under `task: rl` with a freshly
   initialized value head (missing value-head keys only).
3. Run the GPU probe inside the job step, then one iteration with
   `runner.max_epochs=1` and check: 128 episodes, 1,024 chunk samples,
   16 optimizer steps, finite losses, actor/critic memory, iteration time.
4. Check the before eval's `success_once` is within 5–70%. Native OpenDrawer
   was 58% at its final control; first success at 400 may read higher. If
   above 70%, stop and ask; DrawerToCounter (46%) is the backup.

## Readout and decisions

- Log per iteration: TRAIN `env/success_once` (128 episodes), actor/critic
  loss, approximate KL, clip fraction, gradient norm, value explained
  variance, iteration time.
- **Pass:** TRAIN success over iterations 16–18 beats iterations 1–3 by
  ≥ 10 points (384 vs 384 episodes, about 2.8 SE at 50%), and the
  iteration-18 eval beats before on the same 64 cases (exact McNemar,
  p < 0.05) on `success_once`. Also report `success_at_end`.
- **Pass → next:** Flora arm with the identical config (separate approval
  and Flora calibration at 400), then other tasks under their own 1M.
- **Fail:** report curves before any retry. Retry order, one change at a
  time and each a new run ID: noise 0.5 (RLinf recipe); 5 flow steps; then
  reconsider the task.
- Save full resumable checkpoints every 6 iterations.

## Not changed

Active runs, the Stage 0 interface screen, AD/DPT readouts and existing
calibrations are unchanged.
