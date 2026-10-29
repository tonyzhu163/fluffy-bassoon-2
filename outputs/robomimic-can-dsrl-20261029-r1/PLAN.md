# Original Can DSRL / RL² pilot

2026-10-29, user approved: start on the released low-dimensional RoboMimic Can
benchmark on Baige. Do not change physics or claim a VLA result. Method contract:
[docs/robomimic-can-dsrl.md](../../docs/robomimic-can-dsrl.md).

First qualify the new runtime and learner interface on one exclusive GPU,
16 CPUs, 64 GiB RAM, with a 20-minute scheduler cap. Run the two focused CPU
test modules in that allocation; then a real two-attempt PPO trajectory, an
original DSRL-NA optimizer step, motor sampler parity and complete recovery
continuation. Follow with 8 native trials, K=4 (32 episodes). Preserve all
artifacts on failure. No automatic retry and no learner campaign admitted by
a failed qualification.

After qualification, use measured throughput to admit a pilot within the
remaining original two-GPU-hour test envelope. Targets: original iid DSRL-NA,
retry-sampled DSRL-NA, RL²/GRU and separately trained episode-reset GRU, one
training seed, 48,000 primitive training steps each including DSRL warmup.
Evaluate on the same 8 held-out initial-scene seeds and K=4 attempts. Compare
frozen and online-updated DSRL on the same interaction budget, and normal,
memory-reset and reward-blind GRU. A deadline produces a partial result and a
complete recovery, not a successful learning claim. No new jobs above the
two-hour test envelope without a separate resource decision.

The pilot establishes runtime and learning-signal feasibility. It is far
shorter than the authors' full learning experiments. Do not label it a curve
reproduction, a significant method win or cross-task meta-learning. Longer
learning curves and multiple training seeds are separate follow-on work.

Source code is reviewed against the current `main` before each launch and
committed; record the exact source in every run receipt. Keep run worktrees
immutable. The main clone remains on main and concurrent edits are preserved.
