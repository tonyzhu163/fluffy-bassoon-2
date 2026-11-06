# RoboTwin re-entry audit: why VLRLA's RLinf baseline has not replicated

2026-11-06, Asia/Shanghai. Read-only audit by a Claude Code session; no jobs,
GPU time or active-run changes. Sources: repository evidence, the
[VLRLA paper](https://arxiv.org/html/2610.19517v1#S5), the
[RLinf RoboTwin guide](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/robotwin.html)
and the [Sidney model card](https://huggingface.co/SidneyXie/pi05_robotwin).

## What VLRLA reports and what it leaves out

Table 1 gives SFT pi0.5 / RLinf GRPO overall averages of 50.9 / 60.0 (on
`place_phone_stand`, 36.7 / 50.8). The paper states only that it uses "the official
implementation provided by RLinf" on 8 A100-80G GPUs, starting from its own
pi0.5 SFT model. It does not give GRPO steps, budget, per-task or multi-task
training, or hyperparameters; Table S2 covers PPO history collection only. GRPO
improves on the six "unseen" tasks too, so it was trained online on all twelve.
RLinf publishes no pi0.5 GRPO RoboTwin recipe or result (pi0.5 PPO on
`adjust_bottle` only) and ships presets for only two of the twelve tasks. The baseline is
therefore an unreleased VLRLA run, not an RLinf artifact we can rerun.

## Our attempts

| Attempt | Motor | Outcome |
| --- | --- | --- |
| Motus-era source PPO/SAC (Oct 29–Nov 1) | Motus | Source success flat near 17%; SAC 11.5%; no memory benefit ([FINDINGS](../../docs/FINDINGS.md#robotwin-motus--vlrla-style)) |
| Matched Sidney PPO/GRPO (Nov 6) | `SidneyXie/pi05_robotwin@e49e2ab6` | 2 iterations / 40 updates, phone stand; training collections 17.2% → 21.9% (PPO) and 17.2% → 18.0% (GRPO); no held-out evaluation ([result](../robotwin-sidney-matched-baselines-20261106-r1/RESULT.md)) |

## Root causes, ranked

1. **Base policy far below its published level in our deployment.** Sidney reports
   66% on `place_phone_stand` (Easy, 100 episodes); our RLinf collection starts at 17%.
   Differences from LeRobot: RLinf sends the 50-target chunk through
   `gen_sparse_reward_data`, which downsamples the path to 500 or fewer physics points
   (LeRobot calls `take_action` per target); a generated instruction instead of
   `place phone stand`; K5 Flow-SDE instead of K10; camera settings. The
   initial-policy diagnostic `job-on94vctdunlt` and the controller/prompt 2×2
   `job-qtjluslns1jj` (running at 16:21) address this.
2. **Different SFT motor from VLRLA's.** VLRLA's SFT and Sidney differ by large
   amounts per task (e.g. `place_dual_shoes` 17.2 vs 72). Raw Table 1 percentages are
   not a matched target; only gains over our own SFT are.
3. **Far too little training.** 2 of VLRLA's 25 PPO iterations on 4 rather than 8 GPUs;
   about half of the groups have constant reward (61/128), so GRPO gets no signal.
4. **GRPO is our own adaptation** (group 8, PPO-derived hyperparameters); no
   released pi0.5 GRPO recipe exists.
5. **Missing infrastructure for six or twelve tasks:** five tasks have no seed banks
   (`pick_diverse_bottles`, `place_object_stand`, `scan_object`, `place_dual_shoes`,
   `stamp_seal`); task configs exist only for `adjust_bottle` and `place_empty_cup`; the
   fast path does not measure executed actions.
6. **Fragile code:** the re-entry runtime lives in `outputs/` run folders imported
   across runs (`robotwin-sidney-rlinf-ppo-20261105-r1/train_entry.py`,
   `robotwin-sidney-qualification-20261105-r1/model_worker.py`); 27 Motus-era
   `src/fluffy_bassoon/meta_rl/robotwin_*.py` modules are unused.
7. **Budget:** 37.00 GPU-hours actual at 16:21 (45.31 with current reservations) of the 50-hour cap;
   one task-pair of 2 iterations cost 16.27 GPU-hours.

## Path back

1. **Gate 0, base policy parity.** Reproduce Sidney's published success on
   `place_phone_stand` and one or two other tasks, using LeRobot's per-target controller,
   the short prompt and K10 ODE, then the same inside RLinf. Do not train until
   deterministic eval is within noise of the card.
2. **Gate 1, one-task positive control.** On a task with mid-range SFT headroom,
   run RLinf PPO long enough to show a held-out paired gain
   (RLinf's own pi0.5 PPO `adjust_bottle` recipe is the closest released reference),
   then GRPO at matched controls. Lesson carried forward: source learners must
   improve before histories can teach a meta-learner.
3. **Gate 2, scale.** Promote the shared runtime into `src/` with core tests,
   build the five missing seed banks, then request the six- or twelve-task budget
   (above the 50-hour cap; needs human approval).
4. Report VLRLA Table 1 as reference only; our baseline is gain over our own SFT.
