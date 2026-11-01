# Flora-Rewarder PPO

**Current first step (2026-11-01).** The paper stays about in-context RL for
frozen VLAs ([PAPER.md](PAPER.md)). Before more ICRL work, get an ordinary RL
loop working where the reward comes from a learned video rewarder instead of
the simulator. That loop is the reward source later in-context learners need
on tasks without a success checker.

Live plan and status: [run plan](../outputs/flora-ppo-comparison-20261101-r1/PLAN.md),
[coordination](../outputs/coordination.md).

## Rewarder

- [`nvidia/Flora-Rewarder`](https://huggingface.co/nvidia/Flora-Rewarder)
  ([project page](https://research.nvidia.com/labs/gear/flora)): a 27B video
  reward model. Outputs an episode score 1–5 and per-step
  normal/suboptimal/failed labels. Non-commercial license; download with the
  user's own Hugging Face authentication to project NVMe.
- Too slow for dense per-step reward. Use one terminal outcome label per
  episode. Its reported failed-step recall is only 53%, and its training
  simulators are not published, so measure its agreement with the simulator
  before trusting it on a task.

## Benchmarks, in order

1. **LIBERO** is a test bed only, not a reported benchmark: pi0.5 leaves too
   little headroom. Use it to get the loop working, starting from RLinf's
   published 40-task few-shot pi0.5 SFT checkpoint, and to measure Flora
   against simulator success.
2. **RoboCasa365** is the reported benchmark, starting with atomic tasks from
   its 50-task evaluation subset; composite tasks come later. It is
   NVIDIA's GR00T benchmark, closer in appearance to the rewarder's likely
   training data and far from saturated. Always report true simulator success.

## First test: LIBERO90 task 44, simulator reward vs Flora reward

A single-seed pilot to decide whether the idea is worth pursuing; the
[run plan](../outputs/flora-ppo-comparison-20261101-r1/PLAN.md) is authoritative.
All other jobs are paused for it.

- **Policy:** the published PyTorch
  [`RLinf/RLinf-Pi05-LIBERO-SFT@45ccfcc4`](https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-SFT/tree/45ccfcc4e28634f1576ebf78cab0fbe2fd82432d),
  trained with 40 demonstrations across the four ten-task LIBERO suites
  ([paper, section 5.2.1](https://arxiv.org/html/2510.25889v1#S5.SS2.SSS1)).
  Both arms use the same weights and bundled normalization through RLinf's
  native loader, with strict backbone loading and a fresh shared baseline.
  No additional SFT or JAX conversion is required. This is a different
  initializer from historical OpenPI40 `c565…`; its success rate is measured
  anew. RLinf130 is excluded because its SFT includes task 44.
- **Recipe:** RLinf's published pi0.5 LIBERO PPO config restricted to task 44
  (`KITCHEN_SCENE9_turn_on_the_stove`), 32 envs × 4 rollout epochs at H400
  (~53k actions per iteration). Training disables success termination so
  episode length cannot leak success. One terminal reward per episode, with no
  extra reward terms.
- **Arms:** S gets 1 if the simulator says the stove is on in the final state,
  otherwise 0. F gets Flora's graded `(expected_score − 1) / 4` partial credit;
  there is no threshold and no tuned mapping. F's trainer can never read the
  simulator label, which is logged to a sidecar for analysis.
- **Steps:** (1) Flora smoke test on one episode. (2) 64 baseline episodes
  labeled by both, giving the start rate, score distributions and AUROC; stop
  if the start rate is outside 5–70% or AUROC < 0.7. (3) A relabel hook before
  GAE plus CPU tests: S with a stub rewarder reproduces the plain config, and F
  cannot see the simulator label. (4) Up to 10 iterations per arm, with 50
  simulator evaluation episodes (states 40–49 × 5 draws) after each. (5)
  `RESULT.md`: true-success curves against actions and GPU time, Flora score vs
  truth (rising score with flat success means reward exploitation), and costs
  with Flora's compute charged to F.
- **Stopping:** plateau after ≥6 iterations when best success gains < 4 points
  over the last four evaluations (futility if < 10 points above baseline);
  ceiling at three evaluations ≥ 48/50; hard action, time and wall-clock limits.
  Decisions use simulator success only. Stop at a closed boundary with full
  recovery.
- **Budget:** ~76k all-in actions per iteration with 4/8 workers, including
  six unscored evaluation padding episodes, under 1M per arm. GPU-hours
  are uncapped for these two arms by human authorization, but each scheduler
  job may request at most 8 GPUs and must run under 8 hours. Logs are plain
  text heartbeats, with no dashboard.

Task 44 has a historical simulator-reward control: ordinary DSRL took the
separate OpenPI40 native pi0.5 initializer from about 5/20 to 20/20 there
([FINDINGS.md](FINDINGS.md#libero)). This is not the starting baseline or a PPO
result for the ready PyTorch checkpoint used in this comparison.
