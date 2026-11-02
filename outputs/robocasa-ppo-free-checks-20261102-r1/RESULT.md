# OpenDrawer PPO free checks

2026-11-02. Analysis only: no simulator episodes, GPU allocations or optimizer
steps. Reads retained records from the
[original/S/F terminal review](../robocasa-original-pi05-eval50-20261102-r1/TERMINAL-REVIEW.json)
and the [reward-gap investigation](../robocasa-ppo-reward-gap-investigation-20261102-r1/REPORT.json).
One training task, one PPO seed, six updates per arm.

## 1. Paired held-out readout

All three policies share case IDs and per-chunk motor-noise seeds. Exact
two-sided paired (McNemar binomial) tests on discordant cases, from the
terminal review; the S vs original ID value was recomputed by hand (11/5 →
p=0.210).

| Comparison | Panel | Wins | Losses | p |
| --- | --- | ---: | ---: | ---: |
| Simulator PPO vs original | ID | 11 | 5 | 0.21 |
| Flora PPO vs original | ID | 7 | 4 | 0.55 |
| Flora vs simulator PPO | ID | 3 | 6 | 0.51 |
| Simulator PPO vs original | OOD | 11 | 7 | 0.48 |
| Flora PPO vs original | OOD | 9 | 9 | 1.00 |
| Flora vs simulator PPO | OOD | 4 | 8 | 0.39 |

No comparison is significant. Many cases flip in both directions (16–18
discordant of 50 for S vs original), so the panels are noisy per case. Frozen
features are not bitwise identical across runs (max initial base-chunk
difference 0.005 ID, up to 1.94 OOD for F), so part of the flipping may be
numerical drift rather than the residual.

## 2. Flora validity on PPO data

Already answered by the investigation: AUROC 0.925 over 384 Flora-arm TRAIN
episodes; Score 5 on 58/183 failures and 196/201 successes. The planned
near-miss breakdown cannot be done from retained data: final drawer-joint
values were not saved, so frames cannot separate near-threshold opening from
target confusion or occlusion.

## 3. Source improvement (per-update TRAIN success, 64 episodes each)

| Arm | U1 | U2 | U3 | U4 | U5 | U6 | First two | Last two |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Simulator PPO | 35 | 37 | 26 | 32 | 34 | 33 | 72/128 | 67/128 |
| Flora PPO | 36 | 35 | 30 | 29 | 37 | 34 | 71/128 | 71/128 |

Per-batch Flora AUROC ranges 0.881–0.967. DEV (8 cases) stays at 6–7/8 for
both arms from an initial 7/8.

## Conclusion

Neither PPO arm improved on its own TRAIN distribution over six updates. The
held-out gains over the original policy are not significant. The current
OpenDrawer histories therefore contain no learning progress for AD to
distill; an AD/DPT result on them tests the pipeline, not in-context
improvement. Before collecting meta-learning histories, the source PPO needs a
setting where it measurably improves (more updates, larger batches or a
lower-success task), checked on TRAIN success per update.
