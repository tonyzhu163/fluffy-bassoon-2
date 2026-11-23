# Matched-source visual readers in native UR5e rollouts

Changing only the yaw readout improved second-attempt success on this eight-task π₀.₅ panel. The learned reader succeeded on 46 of 64 held-out queries and the linear reader on 39. Eleven queries succeeded only with the learned reader and four only with the linear reader. The paired difference was +10.94 percentage points (95% task-bootstrap interval +1.56 to +21.88). On 96 total queries the counts were 70 and 62, a difference of +8.33 points (+2.08 to +16.67). [Full native summary, including task effects](evidence/gpu01-summary.json) · [Per-query native outcomes and source hashes](evidence/gpu01-cases.csv)

| Command settings | Queries | Learned successes | Linear successes | Learned only | Linear only | Ties | Learned / linear yaw MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All | 96 | 70 | 62 | 14 | 6 | 76 | 6.37° / 8.93° |
| All held-out | 64 | 46 | 39 | 11 | 4 | 49 | 6.53° / 9.16° |
| Held-out combinations | 32 | 23 | 17 | 8 | 2 | 22 | 6.20° / 11.16° |
| Held-out continuous angles | 32 | 23 | 22 | 3 | 2 | 27 | 6.87° / 7.17° |
| Familiar | 32 | 24 | 23 | 3 | 2 | 27 | 6.06° / 8.47° |

Both readers used 1,344 labeled Panda/Sawyer traces. Both evaluated the same three motion intervals using unit-length planar vectors and equal weights for yaw alignment. Every paired query used the same learned gain and gripper estimates, reset state, prompt, first policy proposal and noise schedule. Success comes from the simulator on a fresh query after a 15-control support trace and a reset. The learned arm's 70/96 uses the common equal-weight solver. The historical headline's 76/96 uses the original learned weighting. Comparing those two totals also spans separate simulator runs. [Frozen source fit, offline uncertainties and feature differences](OFFLINE.md) · [Schedule and balancing checks](LIVE.md)

The evidence supports a visual-readout contribution in this setup. Six of the seven net held-out successes came from new combinations of familiar factor values. On continuous angles the net difference was one success and the paired offline yaw interval included zero. The two readers use different DIS settings, image features and pooling. The fixed camera, eight task identities, single checkpoint and single source fit limit what can be attributed to the architecture or generalized to other settings. The declared native interval resamples eight tasks with replacement 2,000 times, retaining all within-task paired transforms. [Registered prediction](PLAN.md) · [Offline metric intervals](evidence/metrics.json)

Baige Distributed Training job `job-sjubp8q0tg1x` exited zero on revision `08cf720efbb83231342fe22b285038065f254cda` after 1,201 seconds on two Pro6K GPUs (0.67 launcher GPU-hours). An independent readback checked all 96 raw records against the frozen panel and re-counted every cell. All resets and opening proposals matched within pairs, every renderer reported NVIDIA, and native reward sequences agreed with the recorded controls and returns. Full episode records are on the shared run volume under `gpu01/shard-*/`. [Readback and raw location](evidence/gpu01-readback.json) · [Deployment](LAUNCH.md)
