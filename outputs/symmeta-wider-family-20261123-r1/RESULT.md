# Wider-family command-map recovery on UR5e

Baige Distributed Training job `job-4tjjgreq8u9k` completed PR #7 at `4546516cecedc08d5c8a7d41cae719b4ceee2246` on one RTX 4090. The launcher exited 0 after 28 min 29 s (0.475 GPU-hours from launcher start to exit). The production evaluator ran its full 112,500 updates across seeds 41, 42 and 43. All 432 method–condition–seed rows are present. [Pinned plan and launcher](https://github.com/tonyzhu163/fluffy-bassoon-2/tree/4546516cecedc08d5c8a7d41cae719b4ceee2246/outputs/symmeta-wider-family-20261123-r1) · [Summary JSON](gpu01w454/wider-family-summary.json) · [Training log](gpu01w454/run.log) · [Device probe](gpu01w454/gpu.json) · [Exit receipt](gpu01w454/exit.txt) · [Independent checks](gpu01w454/verification.json)

## A tuned Perceiver recovers yaw on the paper’s held-out shifts

The tuned Perceiver had lower mean absolute yaw error than the paper’s SymMeta checkpoint in each of three seeds on the same 64 out-of-distribution (OOD) UR5e supports. It learned the yaw, gain and gripper outputs directly, without a Procrustes yaw solve. Both factor-output readers use the specified yaw/gain/gripper parameterization and receive images, issued commands and local robot measurements from 15 controls. The last column measures planar command-map error across all 96 **recorded** supports, including the 32 familiar cases. [Per-seed measurements](gpu01w454/wider-family-summary.json)

| Reader and training condition | OOD yaw MAE by seed, 41 / 42 / 43 | OOD yaw MAE, seed mean ± SD | Mean operator error, all 96 |
| --- | ---: | ---: | ---: |
| Perceiver, original seven-shift checkpoints | 97.08° / 92.34° / 97.48° | 95.63° ± 2.33° | 0.967 |
| Perceiver, continuous labels, original **scaling-sweep** schedule | 78.75° / 112.80° / 99.02° | 96.86° ± 13.98° | 1.380 |
| Perceiver, continuous labels, lower rate and warmup | 5.42° / 5.83° / 5.48° | **5.58° ± 0.18°** | **0.146** |
| SymMeta, original seven-shift checkpoints | 6.82° / 7.00° / 7.29° | 7.04° ± 0.19° | 0.154 |
| SymMeta, continuous labels | 6.45° / 6.07° / 5.55° | 6.03° ± 0.37° | 0.147 |

The two continuous-label Perceiver runs used the same source traces, relabeling, architecture, minibatch sampling and 2,500-update budget. Changing the peak learning rate from 0.0015 to 0.0003 and adding 200 warmup updates reduced mean OOD yaw error from 96.86° to 5.58°. This paired comparison identifies the effect of those schedule changes together. The original seven-shift checkpoints used a separate training recipe. Their schedule-only control and the tuned reader’s native retry success on all four arm–policy panels are open. [Continuous training code](eval_wider_family.py) · [Original training code](../xemb-icml-campaign-20261121-r1/stage1_offline_gate.py)

## Offline operator error

Lower values indicate more accurate command-map recovery. Entries give the mean error and its population standard deviation across the three seed means. All five fixed solvers use the same frozen visual reader within each seed. The 1,466 source traces came from Panda and Sawyer. The 96 held-out UR5e traces contributed four interface draws per relabeled family. The delay comparison uses two valid chunks on both sides since a one-chunk shift leaves two matched motions in each three-chunk recording. It scores recovery of the original spatial map under delayed commands. The recorded out-of-distribution yaw error for the frozen Procrustes reader reproduced the archived three-seed result of 7.04° ± 0.19°.

| Solver or model | In-family | Unequal gains | Shear | Axis swap | General 2×2 | Two-chunk control | 250 ms delay |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No adaptation | 1.321 | 1.330 | 1.336 | 1.437 | 1.406 | 1.268 | 1.268 |
| Procrustes | 0.149 ± 0.004 | 0.222 ± 0.002 | 0.303 ± 0.004 | 1.257 ± 0.036 | 0.830 ± 0.017 | 0.162 ± 0.008 | 0.437 ± 0.018 |
| Factored least squares | 6.037 ± 0.002 | 6.037 ± 0.002 | 6.037 ± 0.002 | 6.037 ± 0.002 | 6.037 ± 0.002 | 6.608 ± 0.002 | 16.443 ± 0.002 |
| Visual least squares | 4.205 ± 0.473 | 4.205 ± 0.473 | 4.205 ± 0.473 | 4.205 ± 0.473 | 4.205 ± 0.473 | 6.499 ± 1.179 | 13.688 ± 2.749 |
| Factored ridge | 0.152 ± 0.002 | 0.199 ± 0.002 | 0.284 ± 0.001 | 2.141 ± 0.0002 | 1.939 ± 0.002 | 0.142 ± 0.008 | 0.290 ± 0.005 |
| Visual ridge | 0.166 ± 0.002 | 0.227 ± 0.002 | 0.286 ± 0.003 | 1.367 ± 0.003 | 0.819 ± 0.003 | 0.153 ± 0.006 | 0.374 ± 0.009 |
| Perceiver, continuous family, warmup | 0.150 ± 0.0004 | 0.205 ± 0.001 | 0.297 ± 0.001 | 1.133 ± 0.007 | 1.028 ± 0.002 | 0.147 ± 0.008 | 0.262 ± 0.018 |
| Perceiver, full map, trained on all families | 0.244 ± 0.014 | 0.794 ± 0.604 | 0.432 ± 0.083 | 0.413 ± 0.018 | 15.212 ± 11.003 | 0.730 ± 0.600 | 63.792 ± 87.212 |

The in-family median command singular-value ratio was 0.099 (quartiles 0.058 and 0.136). The unregularized fits had high error on these short, directionally concentrated commands. The five relabeled families reuse the same motions. For a full-rank unregularized visual fit, an invertible change of command coordinates transforms the fitted operator along with the target, preserving the residual map error. The repeated values therefore share an algebraic explanation. Source-selected ridge regularization brought the factored fit close to Procrustes in-family and reduced error on unequal gains, shear and delay. The factored ridge fit had higher error on axis swaps and general maps. The three-chunk recordings limit conclusions about recovery from longer supports.

Training the full-map Perceiver across families reduced axis-swap error in every seed. On general maps its seed means were 11.38, 30.19 and 4.07, with a mean of the three within-seed medians of 1.37. Large errors on a subset of cases strongly affect the mean. The continuously relabeled factor-output Perceiver with warmup reached 0.150 mean error in-family. Its configuration changes both warmup and learning rate, so the comparison measures their combined effect.

Each relabel reuses the recorded images and motion. Across families, 12.7–15.8% of relabeled planar commands exceeded the unit box in at least one axis. The draws share the same underlying 96 supports. The experiment measures offline calibration. A claim about closed-loop task success would require rollouts.
