# Genuine source views narrow the UR5e registration advantage

**The spatially calibrated Perceiver caught most of the original UR5e yaw gap. Explicit registration retained a 1.41° advantage on the same 64 genuine ±30° supports (95% paired task interval 0.34° to 2.35°).** On the preregistered Kinova3 assigned ±36° views, the Perceiver averaged 0.40° lower error, with the interval permitting either reader to lead. These results narrow the earlier claim to a small, orbit-direction-dependent UR5e advantage at this training budget.

A registers the observed image to the public reset-height plane before estimating the command map. B keeps the image on the observed grid and receives the same H and local counterpart coordinates before visual pooling. In each exposure regime, the two readers received identical source observations, control relabels, initialization by seed, 15-control support budget, parameter count and optimizer opportunities. The genuine and matched digital regimes differ on the same replayed source trajectories. The digital companion projects each fresh nominal recording with the genuine calibrated camera H, so it retains the projection's missing pixels and lacks depth-dependent scene changes. [Registered design](PLAN.md) · [Matched exposure control](MATCHED-EXPOSURE.md) · [Source evidence](SOURCE.md)

## Paired yaw results

The gap is A minus B absolute wrapped yaw error in degrees. Negative values favor registration. The exposure effect is the genuine gap minus the matched digital gap. Intervals are paired 95% bootstrap intervals over eight whole tasks, retaining all cases, views and seeds. Both source arms are evaluated on 64 validation tapes from eight source tasks excluded from every fit. Each target panel contains 64 supports. The Kinova3 preregistration assigns one signed orbit per case.

| Evaluation | Genuine A / B MAE | Genuine gap [95% interval] | Matched digital A / B MAE | Matched digital gap [95% interval] | Exposure effect [95% interval] |
|---|---:|---:|---:|---:|---:|
| Disjoint Panda/Sawyer, ±18°/±45° | 17.31 / 16.36 | +0.96 [−0.07, +2.08] | 19.63 / 25.04 | −5.40 [−8.42, −2.81] | +6.36 [+4.25, +8.78] |
| UR5e, both ±30° | 6.93 / 8.34 | −1.41 [−2.35, −0.34] | 8.31 / 11.54 | −3.23 [−4.76, −1.77] | +1.82 [+0.40, +3.96] |
| UR5e, assigned native orbit | 6.88 / 8.80 | −1.92 [−2.95, −0.84] | 8.66 / 11.67 | −3.01 [−4.82, −1.22] | +1.09 [−0.80, +3.25] |
| Kinova3, assigned native orbit | 9.27 / 8.87 | +0.40 [−2.25, +2.93] | 11.48 / 13.75 | −2.27 [−5.24, +0.32] | +2.67 [−1.13, +5.99] |
| Kinova3, both ±36° | 9.19 / 9.33 | −0.15 [−2.13, +1.89] | 11.49 / 13.10 | −1.60 [−4.47, +0.56] | +1.46 [−0.74, +3.92] |

On the UR5e both-orbit endpoint, genuine exposure reduced B's error by 3.20° [2.06, 4.79] and A's by 1.38° [0.73, 1.96] relative to their matched digital companions. At +30° the resulting A–B difference was −3.29° [−4.75, −1.68]. At −30° it was +0.47° [−0.48, +1.40]. Individual realized seed differences were −0.87°, −4.01° and +0.64°. Nominal A/B was 4.65°/5.05° with a paired difference of −0.40° [−1.12, +0.30]. On Kinova3, nominal A/B was 6.50°/6.07°, difference +0.43° [−0.47, +1.24]. [UR5e budgets, signs, seeds and tasks](evidence/fit01/UR5e-summary.json) · [Kinova3 budgets, signs, seeds and tasks](evidence/fit01/Kinova3-summary.json) · [Source validation](evidence/readback01/source-summary.json)

The original 24-task synthetic-only fit had an 8.13° UR5e advantage. On identical observations the gap narrowed by 6.72° [3.80, 9.79] from that pair to the genuinely exposed pair. Holding the 16-task split, actual H, tape/control draws and optimizer opportunity fixed isolates the smaller 1.82° change in the table. The larger historical comparison also changes task coverage, the camera distribution and digital exposure. On Kinova3's original assigned views the earlier gap was −3.38°. The paired shift to +0.40° was +3.77° [+0.62, +6.72], after matching every original pixel, H, sensor and support digest. The Kinova3 matched exposure interaction has wider uncertainty. [Primary UR5e audit](../symmeta-camera-frame-review-20261124-r1/physical-readback.json) · [Kinova3 original-pair readback](evidence/readback01/Kinova3-original-change.json)

## Exposure and learning

The earlier independent eight-tape source check already found an A advantage on ±15° physical orbits, 2.91° and 1.90° respectively, on one familiar task/layout. We reused those predictions. The new source set covers sixteen training and eight disjoint validation tasks. Both arms saw camera-center displacements of 0.137–0.881 m and nonnominal H center shifts of 5.85–32.80 pixels. Every genuine pose was crossed with full observations and four separately assigned edge masks. On the selected budget, each seed drew 80,000 genuine-stream and 80,000 original digital examples per estimator. The genuine stream contained exactly 40,000 all-valid examples per seed. Moved-camera draws numbered 71,034/71,057/71,169. All 5,760 training tape/pose/mask combinations were visited in every seed. Each of 128 physical training motions was reused 625 times on average, with fresh continuous interface relabels. [Earlier source comparison](evidence/reused-source-check.json) · [Per-seed draw counts](evidence/readback01/draw-summary.json) · [Signed geometry and missingness](evidence/readback01/geometry-summary.json)

Source validation mean orbital yaw MAE across the three reader seeds was as follows. Fresh initialization excludes every validation task from digital pretraining as well as genuine training. The prespecified candidate budgets were 2,500 and 5,000. B under genuine exposure selected 5,000 for all four models. Initialization, all fixed training-subset curves, angle-specific errors and actual loss are in the checkpoint record.

| Updates | Genuine A | Genuine B | Digital companion A | Digital companion B |
|---:|---:|---:|---:|---:|
| 0 | 112.76 | 112.77 | 112.76 | 112.77 |
| 1,000 | 25.05 | 30.72 | 26.68 | 38.80 |
| 2,500 | 21.51 | 21.29 | 23.09 | 33.33 |
| 3,500 | 23.64 | 21.21 | 22.87 | 30.52 |
| 4,500 | 17.79 | 16.23 | 19.81 | 25.44 |
| 5,000 | 17.31 | 16.36 | 19.63 | 25.04 |

At the prespecified earlier 2,500-step readout, the genuine A–B error was −2.53° [−3.32, −1.68] on both UR5e orbits and −1.17° [−3.06, +0.63] on both Kinova3 orbits. Limited physical support still matters. On the eight reserved tasks, genuine B averaged 15.11° on Panda and 17.61° on Sawyer. The large task-level uncertainty and changing late curves leave its asymptotic performance open. [Full source curves](evidence/fit01/curves.json) · [Selected budget and 24 weight digests](evidence/fit01/sealed.json)

## What the controls establish

With identical raw observations and selected weights, giving the opposite calibrated orbit raised genuine A/B yaw MAE by 47.67°/40.41° on UR5e and 54.05°/45.69° on Kinova3. Identity metadata raised UR5e A/B error by 19.60°/16.06°. Switching between raw and registered feature grids at fixed weights and true H cost 19.23° for A and 16.26° for B on UR5e. This control switches resampling, DIS, validity and local-coordinate channels together. On genuine source validation, introducing one of the four independent edge masks raised B error by 0.55° [0.11, 1.10] relative to the same unmasked observation. B uses calibrated camera information. Its improvement under physical exposure is compatible with a synthetic shortcut, better training coverage of motion or both. Parallax, newly visible content and the projected validity mask change together in the matched digital replacement. [Source controls](evidence/readback01/source-summary.json) · [UR5e fixed-weight input intervention](evidence/readback01/UR5e-cross-summary.json) · [Kinova3 fixed-weight input intervention](evidence/readback01/Kinova3-cross-summary.json)

A reproducible positive A–B interval on genuine held-out orbits would reverse the directional registration claim. The UR5e interval still favors A. The independent Kinova3 interval permits either ordering. The point estimates and matched exposure test support an exposure-dependent advantage and give little reason to expect the original 8.13° gain on a new body. The remaining UR5e effect concerns yaw estimation for a fixed policy, one orbit family and this finite source budget. This run measured native-orbit predictions. Task success under the newly fitted estimators remains unmeasured.

## Native handoff and execution

On the actual assigned observation for native deployment, seed-41 A/B yaw MAE was 6.86°/8.48° for UR5e and 8.89°/8.27° for Kinova3. Both readers recovered all assigned seed-41 gripper signs on both bodies. The full observation, camera, trajectory, checkpoint and predicted yaw/gain/sign are preserved per record. Parent owns any conditional reward comparison on the frozen native designs. Shared exports are under `/mowenjie/fluffy-bassoon-2/outputs/symmeta-camera-real-source-20261124-r1/readback01/`. [UR5e native predictions](evidence/readback01/UR5e-native-seed41.json.gz) · [Kinova3 native predictions](evidence/readback01/Kinova3-native-seed41.json.gz)

Baige Distributed Training job `job-o2g2s29tk504` exited 0 on one exclusive RTX 4090. The launcher ran for 1,389 seconds, 0.386 GPU-hours, including probe, input loading, four three-seed fits, both budgets and target scoring. CPU preparation used 878 seconds total, including the failed first landmark diagnostic. The final code pin was `57fca56ab35a878d8de4672ac33118152fc2d248`. All 24 actual checkpoint digests matched the source-only seal. Each panel contains 10,752 primary and sensitivity prediction rows plus 2,304 fixed-weight cross-input rows. Each seed-41 native file contains 128 model/case rows. [GPU probe](evidence/fit01/gpu.json) · [UR5e release receipt](evidence/fit01/UR5e-receipt.json) · [Kinova3 release receipt](evidence/fit01/Kinova3-receipt.json) · [Independent weight hashes](evidence/readback01/checkpoints.sha256) · [Exit receipt](evidence/fit01/exit.txt)
