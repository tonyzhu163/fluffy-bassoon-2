# SymMeta ICML paper

## Reported comparisons

- [x] The abstract, first figure and main results table distinguish a calibrated second attempt from single-attempt streaming and an uncorrected retry. The task identities come from the LIBERO-Pro2.0 held-out split. The twelve command settings and the arm substitutions are part of this study. [Paper](main.tex)
- [x] The UR5e × π₀.₅ closed-loop FK comparison is complete. Both methods used the same learned gain and gripper estimates. FK used joint encoder measurements and trusted world-frame kinematics for yaw. Native success was 78/96 for FK and 76/96 for the seed-41 visual reader. The paired difference was +2.1 percentage points (95% task-cluster interval −2.1 to +7.3). [Pairing protocol](../../outputs/symmeta-fk-ur5e-20261123-r1/LAUNCH.md) · [Verified result](../../outputs/symmeta-fk-ur5e-20261123-r1/RESULT.md)
- [x] The brightness-translation comparator uses global weighted brightness constancy. Its projection is fitted on nominal source-arm probes. The published ICBA and ReGround methods are cited as related work, and all measured Perceiver and affine results are identified as local comparisons. The geometric prior cites Choy et al. (CVPR 2020). [References](references.bib)
- [x] The paper uses the official descriptions of the five static perturbations and reports the visual-stress results separately from the base-scene experiment. [LIBERO-Pro2.0 catalogue, Table 15](https://arxiv.org/html/2610.24587v1)
- [x] On the original 1,344 seven-shift traces, changing only the Perceiver learning-rate schedule reduced yaw MAE on 64 recorded OOD UR5e supports from 95.63° to 83.37° (paired change −12.26°, 95% task-bootstrap interval −17.04° to −7.39°). Both fits used 32 positional slots, the original sampler, batch 64, 3,500 updates and seeds 41–43. [Verified control](../../outputs/symmeta-seven-schedule-20261123-r1/RESULT.md)
- [x] Figure 4 compares the tuned direct Perceiver with fresh same-seed SymMeta on all four native retry panels. Each used 96 queries per seed, including 64 OOD, for 1,152 pairs across seeds 41–43. The Perceiver succeeded on 54.2–79.7% of held-out retries. All four paired eight-task intervals included zero. The appendix gives both readers’ success counts and intervals overall and on OOD queries. The tuned weights were refitted and frozen before rollouts. [Native results](../../outputs/symmeta-perceiver-retry-20261123-r1/RESULT.md) · [Verified figure and paper](../../outputs/symmeta-perceiver-retry-20261123-r1/PAPER-QA.md)

- [x] The related work identifies the published rotation, gain and latency experiments and the success of target-fitted linear calibration on the two static shifts. The appendix documents the target-data budgets for the released rotation condition and the passive visual reader. [Published Table 1](https://arxiv.org/html/2611.20244v1#S4)

- [x] Reproduced the released ReGround 90° rotation cell and tested frozen streaming SymMeta on all 1,400 assigned episodes. The paper reports success, unsafe incidence, calibration budgets and opening agreement. A separate calibrated retry on 280 released seed-0 starts reports the paired effect of adding wrist correction. [Single-attempt result](../../outputs/reground-rotation-reference-20261123-r1/RESULT.md) · [Calibrated retry result](../../outputs/reground-retry-wrist-20261123-r1/RESULT-RETRY02.md)

## Open experiments

- [ ] Determine how FK yaw affects native success on IIWA × π₀.₅, Kinova3 × π₀.₅ and UR5e × π₀. Report the number of qualified pairs for each arm. [Cross-arm qualification](../../outputs/symmeta-fk-crossarm-20261123-r1/RESULT.md)
- [ ] Compare with a classical visual tracker given target camera geometry or a gripper fiducial. Report the target calibration data and cost.
- [ ] Measure native success under moderate camera roll or azimuth, or test a scene-fixed reference for yaw. The existing camera tests measure offline yaw. [Camera evidence](../../outputs/symmeta-camera-command-20261123-r1/RESULT.md)
- [ ] Replay the same policy-noise bank for demonstration-fitted adapters and the seed-41 reader on the 32 common queries. The existing comparison matches query resets and uses separate policy-noise draws.
- [ ] Evaluate new query layouts and randomly sampled yaw and gain magnitudes in native retries. The four-panel evaluation used twelve fixed interfaces on eight tasks at layout 13. The continuous-label training draws and offline relabelings reuse recorded motions.
- [ ] Measure how horizontal command access, learned chunk weights and vertical-parallax correction affect native success. The existing parallax ablation measures yaw error.
- [ ] Test whether a second camera helps recover continuous 3D rotations and whether the planar result transfers to physical hardware.
- [ ] Test whether a second independent command direction improves full-map recovery at fixed support duration. The recorded median singular-value ratio was 0.099. Procrustes and visual ridge had general-map errors of 0.830 and 0.819 versus 1.406 without adaptation. The full-map Perceiver improved axis swaps to 0.413, but had a mean general-map error of 15.212. Active or longer supports require new traces. [Offline result](../../outputs/symmeta-wider-family-20261123-r1/RESULT.md)

## Open reporting and maintenance

- [ ] Complete the appendix episode ledger beyond the four main panels. [Paired reanalysis](../../outputs/symmeta-paired-statistics-20261123-r1/RESULT.md)
- [ ] Review the impact statement before submission.
- [ ] Consider an optional JAX/Flax dependency group for running the reader tests off cluster.
