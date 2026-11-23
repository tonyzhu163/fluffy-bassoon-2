# SymMeta ICML paper

## Reported comparisons

- [x] The abstract, first figure and main results table distinguish a calibrated second attempt from single-attempt streaming and an uncorrected retry. The task identities come from the LIBERO-Pro2.0 held-out split. The twelve command settings and the arm substitutions are part of this study. [Paper](main.tex)
- [x] The UR5e × π₀.₅ closed-loop FK comparison is complete. Both methods used the same learned gain and gripper estimates. FK used joint encoder measurements and trusted world-frame kinematics for yaw. Native success was 78/96 for FK and 76/96 for the seed-41 visual reader. The paired difference was +2.1 percentage points (95% task-cluster interval −2.1 to +7.3). [Pairing protocol](../../outputs/symmeta-fk-ur5e-20261123-r1/LAUNCH.md) · [Verified result](../../outputs/symmeta-fk-ur5e-20261123-r1/RESULT.md)
- [x] The brightness-translation comparator uses global weighted brightness constancy. Its projection is fitted on nominal source-arm probes. The published ICBA and ReGround methods are cited as related work, and all measured Perceiver and affine results are identified as local comparisons. The geometric prior cites Choy et al. (CVPR 2020). [References](references.bib)
- [x] The paper uses the official descriptions of the five static perturbations and reports the visual-stress results separately from the base-scene experiment. [LIBERO-Pro2.0 catalogue, Table 15](https://arxiv.org/html/2610.24587v1)

## Open experiments

- [ ] Determine how FK yaw affects native success on IIWA × π₀.₅, Kinova3 × π₀.₅ and UR5e × π₀. Report the number of qualified pairs for each arm. [Cross-arm qualification](../../outputs/symmeta-fk-crossarm-20261123-r1/RESULT.md)
- [ ] Compare with a classical visual tracker given target camera geometry or a gripper fiducial. Report the target calibration data and cost.
- [ ] Measure native success under moderate camera roll or azimuth, or test a scene-fixed reference for yaw. The existing camera tests measure offline yaw. [Camera evidence](../../outputs/symmeta-camera-command-20261123-r1/RESULT.md)
- [ ] Replay the same policy-noise bank for demonstration-fitted adapters and the seed-41 reader on the 32 common queries. The existing comparison matches query resets and uses separate policy-noise draws.
- [ ] Evaluate fresh layouts and randomly sampled yaw and gain values. Reader seeds currently reuse the same 96 queries and eight task clusters.
- [ ] Measure how horizontal command access, learned chunk weights and vertical-parallax correction affect native success. The existing parallax ablation measures yaw error.
- [ ] Test whether a second camera helps recover continuous 3D rotations and whether the planar result transfers to physical hardware.

## Open reporting and maintenance

- [ ] Complete the appendix episode ledger beyond the four main panels. [Paired reanalysis](../../outputs/symmeta-paired-statistics-20261123-r1/RESULT.md)
- [ ] Review the impact statement before submission.
- [ ] Consider an optional JAX/Flax dependency group for running the reader tests off cluster.
