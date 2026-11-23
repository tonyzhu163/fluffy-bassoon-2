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
- [ ] Test command maps outside yaw, gain and gripper sign: unequal axis gains, shear, an axis swap, a general 2×2 map and a 250 ms delay. Compare the Procrustes solver, least-squares maps on the same reader, and Perceivers trained on continuously sampled maps. The offline run is scheduled. Run native success only where the offline ranking is not decisive. [Run plan](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/4546516cecedc08d5c8a7d41cae719b4ceee2246/outputs/symmeta-wider-family-20261123-r1/PLAN.md)
- [ ] Train a Perceiver with one token per control, a tuned learning rate and a linear-map output before attributing its OOD failure to learned encoders. The headline failure is at 7 training transforms. The width-192 scaling model reaches 15.9°, 17.6° and 14.7° OOD yaw error per seed at 32 transforms, and 12.4°, 101.1° and 18.8° at 128. [Scaling summary](../../outputs/xemb-icml-campaign-20261121-r1/evidence/stage3-scaling-summary.json)
- [ ] Reproduce the released ReGround 90° rotation result and run SymMeta on the same cases. Report the target-robot data each method uses. [Plan](../../outputs/reground-rotation-reference-20261123-r1/PLAN.md)

## Open reporting and maintenance

- [ ] Complete the appendix episode ledger beyond the four main panels. [Paired reanalysis](../../outputs/symmeta-paired-statistics-20261123-r1/RESULT.md)
- [ ] Review the impact statement before submission.
- [ ] State in related work that ReGround's π₀.₅ results cover a 90° action-frame rotation and a gain change as well as latency, and that its linear-calibration baseline reaches 0.98 and 0.94 success there. State the difference: ReGround fits on target-robot state from 1,000–100,000 transitions, and SymMeta uses 15 controls and no trusted kinematics. [ReGround](https://arxiv.org/abs/2611.20244)
- [ ] Align the project findings index with the paper's comparator names.
- [ ] Consider an optional JAX/Flax dependency group for running the reader tests off cluster.
