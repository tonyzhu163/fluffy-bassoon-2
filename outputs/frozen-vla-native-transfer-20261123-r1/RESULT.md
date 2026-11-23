# The calibrated 3D motion correction rescued one drawer start and lost both moka-pot starts

Frozen π0.5 on the simulated Panda solved all six released BASE starts. With the same checkpoint on the simulated Kinova3, untreated actions solved four. A full 3D translation correction solved three and a planar correction solved four. Both used only 15 calibration controls on independent start 11. The more accurate motion predictor lost one success against planar correction and one against untreated actions. The native evidence does not support promoting it over either control.

| Task and released start | Panda | Kinova3 untreated | Kinova3 3D | Kinova3 planar | Kinova3 privileged tracker |
| --- | ---: | ---: | ---: | ---: | ---: |
| Drawer and bowl 14 | 175 | fail | 179 | 169 | 172 |
| Drawer and bowl 15 | 233 | fail | fail | fail | fail |
| Two moka pots 14 | 495 | 496 | fail | 450 | 497 |
| Two moka pots 15 | 336 | 361 | fail | fail | fail |
| Cream cheese and bowl 14 | 86 | 94 | 94 | 94 | fail |
| Cream cheese and bowl 15 | 86 | 99 | 99 | 96 | 86 |
| **Native successes** | **6/6** | **4/6** | **3/6** | **4/6** | **3/6** |

Numbers are controls to the first native success. Failed episodes exhausted their respective horizons. Relative to untreated Kinova3, 3D gained one and lost two (−16.7 percentage points), and planar gained one and lost one (0 points). The only discordant 3D–planar comparison favored planar. Their paired two-sided sign-test probabilities are each 1.0. Two layouts per task and selection using historical layout 13 limit inference. Post-reset objects and target arm dynamics were matched within each target comparison.

A paired source-trajectory tracker solved drawer 14 despite raw target failure. The target could physically accomplish that query under the common Cartesian controller. The tracker also failed on starts the target solved unaided. Its losses leave tracking error, contact and grasp differences unresolved. No measured cause singles out the wrist, gripper or visual appearance.

**Independent untreated repetition changes the result on both moka starts.** A second distributed job on a different RTX 4090 solved 2/6 with raw Kinova3. Across all six starts its reset dictionaries exactly equaled ours, including first camera images, policy-state checksum, instruction and flow-noise bank. All six first proposal hashes differed. The other four episode outcomes and lengths agreed. Within this job, all three Kinova3 action conditions shared each of the six first proposal hashes and all physical, image and noise checksums. Keep raw comparisons paired inside their jobs. The jobs share policy and simulator source commit, model checkpoint, package environment, native action schema and action dtype. Their launchers differ in CPU thread counts, library search paths, a CV2 overlay, physical GPU, and rollout order. The artifacts do not record a per-process compiler/autotune profile or full action arrays for the visual job, leaving both the cause and magnitude of its opening-action difference unmeasured.

Baige Distributed Training job `job-xaugtearcacf` exited 0 on one RTX 4090 at revision `a8a997e7d06bb131d3f9710204c58e106f0f0904`. The workload ran 760 seconds (0.211 GPU-hours). An independent readback validated all 30 native reward tapes, trial identities, physical resets, horizon, noise and starting-policy matches. Full per-episode actions and measurements are on Baige at `/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-native-transfer-20261123-r1/gpu01`. [Evidence and hashes](gpu-evidence.json) · [Comparisons](native-summary.json) · [Independent verification](verify.py) · [Registered conditions](PLAN.md)
