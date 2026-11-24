# The tuned Perceiver follows visual motion and the reported local motion frame

On the 64 held-out UR5e supports, the three shipped Perceivers estimated upright command yaw with **5.83° mean absolute error**. When the true interface yaw changed by +90° and both native input channels changed with it, predicted yaw moved by **+90.36°**. With exactly the same images, rotating issued XY alone moved the estimate **+9.95°**. Rotating only the reported end-effector displacement moved it **+83.94°**, even though the true command map in that control stayed fixed. The opposite rotation gave the same ordering. Reversing the temporal order of the same images moved the estimate almost half a turn. These responses favor a reader that uses visual motion together with the robot’s reported displacement direction. A pure command prior or a static-scene rule cannot explain the paired predictions. [Preregistered comparisons](PLAN.md) · [Per-trace predictions](evidence/predictions.json.gz)

## What changed the estimate

Each entry is a mean on the same 64 physical supports. Brackets give paired 95% percentile intervals from resampling eight whole tasks 20,000 times. Each trace contributes the mean of its three seed-level metrics. The OOD designation follows the original recordings through the interventions. A positive interface increment rotates the new issued coordinates clockwise for the same world command. The local-only rows use the corresponding rotation of reported end-effector XY. A positive displayed image roll is counterclockwise. [All 96, OOD, seed and task estimates](evidence/summary.json)

| Fixed-camera input change | True yaw change | Signed prediction shift, ° [95% interval] | Yaw MAE, ° [95% interval] |
|---|---:|---:|---:|
| Original support | 0° | 0 | 5.83 [4.44, 7.50] |
| Issued XY alone, +90° interface | +90° | +9.95 [9.29, 10.54] | 79.60 [78.64, 80.54] |
| Issued XY alone, −90° interface | −90° | −8.13 [−8.76, −7.41] | 82.32 [81.40, 83.32] |
| Local end-effector XY alone, +90° interface convention | 0° | +83.94 [82.71, 84.93] | 84.39 [82.17, 86.43] |
| Local end-effector XY alone, −90° interface convention | 0° | −80.69 [−82.13, −79.27] | 80.24 [79.11, 81.26] |
| Full native relabel, +90° | +90° | +90.36 [89.91, 90.76] | 5.93 [4.80, 7.46] |
| Full native relabel, −90° | −90° | −90.64 [−90.95, −90.30] | 5.52 [4.30, 7.13] |

For yaw increment d, the relabeled issued vector R(−d)u preserves the physical command because gR(y+d)R(−d)u equals gR(y)u. The largest coordinate residual was 2.22 × 10⁻¹⁶. Across all chunks, both the command and local XY transformations agreed with the original PR7 relabeler. Only the full relabel preserves the native coupling between action and sensor frames. Its mean absolute deviation from the required prediction shift was 2.25° for +90° and 2.10° for −90°. When only the issued direction changed, all 192 OOD seed–trace predictions for each sign fell closer to zero shift than to the required quarter turn. [Native relabeler](../../src/fluffy_bassoon/xemb/wider_family.py) · [Tape and transformation checks](evidence/sources.json)

| Displayed image roll, true map fixed | Signed prediction shift, ° [95% interval] | Distance to the camera-predicted shift, ° | Yaw MAE, ° [95% interval] |
|---|---:|---:|---:|
| +90° counterclockwise | −86.39 [−94.98, −74.96] | 35.50 [33.35, 37.93] | 90.11 [85.99, 94.66] |
| −90° clockwise | +84.27 [77.73, 92.12] | 28.50 [26.32, 30.71] | 84.71 [78.62, 91.80] |

The quarter turns changed only the adapter’s view of the recorded movement. Both before/after frames rotated together and DIS was recomputed from pixels. The opposite-signed camera-angle prediction was closer than zero shift for 159/192 and 165/192 seed–trace pairs respectively. The mean signed shifts track camera angle, with substantial individual errors around that angle. The earlier physical camera-orbit results were obtained with SymMeta. This Perceiver measurement covers exact image-plane roll. [Camera experiment and convention](../symmeta-camera-command-20261123-r1/RESULT.md) · [Paired estimates and residuals](evidence/summary.json)

## Reversing the same image pairs

| Within-chunk change | Simple command/visual rule predicts | Observed absolute shift, ° [95% interval] | MAE against recorded map plus any command relabel, ° |
|---|---:|---:|---:|
| Negate issued XY | Half turn | 7.31 [6.16, 8.42] | 170.95 [169.96, 171.93] |
| Reverse before/after and recompute flow | Half turn | 169.77 [167.30, 172.24] | 167.11 [163.89, 170.19] |
| Both | Return to original | 165.86 [163.04, 168.59] | 16.16 [13.11, 19.30] |

Reversal preserved the unordered pixels, their average-luminance gradients, RGB difference magnitude, spatial coordinates, task, chunk order, commands and proprioception. It changed signed temporal contrast and the recomputed flow, including possible changes in flow magnitude and occlusion. The single visual flip gave a mean signed shift of +177.11° [174.90, 179.28] when expressed around the predicted half turn. Flipping both channels left the output 165.86° away from its original value. The lower 16.16° MAE in that cell results from scoring against a command map that also moved by 180°. These are tests of frozen input dependence. No new robot episode realized the reversed trajectory. [Signed shifts including principal-angle values](evidence/summary.json)

## Why it works and which changes address the failures

The result fits a physical regularity of this benchmark. When the action and proprioceptive frames turn together, local end-effector displacement supplies a genuine direction cue for the command frame, and the camera supplies a second view of the movement. Issued XY has some influence, but changing it alone gives little leverage over yaw. A policy can also favor particular directions on particular tasks. Such action priors can coexist with the measured cue use. The inputs conflict in the isolated interventions, so their response sizes do not give additive contributions to upright accuracy. This test identifies behavioral dependencies and leaves open the learned computation, including whether any layer explicitly aligns directions. [Native interface and local observation map](../../src/fluffy_bassoon/xemb/embodiment.py) · [Prior tuned training comparison](../symmeta-wider-family-20261123-r1/RESULT.md)

The checkpoint turns each of the three valid chunks into a token by concatenating a CNN visual embedding, seven issued controls and five proprioceptive increments. Eight learned latents attend to the tokens. Three four-head latent self-attention blocks and a pooled output head predict yaw, gain and gripper polarity. Cross-attention into learned latent space is the architectural connection to the Perceiver literature. The performance and cue-use claims here come from the shipped model and its measured outputs. [Implemented reader](../../src/fluffy_bassoon/xemb/equiv_v2.py) · [Jaegle et al., Perceiver architecture](https://proceedings.mlr.press/v139/jaegle21a/jaegle21a.pdf)

Two architecture changes address different observed failures. Neither has a measured gain in this study.

- **When camera orientation can change**, express visual displacement in a known robot or world frame before inferring yaw. Camera extrinsics, a known scene reference or independently calibrated joint kinematics can supply that reference. Motion direction in an unanchored camera can absorb camera rotation into the estimated command yaw. An alignment head alone would inherit that ambiguity.
- **When the issued command and reported local frame can change independently**, give the yaw estimator separate inputs for issued commands and achieved, frame-anchored displacement. A confidence-weighted rotation fit from issued XY to measured motion can guarantee the correct response to issued-coordinate relabeling when the motion is informative. Proprioception can help estimate displacement quality or slip, with its coordinate frame modeled explicitly. The full native relabel is already accurate here, so this proposal targets the observed failure under decoupled frames.

## Identity, provenance and reproduction

We restored the three archived **train02pro** PR7-recipe refits actually shipped for native inference. The original PR7 publication used a separate training run. These tests performed zero optimizer updates. Each original support was identified by task, transform, index, seed-specific reference row and NPZ digest. The input is three 64 × 64 before/after RGB pairs, real 7D mean commands, 3D local end-effector and 2D jaw increments, the original three-slot valid mask and dense DIS flow. The loader also checks the reset image. The checkpoint consumes nine derived visual channels per chunk. [Input shapes, digests, checkpoint configs and paths](evidence/sources.json) · [Original exporter](../symmeta-perceiver-retry-20261123-r1/export.py)

| Seed | OOD CPU replay MAE | OOD shipped GPU export MAE |
|---:|---:|---:|
| 41 | 5.645° | 5.642° |
| 42 | 5.992° | 5.985° |
| 43 | 5.846° | 5.839° |

Across all 96 supports and three seeds, the mean absolute CPU versus archived GPU yaw difference was 0.0226°, with a maximum of 0.1174°. The maximum gain difference was 0.002125, and all 288 gripper signs agreed. Hardware and inference batch size differed. Within the CPU replay, undoing the command and image rotations reproduced the input bytes, DIS, all nine visual features and all three output factors exactly on every trace. A separate readback checked the archived keys and recomputed all 3,744 errors, signed deltas and tracking residuals. Eight task clusters limit the uncertainty estimate to this panel and these fits. The endpoint is offline yaw. [Readback](evidence/readback.json) · [Qualification](evidence/qualification.log) · [Completed run](evidence/run.log)

| Source | Revision |
|---|---|
| PR7 wider-family experiment | `4546516cecedc08d5c8a7d41cae719b4ceee2246` |
| Shipped train02pro checkpoints and predictions | `ba3c41e25575d07d7bd8bbfaae2797e7cf9dc0a9` |
| Original camera score archive and support inventory | `3577fcbe0976f7c4b51713d25362aeaacce701c7` |
| This frozen replay and preregistration | `d3d1b45a459db248a3bf113ffdecc5d8a37545e5` |

From the pinned Baige checkout, run the command below with a fresh attempt name. The launcher reads retained data from project shared storage. The full replay used CPU cores 38 and 39, 19.6 s measured replay time, 30.5 CPU-seconds and 570 MiB peak resident memory. The complete launcher took 20 s and used zero GPU-hours. [Launcher](run.sh) · [Replay](replay.py) · [Paired task bootstrap](summarize.py) · [Exit receipt](evidence/exit.txt)

```sh
bash outputs/symmeta-perceiver-mechanism-20261124-r1/run.sh \
  d3d1b45a459db248a3bf113ffdecc5d8a37545e5 fresh_attempt full
```
