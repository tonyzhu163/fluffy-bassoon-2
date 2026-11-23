# A fixed Panda render recovers part of the frozen motor's proposal on all ten tasks

A Panda arm generated from the UR5e end-effector pose and jaw reduced mean translation-proposal drift from **0.05959 to 0.02822 RMS** on full native UR5e images. All ten LIBERO-Goal task pairs improved. The mean paired reduction was **0.03137** (95% task-bootstrap interval **0.01818–0.04631**). Moving the same sprite off the arm gave **0.09216** RMS. The aligned insert beat this displaced control in all ten pairs by a mean **0.06394** [0.05144, 0.07480].

Each score compares the frozen SmolVLA's 50-step translation chunk with its proposal on the actual Panda external image, in unnormalized policy command units. Lower scores indicate closer agreement. This CPU experiment used static native simulator images and measured forward-pass consistency. Task success and physical execution remain unmeasured.

## Per-task translation RMS

All edited columns use the same full, unaltered native UR5e frame as the background. **True joints** replaces the inferred Panda joints with the hidden Panda angles in the fixed rig. **True visibility** also clips that sprite to the task-scene Panda silhouette. **Matched cutout** pastes the actual visible Panda scene pixels. These three columns use privileged evaluation inputs.

| LIBERO-Goal task | No-op | Public IK | Displaced | True joints | True visibility | Matched cutout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Open the middle drawer of the cabinet | 0.03845 | 0.00889 | 0.09370 | 0.00889 | 0.00855 | 0.00855 |
| Open the top drawer and put the bowl inside | 0.07195 | 0.04406 | 0.09501 | 0.04375 | 0.06266 | 0.06266 |
| Push the plate to the front of the stove | 0.07523 | 0.03609 | 0.07968 | 0.03625 | 0.03871 | 0.03871 |
| Put the bowl on the plate | 0.02075 | 0.01020 | 0.07552 | 0.01016 | 0.01076 | 0.01076 |
| Put the bowl on the stove | 0.08796 | 0.03468 | 0.09912 | 0.03510 | 0.05245 | 0.05245 |
| Put the bowl on top of the cabinet | 0.02150 | 0.01330 | 0.09700 | 0.01332 | 0.01312 | 0.01312 |
| Put the cream cheese in the bowl | 0.12965 | 0.04944 | 0.12518 | 0.04958 | 0.08923 | 0.08923 |
| Put the wine bottle on the rack | 0.03417 | 0.02485 | 0.04657 | 0.02498 | 0.02470 | 0.02472 |
| Put the wine bottle on top of the cabinet | 0.08454 | 0.03721 | 0.10688 | 0.03770 | 0.04441 | 0.04441 |
| Turn on the stove | 0.03170 | 0.02352 | 0.10295 | 0.02348 | 0.02320 | 0.02320 |
| **Mean** | **0.05959** | **0.02822** | **0.09216** | **0.02832** | **0.03678** | **0.03678** |

The immediately executable first translation also moved closer on 8/10 pairs (mean RMS 0.02960 to 0.01851, paired reduction 0.01110 [0.00220, 0.01820]). Whole-chunk rotation RMS went from 0.00688 to 0.00440 and gripper RMS from 0.24555 to 0.19396. Comparisons are within channel group.

## Which source-rendering errors moved the motor?

1. **IK redundancy.** The rest-posture IK matched the tool pose within 1.6e-14 m and 2.2e-14 rad. The mean joint RMS difference from hidden Panda angles was 0.02175 rad (range 0.00020–0.04343). In the fixed rig, inferred versus true angles gave a mean silhouette IoU of 0.99880. Switching to true angles changed the proposals directly by 0.00053 translation RMS [0.00038, 0.00068]. Distance to the common source reference changed by just +0.00009 on average.
2. **Occlusion.** At the true joint angles the fixed rig rendered 5–151 extra arm pixels behind task-scene objects per frame (mean 65.3), and missed zero scene-visible arm pixels. Mean true-rig/scene silhouette IoU was 0.98330. Applying correct visibility moved the proposals directly by **0.01231** RMS [0.00459, 0.02137]. On the cream-cheese task alone it raised distance to the source from 0.04958 to 0.08923. Overall, this correction *increased* reference distance by 0.00846 [0.00184, 0.01706]. An insert that covers object pixels can improve this action metric despite a visible compositing error.
3. **Pixels and residual arm.** Switching the visibility-corrected fixed-rig RGB to actual matched Panda pixels moved the proposal directly by about 0.00001 RMS on average, with identical chunks on nine tasks. The public paste leaves UR5e-only protrusions covering 0.36–2.35% of frame area. All edited conditions retain the native background and its shadows outside the insert. Even the matched cutout leaves 0.03678 reference distance. These controls isolate the IK and visibility interventions. They do not apportion that remaining distance to individual background effects.

## Deployment inputs and scope

The adapter uses the released Panda geometry, materials, kinematics, fixed rest posture, exact simulator base/camera transforms and intrinsics, shared world/tool frame, fixed tabletop lighting, and a common PandaGripper with known jaw coordinates. The fixed installation came from a separate LIBERO-Spatial task at seed 201. No fitting or demonstrations were needed. Each query supplies the UR5e external RGB, public hand pose and finger positions. A repeat with pose-and-jaw-only records reproduced all ten sprites byte for byte. The sprite alpha comes from the isolated rig's geometry IDs. Per-query Panda imagery, scene segmentation and joint angles enter only the benchmark fixture, scoring reference and diagnostic controls.

The motor received identical target wrist, state, prompt and Gaussian noise for every condition within each pair. Both the quaternion convention and mapped pose use the fixed source installation. All ten reference chunk hashes and no-op scores reproduced the parent experiment. The SHA-256-verified SmolVLA checkpoint ran strictly loaded in float32 eval mode with zero trainable parameters and zero GPU-hours.

Support is one contact-free near-reset pose and one noise draw for each of ten tasks, all in one tabletop installation. Both fingers were at the same reset opening in all queries. The intervals resample whole task pairs 20,000 times. Camera/lighting calibration error, other grippers and postures far from rest remain untested.

[Deployment rig](rig.py) · [Fixed installation](installation.json) · [Reproduction and pins](REPRODUCE.md) · [Per-task geometry](geometry.csv) · [All task scores](scores.csv) · [Paired estimates](summary.json)
