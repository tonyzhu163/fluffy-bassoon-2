# Public image registration recovers most of the frozen PR7’s camera-induced yaw error

## Problem and hypothesis

Does camera transfer require refitting the passive command reader? We hypothesized that a deterministic public-calibration warp would recover much of the original PR7 Perceiver’s accuracy with every weight frozen. The competing explanation predicted a substantial additional advantage for camera-trained readers on genuine moved views. Before scoring, we favored digital recovery and had less confidence in physical recovery because a single plane approximates a three-dimensional scene. [Predeclared test](PLAN.md)

## Paired test

The same three shipped checkpoints (seeds 41, 42, 43) saw the same three issued command chunks and reported Cartesian and gripper motion on 64 OOD UR5e supports from eight tasks. PR7 trained on Panda and Sawyer. For each moved view, we used the main camera reader’s observed RGB and masks, applied its inverse warp with the public H, recomputed shipped DIS on the registered, neutral-filled frame pairs and passed them to the unchanged nine-channel reader. The opposite-H control changed only the supplied calibration. The registered sensitivity arm supplied A’s exact boundary-masked DIS to the same frozen reader. Reference pixels, task identity, effect labels and H tokens were absent from model inputs. Refitted A registers before extracting features. Refitted B operates in the observed grid and receives public per-pixel reference coordinates, overlap and global H. Both used the common 5,000-update weights selected on B’s source validation. [Checkpoint provenance](../symmeta-perceiver-mechanism-20261124-r1/evidence/sources.json) · [Shared stream](../symmeta-camera-frame-reader-20261124-r1/evaluate.py) · [Refit seal](evidence/refit/sealed.json)

| Yaw MAE in degrees | Digital rolls ±20°, ±45° | Physical orbits ±30° |
|---|---:|---:|
| Frozen, raw observation | 38.23 | 27.65 |
| Frozen, registered observation | **9.28** | **9.59** |
| Frozen, opposite calibration | 66.33 | 59.48 |
| Frozen, registered with A’s exact masked DIS | 7.56 | 9.64 |
| Refitted A, registered | 6.34 | 8.39 |
| Refitted B, spatially calibrated in observed grid | 6.10 | 16.52 |
| Frozen registered minus raw, 95% paired interval | −28.95 [−32.77, −24.73] | −18.06 [−19.35, −16.88] |
| Frozen registered minus refitted A | +2.94 [+0.98, +5.10] | +1.19 [+0.41, +2.06] |
| Frozen registered minus refitted B | +3.18 [+1.51, +5.13] | −6.93 [−9.43, −4.08] |

All contrasts pair the same source tape, chronological observed frames, supplied H and seed. Each task gets equal weight after averaging its supports, views and reader seeds. The intervals resample eight whole tasks 20,000 times with seed 1124. A separate program reconstructed losses directly from all individual predictions and enumerated the exact eight-task bootstrap distribution. The frozen-versus-raw gains held in every physical task and seed. Digital gains by seed were 30.36°, 28.30°, 28.18° and physical gains were 17.02°, 17.81°, 19.36°. [Digital task and seed estimates](evidence/digital-join/joined-summary.json) · [Physical task and seed estimates](evidence/physical-join/joined-summary.json) · [Independent readback](evidence/readback/readback.json)

**Digital.** Raw inference with a fixed final-angle correction reached 16.18°. Registering the pixels lowered error a further 6.90° [4.55, 9.01]. The registered nonquarter error still exceeded upright by 3.45° [1.52, 5.52]. At both exact quarter turns, inverse registration recovered the original image pairs, DIS and all three heads exactly. All 192 upright OOD support-seed predictions exactly reproduced the earlier CPU mechanism replay. Against the archived GPU export the maximum differences were 0.1174° yaw and 0.00213 gain, with identical gripper signs. [All digital predictions](evidence/digital02/digital-predictions.json.gz) · [Parity measurements](evidence/digital02/digital-checks.json)

**Physical.** The frozen registered errors were 7.05° at −30° and 12.12° at +30°, versus raw errors of 32.46° and 22.84°. Pooled error exceeded the same trajectories at nominal by 3.77° [1.58, 5.74]. Refitted A’s advantage concentrated at +30°, where it reduced error by 3.39° [2.17, 4.52] against frozen registration. At −30° frozen registration minus A was −1.00° [−2.05, +0.09]. The original eight trajectories on two tasks had raw and registered errors of 28.65° and 7.63°. On the additional 56 trajectories the corresponding equal-task means were 27.56° and 9.73°. The combined result includes each trajectory once. The physical bank rendered all views on each same-state replay using archived command means held for five steps. Fresh replays can differ from the original collection at individual primitive steps. [All physical predictions](evidence/physical/physical-predictions.json.gz) · [Paired geometry and replay fidelity](../symmeta-camera-pose-pairs-20261124-r1/RESULT.md)

**Gain and gripper.** At physical ±30°, raw, frozen registered, A and B had absolute log-gain errors of 0.1224, 0.1219, 0.1035 and 0.1115. Frozen registration minus raw was −0.0006 [−0.0028, +0.0019], and minus A was +0.0183 [+0.0094, +0.0280]. The same four arms misclassified 3, 4, 1 and 0 gripper signs among 384 support-seed-view predictions. At nonquarter digital rolls their log-gain errors were 0.1169, 0.1124, 0.1137 and 0.1060, with 3, 2, 0 and 0 gripper errors out of 768. Each export includes the actual yaw, gain and sign inferred on the moved view. [Digital refitted factors](evidence/readback/digital-refit-factors.json.gz) · [Physical refitted factors](evidence/readback/physical-refit-factors.json.gz)

## Updated prior

I now favor registration of the existing checkpoint as the explanation for most of the yaw recovery on the tested physical orbits. It removed 18.06° of the 21.83° excess over the frozen same-trajectory nominal view. Opposite calibration raised yaw error by 49.89° [47.98, 51.62] on the identical physical observations. The geometric information matters, and source refitting alone with B’s input encoding recovered substantially less of the physical loss.

The stronger claim that source camera training adds negligible value lost support. Refitted A improved physical yaw by another 1.19° and improved gain estimation. On digital nonquarter rolls both refits improved yaw by about 3° relative to the primary frozen procedure. With A’s exact masked DIS, the frozen-reader gaps to A narrowed to 1.22° digitally (exact task interval 0.18° to 2.16°) and measured 1.25° physically (0.47° to 2.11°). The comparison with A includes its trained calibration features as well as changes to learned weights. These measurements support inference on the observed UR5e camera grid. Native retry success and transfer to other cameras or arms require their own evidence.

## Reproduction and calibration identity

The proof and both full panels used Baige Linux cores 50 and 51. Frozen inference and comparison jobs consumed 200.85 measured CPU seconds in total, including the digital repeat, joins and readbacks. This worker used zero GPU-hours and collected zero rollouts. The upstream source fit and official physical scoring used their owners’ allocations. [Run manifest and files](READY.json)

Run the following commands from the frozen code checkout on Baige. The named attempts already exist in the recorded run, so new executions need fresh attempt names. The eight-support proof used the historical original-release bank. The physical run used the complete bank whose SHA-256 starts with 8071d5d4. The primary refit predictions are in fit01 (digital) and phyGpu01 (physical) of the matched camera reader.

```bash
pin=7022a39ab0384f515a1930eaa81e180eab829bf2
worker=outputs/symmeta-camera-frozen-preprocess-20261124-r1
banks=/mowenjie/fluffy-bassoon-2/outputs/symmeta-camera-pose-pairs-20261124-r1/cpu01
bash "$worker/run.sh" "$pin" proof01 proof "$banks/original-release/bank.json"
OPENBLAS_CORETYPE=Sandybridge bash "$worker/run.sh" "$pin" digital02 digital
bash "$worker/run.sh" "$pin" physical01 physical "$banks/ood64/bank.json"
```

The first cross-worker join caught two secondary synthetic-plane H hashes that differed across host linear-algebra kernels. Replaying the original code with a kernel that reproduced both upstream public matrix hashes gave exact nine-view calibration joins. All 8,064 digital predictions retained identical registered RGB, DIS, controls and decoded heads across the two runs. The primary digital rolls matched on the first pass. Physical H came directly from the serialized bank. [Matrix comparison](evidence/h-check/h.jsonl) · [All-row kernel replay](evidence/readback/kernel-overlap.json) · [Frozen launcher](run.sh) · [Independent readback code](readback.py)
