# Image-estimated roll correction reduces frozen SmolVLA proposal error

On eight standard LIBERO Panda development tasks, estimating camera roll from the two reset images reduced mean action-chunk RMSE from **0.2069 to 0.0172** relative to the upright proposal at the same state, instruction and action noise. The paired reduction was **0.1897**, with a 95% whole-task bootstrap interval of **0.1303 to 0.2517**. All four shifted conditions reached the perfect-correction result on seven tasks. One living-room scene accounted for the remaining gap.

| Camera mount intervention | Raw RMSE | Estimated correction RMSE | Known-transform RMSE | Paired reduction [95% interval] |
| --- | ---: | ---: | ---: | ---: |
| Agent ±90° | 0.1322 | 0.0200 | 0.0017 | 0.1123 [0.0700, 0.1493] |
| Wrist ±90° | 0.2816 | 0.0144 | 0.0012 | 0.2672 [0.1754, 0.3651] |
| Equal weight across mounts and signs | 0.2069 | 0.0172 | 0.0014 | 0.1897 [0.1303, 0.2517] |
| Both upright | 0 | 0.0134 | 0 | −0.0134 [−0.0401, 0] |

Each task contributed one independently seeded noise tensor and one reset measured at all five interventions. Each comparison changed only a camera mount. The known-transform control rotated the **actual shifted render** back into alignment. Its average remaining image error was 0.0088 RGB levels on a 0–255 scale. The independent sampling unit for the intervals is the task (n=8).

**The missed scene matters on the nominal mount too.** The adapter inferred the correct correction on 16/16 rolled wrist images and 14/16 rolled agent images. On the basket task (`libero_10-0`), the agent template chose the opposite correction for both signs. It also applied a spurious 90° correction to the upright agent view, moving the upright proposal by 0.1070 RMSE and affecting both wrist-roll comparisons. The other seven upright proposals were unchanged.

The pooled first-control RMSE fell from 0.1943 to 0.0061. Across full chunks, translation RMSE fell from 0.1837 to 0.0215, rotation RMSE from 0.0285 to 0.0027, and gripper RMSE from 0.3985 to 0.0195. All values use the checkpoint's unnormalized command channels. [Per-task results and bootstrap intervals](evidence/summary.json) · [Per-condition estimates, pixel errors and noise hashes](evidence/scores.json) · [Distinct frozen proposals](evidence/proposals.npz)

**Frozen forward-pass consistency, 0 GPU-hours.** The 450,046,176-parameter policy ran on CPU in evaluation mode with zero trainable parameters. A fresh process compared all 500 stored checkpoint tensors to their loaded values and reproduced the first nominal proposal bit for bit. All eight nominal chunks were finite 50×7 arrays, with positive temporal variation in every channel. Changing only the first task's noise moved its nominal chunk by 0.0153 RMSE. An independent render of that task reproduced the same physics and all sensor/shifted image hashes. [Reload and replay verification](evidence/verification.json) · [Rendered input provenance](evidence/frames.json) · [Checkpoint, processors, calibration and reproduction](README.md)

These results support a small input adapter for fixed-camera quarter turns when the scene supplies reliable orientation cues. They cover one initial Panda pose per task and three roll classes. The development split is held out from template fitting, while the public motor itself was trained on LIBERO. Arbitrary pitch, yaw, translation, occlusion, motion through an episode, different robot arms and native task success remain untested. A matched rollout on the basket scene would test whether the false correction also disrupts contact or task completion.
