# Diagnose the Sidney phone-stand gap

The model card reports 66/100 easy. The frozen native benchmark stops at 400 targets. Pinned LeRobot 92f96f33 configures 1,200 targets, uses the task name as the default phone prompt, and seeds consecutive episodes from 1000. Its setup omits eval_mode. Native RoboTwin sets its internal limit only in evaluation mode. The phone task, native controller and task configs are byte-identical across the LeRobot Docker pin 0aeea2d6 and our fa8d97b0. The two runtime stacks have different SAPIEN versions. The card does not pin the evaluation used for its table.

Use the same original weights, batch-one eager sampler, K10/H50, saved processors, native cameras and physics. First load the model through the card's PI05Policy.from_pretrained and compare every saved tensor. Rotate three conditions on each of 16 indices, with shared per-index Gaussian seeds. On expert-qualified frozen seeds use (i) actual scene prompt and (ii) short task name. Use (iii) short task name on release-default sequential seeds 1000–1015. Keep all sequential reset failures visible. Stop at success or 1200. Score at 400 and 1200 on the same trajectory. Record the height gained, closest phone/stand functional-point alignment, and both grippers at closest approach. These boundary samples measure failure modes; native success also checks substeps.

The first two arms estimate prompt and horizon effects on identical seeds. The third contrasts cohorts under a shared controller and sampler. This is an isolation experiment; the uploader's exact seeds, software and sampling are unknown. Do not train on these cases.

## Paired wrist clipping test

Training dataset revision 1287871839fae2296bc27b88a5457c3e1eba8e1f, phone episode 19250, has a visible dark fingertip in both wrist views. Its reset state equals the initial evaluation joint vector. In the SAPIEN 3.0.1 baseline the fingertip is mostly clipped. The original camera requests a 0.1 m near plane. Test the two wrist cameras at 0.01 m, leaving head-camera optics, poses, resolution, controller, 400 targets, original weights, noise seeds, and 100 frozen clean prompts and scenes fixed. Read the effective near value from both live camera objects. Compare aligned inputs, individual success switches and exact first action tensors against the frozen native baseline. Distinguish input appearance correction from measured policy success.

## Randomized SFT baseline

Repeat the 1 cm wrist setting on the 100 preselected randomized scenes. Use their frozen instructions, action-noise seeds and 400-target horizon, with the original released policy. Compare each episode to its 10 cm randomized partner and verify both wrists' clipping distances plus exact initial head RGB and joint state. The model card supplies a comparison value for clean phone stand only. Allocate one Pro6K GPU for at most 190 minutes.
