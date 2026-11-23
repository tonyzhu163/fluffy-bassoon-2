# Panda arm rendering changed all ten frozen proposals and neither native outcome

On the two preregistered Kinova3 drawer starts, raw, Kinova render and Panda render each succeeded once. Panda changed the next five proposed controls on every archived frame and took 78 additional controls on the successful native start. The failed start reached the 300-control horizon under all three views. These results establish visual sensitivity in the frozen π₀.₅ motor. They provide no observed reward gain on this panel. [Independent reconstruction](native-readback/reconstruction.json) · [Registered question and starts](PLAN.md)

## Absolute native outcomes

Each cell gives the **undiscounted native return and number of executed controls**. The task was LIBERO-Pro2 BASE, `libero_goal/3`, with the original instruction “open the top layer of the drawer and put the bowl inside”. Both layout triplets completed all three branches. There were zero incomplete or censored triplets.

| Kinova layout (seed) | Raw | Kinova render | Panda render | Panda − raw return |
|---|---:|---:|---:|---:|
| 16 (41810316) | 1 in 206 | 1 in 206 | 1 in 284 | 0 |
| 17 (41810317) | 0 in 300 | 0 in 300 | 0 in 300 | 0 |

On layout 16, each branch received zero at every preceding control and 1 at successful native termination. Layout 17 produced 300 zero rewards per branch and truncated at the task horizon. That horizon is an observed failure. The paired success difference was 0 percentage points, with zero rescues, zero losses and two ties. [Six episode records and reward streams](native-readback/gpu01/) · [Paired readback](native-readback/reconstruction.json)

These are two fixed starts on one task. To illustrate sampling uncertainty, if starts were independent draws from a common population, zero discordant outcomes in two pairs gives a one-sided exact 95% upper bound of 77.6% on the probability that Panda changes success in either direction. The two-sided exact 95% success interval for 1/2 in each condition is 1.3–98.7%. The six branch executions represent two paired observations. [Uncertainty calculation](native-readback/reconstruct.py)

## Ten within-frame motor contrasts

One loaded frozen OpenPI `pi05_libero` checkpoint saw each frame four times in the order raw, target-on-target Kinova render, Panda render and raw repeat. The table reports the Euclidean distance over 35 coordinates, the first five 7-dimensional commands after clipping to the native [−1, 1] range. Each pair uses the same frame and diffusion noise. Five measurements share each trajectory.

| Archived layout | Native control | Raw repeat − raw | Kinova render − raw | Panda − Kinova render | Panda − raw |
|---:|---:|---:|---:|---:|---:|
| 14 | 0  | 0 | 0 | 0.112867 | 0.112867 |
| 14 | 30 | 0 | 0 | 0.085931 | 0.085931 |
| 14 | 45 | 0 | 0 | 0.112902 | 0.112902 |
| 14 | 60 | 0 | 0 | 0.189536 | 0.189536 |
| 14 | 90 | 0 | 0 | 0.055773 | 0.055773 |
| 15 | 0  | 0 | 0 | 0.078618 | 0.078618 |
| 15 | 30 | 0 | 0 | 0.122881 | 0.122881 |
| 15 | 45 | 0 | 0 | 0.246927 | 0.246927 |
| 15 | 60 | 0 | 0 | 0.267619 | 0.267619 |
| 15 | 90 | 0 | 0 | 0.491431 | 0.491431 |

The entire 10 × 7 float32 proposal was byte-identical for raw, Kinova render and raw repeat on every frame. All ten Panda distances exceeded their measured repeat difference of exactly zero. The largest unbounded single-coordinate change, 2.0038 on layout 15 at control 60, was the gripper command in the seventh proposed action. The native controller replans after five. The independent readback retains every first-action difference, per-dimension chunk RMS, maximum absolute change and bounded five-command distance. [Four action arrays per frame](native-readback/gpu01/) · [Reconstructed contrasts](native-readback/reconstruction.json)

## Camera, noise and native motor checks

- For all ten frames, CPU reconstruction checked the .npz digest and labels, compared both original RGB arrays, rebuilt both 224-pixel policy images and target state, and regenerated each seeded 10 × 32 noise slice. Their digests matched the GPU record. Panda changed 1,946–8,244 external-camera pixels and zero wrist pixels at policy resolution. The instruction and target proprioception matched across calls. The Kinova full render matched the raw image in both cameras. [Input digests and pixel counts](native-readback/frame-inputs.json) · [Original exposure checksums](evidence/cpu01/SHA256SUMS)
- Both native starts used Kinova3 physics, a seven-dimensional bounded relative OSC pose action at 20 Hz, the released prompt and the nominal motor with NVIDIA rendering. The two starts had distinct physical reset hashes. Within each layout the branches matched on physical reset, scene, perturbation plan, action schema and full noise bank. Raw and Kinova render matched on the complete reset, all 42 or 60 input/noise/proposal records and every reward. [Native episode and call files](native-readback/gpu01/)
- At each matched opening, Panda changed only the external-camera motor input digest and changed the proposed chunk. The wrist, state, prompt and noise matched. The first subsequent motor state diverged at control 5 on both layouts as each branch acquired its own feedback. Every later noise slice matched its registered seed and index, including the extra 15 replans for Panda on layout 16. [Native comparisons](native-readback/reconstruction.json)
- All 102 Kinova-render and 117 Panda-render policy boundaries had valid render traces. Each Panda boundary passed the matched Kinova full render, clean RGB/depth, segmentation and depth-visibility checks in both cameras, with an admitted collision-checked IK solution. Within Panda’s own physical trajectory the substitution changed 2,217–9,926 external pixels on 16 and 2,669–8,664 on 17, with zero wrist pixels changed. The corresponding baseline and Panda trajectories subsequently visited different states. [Layout 16 geometry](native-readback/gpu01/views-16-panda.jsonl) · [Layout 17 geometry](native-readback/gpu01/views-17-panda.jsonl)

## Scheduler and evidence

Baige Distributed Training reported **成功** for `job-69hluhi5wgb7` on November 23, Shanghai time. Submission was 19:12:00. The creation/queue phase lasted 1 h 20 min 5 s. Startup ran from 20:32:05 to 20:32:10. The scheduler recorded 9 min 11 s running and success at 20:41:21. Startup through scheduler success was 556 s on one RTX 4090, **0.1544 GPU-hours** against the one-GPU-hour cap. The launcher ran 20:32:12–20:41:14, 542 s, and exited 0. This monitor made zero submissions and zero retries. [Timestamped observations](native-readback/monitor.tsv) · [Launcher exit](native-readback/gpu01/exit.txt)

The host receipt identifies the deployed pin `2b59cff7e173ee8064905256b98a1dd3fd0f79d5` and OpenPI `215abfb217dbac7d5f1273282331b9b1866c0479`. The CUDA probe tied the single RTX 4090 to the scheduler allocation. The in-job input check passed for ten lossless bundles and two layout manifests. The first real frozen quartet was written at 20:37:41 and the first native episode at 20:39:19. The 31 terminal files were read directly from `baige-dev:/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-source-view-20261123-r1/gpu01`; all local copies matched SHA-256 computed on PFS. [Device probe](native-readback/gpu01/gpu.json) · [Raw evidence hashes](native-readback/GPU01-SHA256SUMS) · [Readback hashes](native-readback/READBACK-SHA256SUMS)
