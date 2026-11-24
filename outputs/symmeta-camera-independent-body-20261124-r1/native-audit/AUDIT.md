# Independent Kinova3 native audit

**Problem.** The registered camera reader improved offline yaw inference on physical Kinova3 supports. A frozen policy receives a full yaw, gain and gripper correction, and must finish the task under that correction. The offline result alone leaves the native effect open.

**Hypothesis.** A improves fresh native task success over the pose-conditioned B. Shipped PR7 with the same public warp tests whether ordinary preprocessing can account for any advantage. Raw shipped and the known interface map establish the observed sensitivity range of this particular frozen motor. The native primary is A minus B over all 64 paired cases with source fit seed 41. A gain over raw alone would leave the architectural comparison unresolved.

**Test.** Baige R2 job `job-nn97ohkc3wf8` started `native02` from `d2b307bc4ca8c0cf723b3aaba31755d14a2f8303`. Its registered membership contains eight tasks, eight OOD interface cases per task, 16 task/layout reset identities and five independently executed arms per case. The independent reader accesses the original atomic pair tapes over SSH, with GPUs hidden and a single Baige development CPU core. Exact resource, checkpoint and input receipts are in the JSON evidence. Native01 (`job-70ie4pfjdgat`) exited 1 at the GPU probe before any episode, with `CUDA_VISIBLE_DEVICES` unset. All native results belong to native02. [Registration](../PLAN.md) · [Handoff](../HANDOFF.md) · [Native01 probe](../evidence/native01/gpu.json)

## First physical group

The first complete tape was case 04, libero_10/5, layout 26. All arms used the same pre-action physics, scene, public observation, prompt, noise bank, two motor camera inputs, state input and raw 10 × 7 proposal. The assigned +36° support supplied the four readers’ calibration. The seven named arm joints run from robot0_Actuator1 to robot0_Actuator7. Across 244 policy calls, all 1,211 emitted controls reconciled exactly to their original issued and physical action vectors. Rewards and ending flags agreed at each step. The true-map arm came from the physical interface. [Original 04 tape](04.json.gz) · [Independent numeric readback](first-group.json)

| Arm | Terminal control | Native success |
|---|---:|---:|
| A | 181 | 1 |
| B | 153 | 1 |
| Shipped + public warp | 175 | 1 |
| Raw shipped | 520 | 0 |
| True map | 182 | 1 |

**Update after the first group.** The first shared opening produced distinct executed corrections and a genuine raw timeout. This establishes a working treatment path on this case. It provides no directional A/B result for the full panel.

## What the action tapes establish

The native panel loads one frozen π₀.₅. For each assigned calibration trajectory the readback checks the original moved-view pixels, calibration matrix, public sensors and source tape digests. The four seed-41 triples must match the official score01 predictions exactly. The shipped+warp and A image hashes agree, as do raw shipped and B image hashes. Neither calibration RGB nor the homography is a motor input. Each query recreates the original agentview and wrist cameras. The wrapper checks the live motor inputs against both current camera observations and the policy state, then calls the frozen motor with that item and the recorded noise. Yaw rotates the post-opening local TCP observation. All three factors enter the inverse applied to the motor proposal. The simulator applies the true forward map to each issued action. A hook captures the numerical action delivered to the native environment and carries the native reward and ending flags back through the simulator reply. The independent reader recomputes each seven-channel action from the raw proposal and full factors, including clipping and terminal partial chunks. [Pinned source identities](provenance.json) · [Independent reader](verify_native02.py)

The physical opening test compares the complete live reset objects, environment records and opening proposals within a group. The CPU fixture independently anchors the released task, row, seed, pre-settle scene, horizon and robot model. Its post-settle physics fingerprint can differ from the distributed runtime. The tapes identify the seven Kinova joints individually, separately from the two gripper joints. The pinned environment validates the seven finite joint positions and velocities on every native step. The compact query tapes retain their opening hashes and the per-step seven-channel Cartesian/gripper commands. They do not serialize per-step joint angles, camera frames or full policy state, so later motor inputs are independently traceable to the in-process input check and their saved hashes.

Reproduction uses the original shared tapes, the SHA-checked registration and a pinned single CPU core. The development host's default OpenBLAS kernel first differed in the A issued XY readback on case 04, chunk 1. Selecting the Sandybridge kernel, previously qualified for this native runtime, reconciled the first group exactly under the original equality test. The independent reader writes its evidence to stdout.

```sh
ssh baige-dev 'taskset -c 37 env CUDA_VISIBLE_DEVICES= OPENBLAS_CORETYPE=Sandybridge OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python -B - final' < verify_native02.py
```
