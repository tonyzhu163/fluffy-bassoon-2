# Original records and evaluator provenance

The paper's UR5e × π₀.₅ figure reads the seed-41 campaign summary. Its single-attempt column is `ours_equiv_v2_online`, its calibrated retry is `ours_equiv_v2_passive`, and the immediate exact inverse is `privileged_oracle`. All 96 published rows agree field for field with the original shard bytes at the locations in the provenance manifest. Seven shards came from Baige job `job-8yz6v8qvd9dq` on revision `437da7f677632e9495c813aaa52071e207ba834e`. Goal 6 came from the complete replacement shard in `job-mjhnlm7vqe31` on `ff2ddf0ab03e6fbe243979556d8d8f2ebb9bc29c`. The streaming function, JIT step, motor, simulator worker, transform and reader implementation agree across those revisions. [Paper figure](../../papers/icml-xemb-sysid/plot_icml_figures.py) · [Original shard paths and digests](../xemb-live-wave-20261121-r1/evidence/pi05-provenance.json) · [Published per-query records](../xemb-live-wave-20261121-r1/evidence/pi05-rows.jsonl)

| Historical arm | Native successes | What was available at the first scored control |
| --- | ---: | --- |
| Single-attempt streaming | 20/96 | Identity |
| Calibrated retry | 76/96 | Three measured five-control chunks from layout 11 |
| Immediate exact inverse | 82/96 | True command map |

All 20 streaming successes were also retry successes. The retry rescued 56 queries and lost zero. Its historical gain was +58.33 percentage points, with an exact whole-task bootstrap 95% interval of +41.67 to +75.00. The eight task totals for retry minus streaming were 5, 2, 9, 7, 12, 9, 5 and 7 out of 12. Both the opening information and the onset change in this historical contrast. [Independent raw-shard counts and interval](evidence/archive-readback.json)

## Timing and observations

The original UR5e policy draws a (10, 32) float32 noise tensor per requery and produces a (10, 7) proposal. It executes at most **five** controls and requeries at controls 1, 6, 11, 16, and so on, at 20 Hz. Control 11 begins 0.50 seconds after the first command. The single-attempt estimator starts at (0°, 1, +1). It accumulates its first observation after controls 1–5. After controls 6–10, the original `chunk_idx >= 1` guard first installs `est.estimate()`. The policy input for control 11 already uses the estimated yaw and the same estimate corrects the third chunk's issued commands. Eligible updates continue every five controls through control 120. The estimate from the 24th chunk first serves control 121 and is then held. The original row records only the rounded final estimate. The first applied estimate is scheduled at 11 whenever the episode reaches it. A particular issued command can remain numerically equal to its uncorrected value. The new trace records both the first nonidentity estimate and the first numerically changed issued control.

The native worker applies the hidden command transform to the bounded action and returns the public gripper state, camera observations and end-effector displacement in the uncalibrated frame. At frame zero the motor sees the original observation. On later queries the current estimated yaw rotates the local displacement about the reset anchor back into the motor’s frame. The original π₀.₅ image and quaternion conversions remain in the pinned motor. The correction inversely rotates planar commands, divides all three translation components by gain, leaves wrist actions as issued by the motor and applies gripper polarity before the worker enforces the hidden map. The reader sees measured local motion and the issued command. It has no access to the true transform in the streaming intervention.

## Query identity and reuse

The schedule is eight task identities × twelve command settings in BASE scenes with one initial-state index, 13. The query seed is 41,800,000 plus 10,000 × the sorted suite index, 100 × the task ID, and 13. NumPy's original generator draws standard normals with seed = query seed + 817,117 and casts the full bank to float32 in the pinned runtime. Each arm consumes draw j exactly once for query j. The prepare step validates every bank digest and the runtime verifies every actual tensor supplied to inference.

| Suite | Task IDs | Query seeds | Horizon |
| --- | --- | --- | ---: |
| libero_10 | 5, 8 | 41800513, 41800813 | 520 |
| libero_goal | 3, 6 | 41810313, 41810613 | 300 |
| libero_object | 1, 8 | 41820113, 41820813 | 280 |
| libero_spatial | 0, 8 | 41830013, 41830813 | 220 |

The original per-arm results contain success, control count and either a fixed rounded map or the streaming final rounded map and duration. They omit opening proposals, issued prefixes and post-prefix observations. The recovery log retains one full Goal 6 reset; its fields agree with the native FK and matched-flow reset references. All eight reference resets agree between those two jobs. The GPU launcher checks the historical physics, scene, public frame, motor and reader frame zero, prompt, noise, renderer vendor and action schema. Different qualified GPU models can report different renderer names. The fresh arms must agree on the complete reset including that name. CPU software rendering has its own paired physics and image receipts. No archived outcome enters the intervention contrasts.

## Pinned inputs

- The dataset schedule comes from the native Goal 6 evaluator at `ff2ddf0a` and the historical FK panel SHA-256 `51a4575ff3105b0e7005befb598f94d8bf69ed0c068324c8833a92b11cfd05d0`. The archived retry tapes have ordered SHA-256 `8729fa68343e2a7a0cc6471ab25e7c707555a866e682a3dc999079a7b9dc3554`. Those tapes are read at prepare time to qualify the historical reference only.
- The original seed-41 checkpoint is `/mowenjie/fluffy-bassoon-2/outputs/xemb-icml-campaign-20261121-r1/gpu03v2/stage1/checkpoints/equiv_v2_dis_dz-passive_attempt1-seed41.msgpack`, SHA-256 `c12bace5cb391499d6abd7cc5767f94b1519de7d77940efb0cfcb71ca0f686a4`.
- Frozen π₀.₅ uses OpenPI `215abfb217dbac7d5f1273282331b9b1866c0479` and checkpoint `/mowenjie/fluffy-bassoon/inputs/pi05-libero-openpi-20261027-r1`. The launcher verifies the manifest and all 16 files (12,439,085,481 bytes), with manifest SHA-256 `ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170`.
- The simulator release is `e5500c690c1498eca1b7050350654964190cdf70`. The launcher verifies its clean checkout, pinned config and prepared manifest covering 88 simulator source files. Frozen helper code for the matched-flow/FK input checks, native simulator startup, the GPU probe and the exact task bootstrap comes from `d568535b758f2b97b382f9b3e280376e769b5942` on the existing shared mount. [Input verification](panel.py) · [Job environment](launch.sh)
