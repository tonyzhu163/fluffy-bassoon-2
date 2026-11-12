# Baige BOS storage audit

The mounted BOS namespace contained **5.55 TiB across 236,099 visible objects and 33 archive prefixes** on November 13, 2026 Shanghai time. The inventory includes this batch's new 40.54 GiB of archive payloads. The scan reported zero listing or stat errors. An independent inventory agreed on all shared prefixes except the active KV comparison, which added eight objects between the scans. The mount exposes current objects. Bucket billing, previous object versions and unfinished multipart uploads were unavailable through the current access.

| Largest archive prefixes | GiB | Contents and dependencies |
| --- | ---: | --- |
| Five-task RoboCasa Flow-SDE PPO | 1,446.12 | 75 full archived checkpoint snapshots and five native corpora with audit files |
| RoboTwin restoration | 497.58 | 87.50 GiB of trace inputs and 167.55 GiB of matched Motus behavior PPO/GRPO outputs, among other experiments |
| November 11 output retention | 377.84 | Restores prior PFS releases |
| LIBERO history | 341.23 | History-training and resume artifacts plus the verified KV cache |
| November 4 experiment bundles | 295.89 | Recovery bundles from earlier PFS releases |
| November 5 PPO checkpoint retention | 260.37 | 226.60 GiB under attempt-02 plus archived rollout, environment and video payloads |
| AnyViewPhys archive parts | 232.51 | The canonical r4 restore reads all eight data parts from r3 |
| November 3 RL checkpoint reclaim | 218.52 | Recovery objects for the November 3 PFS release |

[All 33 prefixes](BOS-SUMMARY.json) · [Five-task RoboCasa result](../robocasa-flow-sde-10m-per-task-20261103-r1/RESULT.md) · [RoboTwin PPO and GRPO result](../robotwin-restoration-20261106-r1/PPO-GRPO-AUDIT-RESULT.md) · [AnyViewPhys restore contract](../anyviewphys-cold-archive-20261108-r4/RESTORE.md)

## Deletion candidates

The duplicate scan sampled nine offsets in 1,386 same-size candidate objects of at least 32 MiB, then read all bytes for the 18 matching groups. The LIBERO KV cache shards and the AnyViewPhys tar slices are indexed members of larger collections, and 541 same-size objects in those sets were excluded from comparison.

- **1.80 GiB with an equivalent recovery copy.** Delete `robocasa-flow-sde-10m-20261103-r1/corpora/qualification/OpenDrawer-q1-r2-r1.tar` and retain the r2 tar and manifest. Both tar objects still match their original SHA-256 receipts. The original restore probes verified the same 332 source files, and the current sidecar manifests are byte-identical. The tar hashes differ, and each contains its own named embedded manifest. The execution record identifies r1 as the previous copy-only qualification and r2 as the current qualification with the extra ENV and actor checks. The only exact r1 object references in the two checked repositories are its historical qualification records. [R1 receipt](../robocasa-flow-sde-10m-per-task-20261103-r1/CORPUS-ARCHIVE-QUALIFICATION-R1.json) · [R2 receipt](../robocasa-flow-sde-10m-per-task-20261103-r1/CORPUS-ARCHIVE-QUALIFICATION-R2.json)
- **1.22 GiB in six byte-identical copies.** Remove the three duplicate tars in `qualification-packing-concurrent04` after recording the surviving paths in its qualification receipt. Remove the three restored `policy.pt` copies from the completed RoboTwin CPU restore probes and keep each original. All six pairs passed full-object SHA-256 comparison. [Exact paths and SHA-256 results](BOS-DUPLICATE-PROBE.json)

| Remove | Keep | GiB |
| --- | --- | ---: |
| `qualification-packing-concurrent04/velocity_b8.tar` | `qualification-packing03/velocity-OpenDrawer-full-bank000.tar` | 0.715 |
| `qualification-packing-concurrent04/velocity_ood.tar` | `packed-waves/qualification-fitted-oodfull-pfs04/velocity_bridge/full/CloseFridge/bank-000.tar` | 0.137 |
| `qualification-packing-concurrent04/kv_ood.tar` | `packed-waves/qualification-fitted-oodfull-pfs04/kv_bridge/full/CloseFridge/bank-000.tar` | 0.135 |
| `cpuhistoryimitation01/restored/policy.pt` | `cpuhistoryimitation01/original/policy.pt` | 0.077 |
| `cpudecodedfit01/ic_iql-restore/policy.pt` | `cpudecodedfit01/ic_iql/policy.pt` | 0.077 |
| `cpudecodedfit02/ic_iql-restore/policy.pt` | `cpudecodedfit02/ic_iql/policy.pt` | 0.077 |

The first two cleanup items total **3.02 GiB**. The other candidates involve giving up raw recovery.

- **2.87 GiB of raw qualification data.** The `native-initial-ID-B8-qualification-cold-01` and `native-final-ID-qualification-cold-01` prefixes hold 2.27 GiB and 0.61 GiB. Their archived compressed traces, tensors and videos account for 3,083,538,373 bytes. Keep the archive manifests, audit records, results, logs and case selection records. Releasing the raw objects would remove the ability to replay these qualification episodes. The exact BOS prefixes appear in their archival receipts, and the compact B8 audit and concurrent-archive qualification are tracked. [B8 receipt](../robocasa-eight-hour-babysit-20261104-r1/ARCHIVE-B8-RECEIPT.json) · [Concurrent qualification receipt](../robocasa-eight-hour-babysit-20261104-r1/ARCHIVE-CONCURRENT-RECEIPT.json)
- **8.55 GiB from the failed OpenCabinet attempt01, subject to retention of its failure evidence.** The attempt exited before its first PPO update. Two step-zero DCP shards account for 9,176,523,028 of the 9,185,372,160 tar bytes. The retry has its own checkpoint lineage. This tar is the verified recovery object for the failed attempt's exact initializer. Preserve the diagnostic log, resolved config, accounting and manifests separately before discarding that initializer. [Failure and recovery receipt](../robocasa-six-source-decision-20261112-r1/EXECUTION.json)
- The 14 visible `.tmp` objects total 1.47 MiB. Empty older LIBERO prefixes and the two 30-byte AnyViewPhys markers offer negligible savings.

The largest potential saving requires checkpoint repacking. The five-task campaign holds about 810 GiB in 65 full production snapshots at steps 2 through 120. Its completed report strictly loaded the archived model for every checkpoint in the historical likelihood comparison. A smaller retention format could preserve evaluable model weights, receipts and selected full recovery boundaries if it reproduces the existing model-load and likelihood checks. The raw reference corpora are separate. The audit leaves all BOS objects in place.
