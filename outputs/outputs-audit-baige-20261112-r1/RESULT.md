# Baige output audit, November 12, 2026

The audit covered both PFS output trees. It scanned **1,495,406 entries**
from **19:19:19 to 19:25:50 Shanghai** with
**0 read errors**. The duplicate cleanup reclaimed
**78.61 GiB**, including **69.90 GiB** from the output
trees. At **20:57:48 Shanghai**, PFS had **221.80 GiB free**.

| Output tree | GiB scanned | GiB after duplicate removal | Entries scanned |
| --- | ---: | ---: | ---: |
| `fluffy-bassoon/outputs` | 180.31 | 125.79 | 68,450 |
| `fluffy-bassoon-2/outputs` | 627.91 | 612.54 | 1,426,956 |

[Complete inventory of both output trees](OUTPUTS.csv) · [Machine-readable totals](SUMMARY.json)

Post-cleanup output totals subtract verified releases from the measured
snapshot. The live free-space reading also includes concurrent writes. The
scan counted allocated blocks, counted each linked inode once within these
two trees, and followed neither symlinks nor other mounts. The duplicate
search compared same-size files of at least 64 MiB, sampled candidates and
verified surviving matches with full SHA256. The directory inventory covers
files of all sizes.

## Bytes reclaimed

| Change | GiB |
| --- | ---: |
| Consolidated three identical MIKASA initializers across the completed flow runs | 32.71 |
| Consolidated the OpenDrawer step-zero model export and DCP shards | 15.03 |
| Consolidated identical AD, DIT and IC-IQL training arrays | 0.35 |
| Removed the staged copies used by two terminal loop diagnostics after rehashing the BOS originals | 21.80 |
| Consolidated Sidney's identical qualification weights outside the output trees | 8.71 |

The two qualified Sidney paths still resolve to the same verified weights.
The first attempt stopped at an expired scheduler check. A fresh check
cleared the consolidation. The links preserve the path names, content and
permissions. Linked paths share the kept file's modification time and
future in-place changes, so restore an independent file before editing one.
The loop diagnostic input pointers retain the original BOS object and
checksum. Both diagnostics had exited, and the current latency and T1
launch manifests point to other models. LIBERO paths and live checkpoints
were kept in place. Two matched CUDA library groups remain in separate
runtime trees, where the copies already have internal links.
[Sidney receipt](SIDNEY-DEDUP.json) · [Output receipts](OUTPUT-DEDUP.json) ·
[BOS duplicate receipts](BOS-DUPLICATES.json) · [Full hash matches](EXACT-DUPLICATES.json)

## What occupies the output trees

| File type | fluffy-bassoon GiB | fluffy-bassoon-2 GiB |
| --- | ---: | ---: |
| weights-checkpoints | 103.34 | 371.89 |
| archives-packages | 0.10 | 135.16 |
| videos | 0.37 | 61.26 |
| arrays-trajectories | 7.43 | 22.91 |
| other | 10.63 | 14.48 |
| logs-records-configs | 3.92 | 6.63 |

### fluffy-bassoon

| Output | GiB after duplicate removal |
| --- | ---: |
| `iclr2027-rebuttal-20261106` | 35.21 |
| `mikasa-recurrent-flow-20261107-r1` | 32.72 |
| `mikasa-matched-rt-carry-s1000-20261107-r10` | 11.04 |
| `mikasa-matched-rt-reset-s1000-20261107-r10` | 11.04 |
| `mikasa-rt82-factorial-s1000-20261107-r2` | 10.90 |
| `mikasa-carry-input-restore-20261109-r1` | 10.11 |
| `mikasa-flow-t1-20261109-r1` | 5.15 |
| `libero-dualrate-20261107` | 0.87 |
| `mikasa-flow-reset-s1001-20261107-r1` | 0.70 |
| `mikasa-flow-carry-s1000-20261107-r3` | 0.70 |
| `mikasa-flow-carry-s1002-20261107-r1` | 0.70 |
| `mikasa-flow-reset-s1000-20261107-r2` | 0.70 |

### fluffy-bassoon-2

| Output | GiB after duplicate removal |
| --- | ---: |
| `robocasa-eight-hour-babysit-20261104-r1` | 94.05 |
| `robocasa-flow-sde-10m-per-task-20261103-r1` | 85.98 |
| `robotwin-sidney-matched-baselines-20261106-r1` | 75.35 |
| `robocasa-flow-sde-critic-parity-20261104-r1` | 52.15 |
| `robocasa-ppo-grpo-meta-stage1-20261104-r1` | 51.10 |
| `robocasa-opendrawer-flow-sde-20261102-r4` | 29.90 |
| `steering-kl-guard-ablation-20261103-r1` | 28.74 |
| `robocasa-joint-steering-kv-scale50-20261103-r1` | 27.32 |
| `robocasa-joint-steering-velocity-scale50-20261103-r1` | 26.20 |
| `robocasa-three-task-meta-baseline-20261104-r2` | 15.73 |
| `frozen-pi05-endtoend-20261031-r1` | 12.08 |
| `robocasa-opendrawer-flow-sde-20261102-r3` | 7.10 |

## Which bytes need to stay on PFS

Keep the paths named by active and queued jobs and the complete recovery
state needed to resume a held run. Keep published checkpoints and raw
evidence recoverable in BOS with their checksums and per-case results.
Apply those requirements to the specific files used by a job or recovery,
including optimizer, sampler and RNG where required.

- The two closed OpenDrawer TEST512 panels have about **11.2 GiB** of raw
  evaluation payloads suitable for archive after the final reader check.
  Keep the per-case results. [Four-epoch panel](../robocasa-ppo4-grpo40-test512-20261105-r1/RESULT.md) · [Earlier baseline panel](../robocasa-eight-hour-babysit-20261104-r1/BASELINE-RESULT.md)
- The matched RoboTwin PPO and GRPO output has about **35.8 GiB** in
  intermediate step-one checkpoints and raw reset and sampling records.
  Keep the final recoveries and initializers readily available.
  [Matched run](../robotwin-sidney-matched-baselines-20261106-r1/RESULT.md)
- The two paused steering runs have about **50.8 GiB** in worker banks.
  Identify the bank and cursor state needed by each saved recovery before
  archiving the older banks. The KL diagnostic has its own hold.
  [KL hold and recovery scope](../steering-kl-guard-ablation-20261103-r1/PLAN.md)
- The **127.2 GiB** RoboCasa source-member staging cache lives outside
  these output trees. It has verified BOS backing. Its shared retention
  record still calls for checking the actual consumers and clearing the
  hold before release. [Cold backing](../storage-audit-baige-20261112-r1/EXISTING-COLD-SOURCES.json) · [Retention record](../robocasa-eight-hour-babysit-20261104-r1/SOURCE5-RESULT.md)
