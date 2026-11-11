# Baige storage audit and archival, November 11, 2026

Archived **112.02 GiB of PFS allocation** to private BOS and verified all
17 restored objects before releasing their hot copies. The 2 TiB PFS ended at
**213.20 GiB free** and **1,834.80 GiB used**, up from 101.67 GiB free at
census start. Inodes were 55.3% used. No training jobs were stopped
or modified. The development container's separate overlay had
926 GiB free.

## What moved

| Payload | Files | PFS allocation released |
| --- | ---: | ---: |
| RoboTwin Sidney PPO `ppo02/training-rollouts` | 8 | 43.09 GiB |
| RoboTwin Sidney GRPO `grpo01/training-rollouts` | 8 | 43.09 GiB |
| MIKASA failed r3 qualification step-2 tarball | 1 | 25.85 GiB |

Each BOS copy and staged restore matched the SHA256 in its original receipt.
The original MIKASA receipt also records verification of each tar member.
All 180 PPO and GRPO checkpoint files
passed the before/after identity check. The MIKASA successor used a fresh
initializer. Its production checkpoints, the failed run's remaining
diagnostics, and the original per-file receipts are still on PFS.

BOS objects are under
`/mnt/bos/adgfyl/strong-plant-archives/storage-audit-baige-20261111-r1/`.
The original directories contain per-file restore pointers. The restore
command passed a Linux PFS check for data, mode and overwrite protection.
[Restore instructions](RESTORE.md) · [RoboTwin receipts](RELEASE.json) ·
[MIKASA receipt](MIKASA-RELEASE.json) · [Checkpoint check](RELEASE-CHECK.json).

## What still uses PFS

| Category | GiB | Share of measured allocation |
| --- | ---: | ---: |
| Experiment outputs | 1,264.49 | 68.9% |
| Models and datasets | 281.11 | 15.3% |
| Environments and simulator setup | 148.25 | 8.1% |
| Caches | 70.91 | 3.9% |
| Code, worktrees and dependencies | 66.60 | 3.6% |
| Other files and metadata | 4.69 | 0.3% |

The split project's outputs account for **757.45 GiB**.
The original project's outputs account for **489.79 GiB**,
up **409.02 GiB** since the Nov 7 census, with MIKASA
checkpoints and qualification runs leading that growth. The largest staged
input is the **127.22 GiB** steering corpus under
`fluffy-bassoon-2/inputs/steering-source-members-20261105-pfs01`.
Flora weights account for another **51.78 GiB**. AnyViewPhys remains in its
Nov 8 archive, with its PFS restore stub verified.
[AnyViewPhys restore guide](../anyviewphys-cold-archive-20261108-r4/RESTORE.md).

### Largest remaining output runs

| Path under `/mowenjie` | GiB |
| --- | ---: |
| `fluffy-bassoon-2/outputs/robocasa-flow-sde-critic-parity-20261104-r1` | 116.02 |
| `fluffy-bassoon-2/outputs/robocasa-eight-hour-babysit-20261104-r1` | 97.46 |
| `fluffy-bassoon-2/outputs/robocasa-flow-sde-10m-per-task-20261103-r1` | 85.98 |
| `fluffy-bassoon-2/outputs/robotwin-sidney-matched-baselines-20261106-r1` | 75.35 |
| `fluffy-bassoon/outputs/mikasa-fullft-reset-s1000-new-20261107-r4` | 74.74 |
| `fluffy-bassoon/outputs/mikasa-fullft-reset-s1000-new-20261107-r3` | 74.69 |
| `fluffy-bassoon-2/outputs/robocasa-ppo-grpo-meta-stage1-20261104-r1` | 67.06 |
| `fluffy-bassoon/outputs/mikasa-vla16x16-s999-baige-20261107-r2` | 53.81 |
| `fluffy-bassoon/outputs/mikasa-vla16x16-s1004-baige-20261109-r1` | 53.81 |
| `fluffy-bassoon-2/outputs/robotwin-sidney-rlinf-ppo-20261105-r1` | 51.88 |
| `fluffy-bassoon/outputs/mikasa-vla16x16-s1003-baige-20261109-r1` | 42.91 |
| `fluffy-bassoon/outputs/mikasa-flow-t1-20261109-r1` | 37.87 |

## Measurement

The metadata census walked **5,766,024 entries** from 12:57:22
to 13:24:26 Shanghai time with **0 read errors**. It used
allocated blocks and counted 183,775 shared
inodes once. It did not follow symlinks or traverse other mounts or
`.snapshots`. The census saw all 17 files released
during the walk, so the table subtracts their exact allocations.
The resulting 1,836.06 GiB differs from the final filesystem reading
by 1.26 GiB (0.07%) on the live volume.
The raw census is at `/tmp/storage-audit-baige-20261111-r1/census.json` on
the development container and compressed in the local run folder.
[Full category and directory totals](SUMMARY.json) · [Filesystem reading](CAPACITY-FINAL.json) ·
[Archived file verification](RELEASE-CHECK.json).
