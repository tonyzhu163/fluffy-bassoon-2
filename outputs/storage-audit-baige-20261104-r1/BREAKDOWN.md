# Remaining Baige storage after cleanup

**About 1201.4 GiB (79.5% of filesystem usage) is experiment output data.** Environments and setup are 107.3 GiB; project/shared caches are 26.8 GiB.

Read-only census on **2026-11-04, 17:07–17:36 Asia/Shanghai**. The full walk inspected **5,555,782 entries with zero read errors**; shared hardlinked inodes are counted once. Active Stage1 PPO was refreshed separately at the end because it continued writing. The filesystem reading then was **1510.33 GiB used / 537.67 GiB free**, total 2,048 GiB.

| Remaining category | Allocated GiB |
| --- | ---: |
| Current experiment outputs | 1105.38 |
| Original-repository experiment outputs | 96.01 |
| Model and input bundles | 128.79 |
| Environments and simulator setup | 107.28 |
| Code, worktrees and source dependencies | 42.35 |
| Project and shared caches | 26.82 |
| Other files, datasets, exports and tools | 4.85 |

The unique-file allocation sum is 1511.48 GiB, within 1.14 GiB of the filesystem reading. This is a moving census rather than an atomic snapshot; active writes and filesystem accounting explain why these readings need not match exactly.

## Largest individual output bundles

| Bundle | GiB | Interpretation |
| --- | ---: | --- |
| `robocasa-ppo-grpo-meta-stage1-20261104-r1` | 134.48 | Active PPO; keep hot. Refreshed after the main scan; continues growing. |
| `robocasa-flow-sde-critic-parity-20261104-r1` | 116.02 | Held baseline/fixed comparison. Two training arms are about 50 GiB each; preserve complete recovery state. |
| `robotwin-native-continuation-sac-20261030-r1` | 89.31 | Legacy RoboTwin SAC slices/corpus; candidate for a dependency-checked cold move. |
| `robocasa-flow-sde-10m-per-task-20261103-r1` | 85.98 | Five finished source lineages, about 17 GiB per task; retained final resumable states and supporting files. |
| `robotwin-final100-ad-export-20261031-r1` | 82.16 | Legacy private RoboTwin export/projection; existing code still refers to these paths. |
| `robocasa-flow-sde-checkpoint-selection-20261104-r1` | 81.29 | Human-stopped evaluation sweep; includes 51.78 GiB of large staging weight files. |
| `mikasa-vla16x16-s1001-baige-20261025-r3` | 42.91 | Legacy original-repository Mikasa run; 31.90 GiB in train-00010000. |
| `dsrl-pi05-task57-20261031-s0-r1` | 36.54 | Reported DSRL lineage; preserve cited checkpoints, sources and evaluation records. |
| `mikasa-vla16x16-s1002-baige-20261025-r3` | 32.01 | Legacy original-repository Mikasa run; 31.90 GiB in train-00010000. |
| `robocasa-opendrawer-flow-sde-20261102-r4` | 29.90 | Retained final18 teacher and raw source bundle; four-method PREPARE still reads physical chunk references. |
| `steering-kl-guard-ablation-20261103-r1` | 28.74 | KL diagnostic artifacts and raw evaluation evidence. |
| `robocasa-interface-screen-20261102-r3` | 28.23 | Finished interface-screen qualification/evaluation artifacts. |
| `robocasa-joint-steering-kv-scale50-20261103-r1` | 27.32 | Held KV source; recovery state and source data protected. |
| `robocasa-joint-steering-velocity-scale50-20261103-r1` | 26.20 | Held velocity source; recovery state and source data protected. |
| `dsrl-pi05-headroom-20261101-s0-r1` | 22.47 | Retained experiment artifacts; inspect dependencies before moving. |
| `mikasa-vla32x32-s1000-baige-20261025-r3` | 16.69 | Retained experiment artifacts; inspect dependencies before moving. |
| `robocasa-flow-sde-endpoint-test512-20261104-r1` | 16.27 | Retained experiment artifacts; inspect dependencies before moving. |
| `robotwin-grpo-learning-gate-20261028-r2` | 14.99 | Retained experiment artifacts; inspect dependencies before moving. |
| `robotwin-trained-continuation-20261031-r1` | 14.41 | Retained experiment artifacts; inspect dependencies before moving. |
| `frozen-pi05-endtoend-20261031-r1` | 12.08 | Retained experiment artifacts; inspect dependencies before moving. |

## What the smaller categories contain

**Model/input bundles:**

- `fluffy-bassoon-2/inputs/flora-rewarder-031299e4-20261101-r1`: 51.78 GiB.
- `fluffy-bassoon-2/inputs/robocasa365-public-pi05-c484-torch-20261101-r2`: 12.49 GiB.
- `fluffy-bassoon-2/inputs/robocasa365-public-pi05-c484-75000-20261101-r1`: 11.59 GiB.
- `fluffy-bassoon-2/inputs/pi0-libero-openpi-g1748456809412170-20261029-r1`: 11.19 GiB.

**Runtime / environment roots:**

- `fluffy-bassoon-2/setups`: 48.25 GiB.
- `fluffy-bassoon/envs`: 33.94 GiB.
- `fluffy-bassoon-2/envs`: 21.61 GiB.
- `envs`: 2.43 GiB.

**Caches / code:**

- `.cache/huggingface`: 14.43 GiB.
- `fluffy-bassoon-2/cache`: 11.21 GiB.
- `fluffy-bassoon-2/worktrees`: 27.10 GiB.
- `fluffy-bassoon-2/sources`: 8.52 GiB.
- `fluffy-bassoon-2/deps`: 4.93 GiB.

## Next cold-storage priorities

- **Legacy RoboTwin SAC/export bundles: 171.47 GiB.** Their path-bound source dependencies need a recovery mapping before a cold move.
- **Original-repository outputs: 96.01 GiB**, dominated by the two old Mikasa runs. Preserve their historical checkpoints and records in a recoverable archive.
- **Stopped checkpoint-sweep staging: 51.78 GiB in 8 large files.** These are the most obvious duplicated staging payloads to review against canonical checkpoint archives; the sweep's raw records and partial-stop evidence should remain retained. This is a subset of the 81.29 GiB bundle, not an additional category.

Active PPO, held critic/steering recoveries, shared package bases, and the explicitly referenced final18 raw source remain protected. This follow-up was read-only; no additional data was moved during the breakdown census.

## Evidence

[Compact measured breakdown](REMAINING-SUMMARY.json), [staging-file detail](STAGING-DETAIL.json), and [GNU du qualification](CENSUS-QUALIFICATION.json). The hardlink/symlink fixture matched `du -s -B1` exactly. Hardlinks are attributed to one stable path, preferring current model inputs and runtime setup, so separate `du` calls on individual directories can produce different per-folder totals.

The full 7.2 MB raw census remains at `/mowenjie/fluffy-bassoon-2/outputs/storage-audit-baige-20261104-r1/REMAINING-CENSUS.json`; its SHA256 is `bedf9a52a9525f10e13752a707e19209c5cd675c0d87d323ffc25cfde3333d4f`. The compact JSON records the separate active-run refresh and its checksum. Secret/chat roots expose only their top-level byte totals; no file contents were read by the census.
