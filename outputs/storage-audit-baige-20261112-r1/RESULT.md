# Baige storage audit, November 12, 2026

At **18:47:04 Shanghai**, the 2 TiB PFS had
**149.96 GiB free** and **1,898.04 GiB used**
(92.7%). That is **533.54 GiB**
less free space than the November 11 cleanup closeout. Inode use was
61.7%. This was a read-only audit of PFS payloads.

## Main additions since cleanup

| Path under `/mowenjie` | GiB now | Change in GiB |
| --- | ---: | ---: |
| `fluffy-bassoon/cache/libero-original-27e6ec3c` | 177.99 | +177.99 |
| `fluffy-bassoon/scratch/history-T12-reset-s1000-r4` | 92.69 | +92.69 |
| `fluffy-bassoon/scratch/history-T1-reset-s1000-r4` | 92.69 | +92.69 |
| `fluffy-bassoon/inputs/memoryvla-releases-20261112-r1` | 65.21 | +65.21 |
| `fluffy-bassoon/outputs/mikasa-rt-loop-diagnosis-20261112-r1` | 21.81 | +21.81 |
| `fluffy-bassoon-2/setups/libero-gen-history-pilot-20261111-r1` | 17.84 | +17.84 |
| `fluffy-bassoon-2/setups/libero-procedural-pi05-20261111-r1` | 16.78 | +16.78 |
| `fluffy-bassoon/inputs/openvla-oft-libero-spatial-6d0231af-20261112-r1` | 14.39 | +14.39 |
| `fluffy-bassoon/inputs/hamlet-libero-6793e2a-20261112-r1` | 7.57 | +7.57 |
| `fluffy-bassoon/envs/memoryvla-flops-20261112-r1` | 6.83 | +6.83 |

The two history-reset scratch directories carry 92.69 GiB each, primarily
four roughly 20.39 GiB training-state files and a 10.91 GiB policy copy per
lineage. The six pre-20k training states total about 122 GiB and can be
reviewed for cold storage alongside the latest recovery requirements.
The LIBERO cache stages 273,465 frames of final-layer attention K/V.
Its example 320.35 MiB shard spends 320 MiB on BF16 keys and values.
[KV tensor header and cache record](KV-EXAMPLE.json).

## Current usage by category

| Category | GiB | Share |
| --- | ---: | ---: |
| Experiment outputs and scratch | 1,010.82 | 53.3% |
| Models and datasets | 370.77 | 19.6% |
| Caches | 250.02 | 13.2% |
| Environments and simulator setup | 191.41 | 10.1% |
| Code, worktrees and dependencies | 68.42 | 3.6% |
| Other files and metadata | 4.77 | 0.3% |

### Largest storage components

| Path under `/mowenjie` | GiB |
| --- | ---: |
| `fluffy-bassoon-2/outputs` | 627.90 |
| `fluffy-bassoon-2/inputs` | 236.21 |
| `fluffy-bassoon/scratch` | 185.37 |
| `fluffy-bassoon/outputs` | 180.29 |
| `fluffy-bassoon/cache` | 179.39 |
| `fluffy-bassoon/inputs` | 131.68 |
| `fluffy-bassoon-2/setups` | 84.81 |
| `fluffy-bassoon/envs` | 56.34 |
| `fluffy-bassoon-2/cache` | 45.55 |
| `fluffy-bassoon-2/sources` | 31.23 |
| `fluffy-bassoon-2/worktrees` | 27.10 |
| `worldrank/envs` | 24.71 |

### Largest experiment output directories

| Path under `/mowenjie` | GiB |
| --- | ---: |
| `fluffy-bassoon-2/outputs/robocasa-eight-hour-babysit-20261104-r1` | 94.05 |
| `fluffy-bassoon-2/outputs/robocasa-flow-sde-10m-per-task-20261103-r1` | 85.98 |
| `fluffy-bassoon-2/outputs/robotwin-sidney-matched-baselines-20261106-r1` | 75.35 |
| `fluffy-bassoon-2/outputs/robocasa-flow-sde-critic-parity-20261104-r1` | 52.15 |
| `fluffy-bassoon-2/outputs/robocasa-ppo-grpo-meta-stage1-20261104-r1` | 51.10 |
| `fluffy-bassoon/outputs/mikasa-flow-t1-20261109-r1` | 37.87 |
| `fluffy-bassoon/outputs/iclr2027-rebuttal-20261106` | 35.21 |
| `fluffy-bassoon/outputs/mikasa-recurrent-flow-20261107-r1` | 32.72 |

## RGB and offline learner retention

The archived matched Sidney PPO and GRPO pair saved one 5.39 GiB actor file
per rank and iteration. Each rank recorded 2,048 decision rows, each with
one base camera and two wrist views at 640 × 480 in uint8 RGB. The three
uncompressed camera tensors occupied 5,400 MiB, or about 98% of a shard.
The 16 shards totaled 86.2 GiB, with separate environment traces and
sampler records. Zstd level 1 reduced one measured shard from
5.39 GiB to 3.35 GiB.
[Rollout example](RGB-EXAMPLE.json) · [Archived pair](../storage-audit-baige-20261111-r1/RELEASE.json).

| Learner | Data to keep for an active fit |
| --- | --- |
| AD | Chronological observation features, actual executed actions or compatible steering targets, rewards, episode boundaries and source-policy identity. |
| DPT | The same recorded history and separately generated teacher targets for each matched query, with teacher checkpoint, preprocessing and seed provenance. |
| DIT and IC-IQL | Actual executed actions, current and next observation features, rewards, complete chronological transitions, action masks and termination/truncation semantics. |

Use one verified lossless RGB source per unique recorded observation on BOS.
Keep the prepared feature/transition/label shards used by live fits on PFS.
Stage original images for new teacher queries, feature extraction and replay
verification. Full-KV models need their exact K/V representation staged for
fits that read it. Keep the capture manifest with frame identity, view,
preprocessing and hash so it is possible to rederive a feature or label.
[Offline methods](../../docs/METHODS.md) ·
[Native capture fields](../../src/fluffy_bassoon/rlinf/native_ppo_data.py).

For future runs, the RoboCasa capture has an `actor_batch_drop` option
to omit RGB held in the environment trace. The current Cabinet teacher
explicitly captures the full actor batch. A subsequent run can omit exact
duplicates after checking the reconstruction path with its resolved config.
[RoboCasa capture configuration](../../benchmarks/robocasa/configs/robocasa365_pi05_flow_sde_10m.yaml) ·
[Current teacher configuration](../robocasa-six-source-decision-20261112-r1/teacher.sh).

Two existing BOS-backed staging caches account for roughly 305 GiB together.
The 178 GiB LIBERO KV cache matches its cold manifest and has an earlier
complete checksum verification. The 127 GiB RoboCasa steering source-member
cache has a complete verification receipt against its source archives.
Reclaiming these staged copies depends on confirming the current consumers
and checking recovery at release time. [Existing cold copies](EXISTING-COLD-SOURCES.json).

## Measurement

The metadata census scanned **6,502,360 entries** between
18:13:30 and 18:46:36 Shanghai with
**0 read errors**. It measured allocated blocks and counted
324,302 multiply linked inodes once.
It stayed on PFS and excluded the contents of `.snapshots`.
The totals also include a targeted size refresh for two model downloads
that started during the scan. [Model-input refresh](LATE-INPUTS.json).
The adjusted census total of 1,896.19 GiB differs by
1.84 GiB from the final filesystem reading.
The raw scan is in the local run folder and at
`/tmp/storage-audit-baige-20261112-r1/census.json` on the development container.
[Totals and largest directories](SUMMARY.json) · [Capacity](CAPACITY-FINAL.json).
