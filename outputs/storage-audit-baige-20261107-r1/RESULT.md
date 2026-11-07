# Baige storage audit — November 7, 2026

**The shared 2 TiB PFS is 95.8% used: 1962.45 GiB used and 85.55 GiB free at 2026-11-07 13:29:23 Asia/Shanghai.** This is byte pressure: inodes are 56% used (5,521,054 / 10,000,384). No project payload was deleted, archived, deduplicated or moved by this audit; no GPU was allocated.

**Most existing space is experiment output, while most recent growth is new project/model/input material.** AnyViewPhys occupies **269.25 GiB**, including models 135.26, runs 77.42, environments 35.19 and package caches 18.10. A new materialized steering-source cache occupies **127.22 GiB** under `fluffy-bassoon-2/inputs/`. Those two items together are 396.47 GiB and explain most of the increase from the Nov 4 census. The main project's output directory has actually fallen from 1105.38 to 1046.86 GiB over that interval, while individual new RoboTwin runs grew and legacy payloads were removed.

## Complete allocation breakdown

The census walked **5,880,296 entries**, including hidden roots, on Linux from **12:56:43 to 13:22:48** on Nov 7, with **zero read errors**. **260,593 hardlinked regular-file inodes** were counted once. Allocation is `st_blocks × 512`, rather than logical file size; symlinks are not followed and other filesystems and the empty `.snapshots` directory contents are excluded. Secret/chat roots expose only aggregate totals in the published breakdown.

| Category | Allocated GiB | Share |
| --- | ---: | ---: |
| Experiment outputs | 1211.42 | 61.8% |
| Models and datasets | 405.47 | 20.7% |
| Environments and simulator setup | 183.26 | 9.3% |
| Caches | 95.19 | 4.9% |
| Code, worktrees, source dependencies and tools | 61.50 | 3.1% |
| Other files and metadata | 3.93 | 0.2% |

These refreshed allocation categories sum to **1960.77 GiB**. They are a moving census, not an atomic snapshot. The latest `df` reading differs by **1.68 GiB (0.09%)** while other jobs continue writing.

| Project root | Allocated GiB |
| --- | ---: |
| `fluffy-bassoon-2/` | 1463.24 |
| `anyviewphys/` | 269.25 |
| `fluffy-bassoon/` | 154.46 |
| `worldrank/` | 52.40 |

Remaining shared/top-level files account for 21.43 GiB. All project totals use the same cross-directory hardlink attribution, so adding independent `du` calls can overcount shared packages/models.

## Largest output bundles

| Output bundle | GiB | Interpretation |
| --- | ---: | --- |
| `fluffy-bassoon-2/outputs/robotwin-sidney-matched-baselines-20261106-r1` | 223.69 | Completed matched PPO/GRPO; final recovery checkpoints and downstream inputs remain protected. |
| `fluffy-bassoon-2/outputs/robocasa-flow-sde-critic-parity-20261104-r1` | 116.02 | Human-held critic comparison; preserve both complete recovery lineages. |
| `fluffy-bassoon-2/outputs/robocasa-opendrawer-ppo-ablation-3m-20261104-r1` | 100.21 | Ablation arms, staged source prefixes/suffixes and endpoint evaluation payloads. |
| `fluffy-bassoon-2/outputs/robocasa-eight-hour-babysit-20261104-r1` | 97.46 | 78.83 GiB in two source5 task bundles; remaining inputs and evaluations. |
| `fluffy-bassoon-2/outputs/robocasa-flow-sde-10m-per-task-20261103-r1` | 85.98 | Five retained teacher/source lineages, about 17.2 GiB each. |
| `fluffy-bassoon-2/outputs/robocasa-ppo-grpo-meta-stage1-20261104-r1` | 67.06 | Closed PPO/GRPO bundle; down from 134.48 GiB on Nov 4. |
| `fluffy-bassoon-2/outputs/robotwin-sidney-rlinf-ppo-20261105-r1` | 51.88 | Earlier train07/train03/train06 payloads; review alongside matched-lineage dependencies. |
| `fluffy-bassoon/outputs/iclr2027-rebuttal-20261106` | 35.21 | Active Mikasa initializer and raw-RGB dependencies; protect. |
| `fluffy-bassoon-2/outputs/robocasa-ppo-short-chunk-benchmark-20261104-r1` | 34.68 | Closed short-chunk benchmark payloads; retain reported boundary/recovery evidence. |
| `fluffy-bassoon-2/outputs/robocasa-opendrawer-flow-sde-20261102-r4` | 29.90 | Source final18/raw references were previously proven live; protect pending dependency release. |
| `anyviewphys/runs/avp-20261107-p4-v5-r1` | 29.30 | Recent AnyViewPhys generated/run data; protect current project work. |
| `fluffy-bassoon-2/outputs/steering-kl-guard-ablation-20261103-r1` | 28.74 | Retained experiment payload; ownership/dependencies need review before reclamation. |
| `fluffy-bassoon-2/outputs/robocasa-joint-steering-kv-scale50-20261103-r1` | 27.32 | Retained experiment payload; ownership/dependencies need review before reclamation. |
| `fluffy-bassoon-2/outputs/robocasa-ppo40-matched-dev64-20261104-r1` | 26.81 | Retained experiment payload; ownership/dependencies need review before reclamation. |
| `fluffy-bassoon-2/outputs/robocasa-joint-steering-velocity-scale50-20261103-r1` | 26.20 | Retained experiment payload; ownership/dependencies need review before reclamation. |

The 223.69 GiB matched RoboTwin bundle contains PPO **80.75 GiB**, GRPO **80.73 GiB**, qualification `qual03` **31.16 GiB**, `qual02` **21.74 GiB**, and `grad02` **9.30 GiB**. Each production arm contains **43.09 GiB** in `training-rollouts`, about **30.1 GiB** in native policy/checkpoint trees, **5.28 GiB** in initial states and **2.23 GiB** in sampling records. These are subsets of the bundle total.

## Inputs, models, environments and caches

- **127.22 GiB materialized source-member cache:** `fluffy-bassoon-2/inputs/steering-source-members-20261105-pfs01`. It contains three task bundles (Coffee 45.71, Drawer 42.14, Toaster 39.37 GiB). Its `CACHE.json` records **37,440** original compressed members with verified complete PFS readback, backed by the canonical BOS Flow10M corpus archives. That is an existing Nov 5 verification receipt; this audit did not reread the full cold archives. It is staged corpus data even though it lives under `inputs/`.
- **135.26 GiB AnyViewPhys models:** Hunyuan/WorldPlay **73.98**, Cosmos **32.53**, Wan **23.00**, native Cosmos audio **3.70**, plus DA3/SAM/tokenizer assets. Hunyuan holds a **31.88 GiB** full-precision file and a **15.94 GiB** BF16 representation. Different precision representations are not byte-identical duplicates. Recent/current AnyViewPhys work protects this namespace until its owner qualifies any reduction.
- **51.78 GiB Flora canonical weights**, plus **12.49 GiB** RoboCasa Torch and **11.59 GiB** original public checkpoint inputs. They are canonical runtime/input assets.
- **183.26 GiB environments/setup:** original-project environments **49.49**, current simulator setups **48.44**, AnyViewPhys environments **35.19**, WorldRank environments **24.71**, current-project environments **21.94**, plus shared bases. Original `policy` and `simulator` names have been restored since their earlier archival and now use **15.55 GiB** together; their names/age do not establish that they are unused.
- **95.19 GiB directory-classified caches:** current-project caches **51.43**, AnyViewPhys package caches **18.10**, root caches **15.26**, WorldRank caches **9.78**, plus small caches. The 51.43 GiB current cache includes **18.74 GiB** restored pilot material, **11.20 GiB** DSRL inputs, **8.71/7.24 GiB** Sidney model caches and **4.24 GiB** object assets. These directories mix model/input assets with disposable cache material.

Suffix classification finds **1133.19 GiB of model/Torch-serialized binary payloads** (`.pt`, `.pth`, `.safetensors`, `.distcp`, etc.) and **325.63 GiB of archives/packages**. The binary group also includes serialized trajectories and scene/state data, so it is not a pure model-weight total. Directory categories above are the reliable usage comparison. Same filename/size groups in the JSON are discovery leads only; they were not content-hashed as duplicates and have no claimed reclaim total.

## Ranked cleanup review

This is a review queue, **not a list of already safe deletions**. Keep configs, logs, numerical/result records, reported checkpoints and complete latest recovery state under the repository retention policy.

| Priority | Material | GiB | Required before reclaiming |
| --- | --- | ---: | --- |
| 1 | Materialized steering-source members | 127.22 | Check current/queued fit and readout consumers; reconcile canonical BOS receipts and restaging mapping. Evict only when no reader needs the hot material. |
| 2 | Matched RoboTwin `qual02` + `qual03` + `grad02` bulk | 62.20 | Review superseded qualification payloads; retain numerical proof/results and any recovery boundary still cited. Archive needed material before removing hot copies. |
| 3 | Restored pilot's `verified-cache.tar.zst` beside its extracted model | 11.78 | Reconcile its recovery source/checksum and ensure no restore process is using it. Preserve the usable model/cache dependency. |
| 4 | Matched PPO/GRPO training-rollout payloads | 86.18 | They are potential teacher/history inputs; qualify downstream references and an explicit cold restore mapping. Protect final actor/Adam/worker/RNG checkpoints. |
| 5 | Closed RoboCasa ablation / source / benchmark bundles | See table | Retain reported and held recovery checkpoints; cold-tier unused bulk at the lineage boundary. Do not delete by run age. |

The qualification and rollout rows are parts of the 223.69 GiB bundle and must not be added to it as separate usage. The held **116.02 GiB critic comparison**, held joint-steering lineages, canonical teacher/source checkpoints, active Mikasa initializers/dataset/raw RGB/HF cache, and current AnyViewPhys runtimes/models remain protected. No broad cache purge or environment removal is justified by this audit alone.

## Reconciliation and live dependencies

The raw full census totaled **1971.66 GiB**. A follow-up metadata refresh checked all **11** original-project output files at least 256 MiB found by the census. One qualification `model.safetensors` was replaced during the audit: allocation changed from **11,705,950,208 bytes** to **11,993,088 bytes**, with a different inode. Refreshing that one file reduces the census by **10.89 GiB** to 1960.77 GiB. It accounts for almost all of the original 11.51 GiB overcount against the post-scan filesystem reading; later live writes explain the current 1.68 GiB difference. The raw census/CSV remain unchanged and the refresh is a separate receipt.

At audit entry the volume had **45.34 GiB free (98% used)**. Free space ranged roughly 45–89 GiB during the metadata walk; all changes came from concurrent work. The Nov 6 07:15 watch recorded **446.71 GiB free**, so the entry reading had lost about **401 GiB** of headroom. This is a net filesystem change, not a measurement of gross bytes written by one run.

The 12:57 all-clusters/all-queues user task snapshot contained Mikasa `job-a6hnsocn83r6` and `job-mj4dojnuvnjh`; their resolved configs require `mikasa-rt82-factorial.../initializer-t1/t12`, the rebuttal source initializer/raw RGB, `memoryvla_five_vla_v1`, the shared HF cache and pinned worktree. Development-process references also showed current AnyViewPhys CPU work. A later existing AnyViewPhys task page showed `job-i8j58dtwb153` running, created at 13:16:58. That filtered page is additional dependency evidence, not an exhaustive latest scheduler snapshot.

## Other mounted storage and limits

- The development container overlay is separate from the 2 TiB PFS and was **42% used**, with about **1.91 TiB available**. Its `/root` uses **35.22 GiB**, including uv cache **18.64**, pip cache **4.94**, and old LIBERO staging **10.23 GiB**. `/tmp` uses **23.23 GiB**, dominated by checkpoint-retention staging **18.53 GiB**. Clearing these paths does not free `/mowenjie` PFS bytes.
- `lsof +L1` showed no deleted-open files in the development container at both checks. Scheduler containers are separate process namespaces; this is not a cluster-wide open-file claim.
- BOS is a separate archive store. Its FUSE `df` reports a synthetic **256 TiB / zero used**, which cannot establish real quota or billable usage. A full cold metadata listing exceeded its bounded time limit, so no complete BOS usage total is claimed.
- No separate `quota`, `lfs` or GPFS quota utility was available. The verified capacity here is the mounted filesystem's reported 2 TiB limit.
- Supplemental bounded `du`/model/cold-listing attempts that timed out were not used as complete totals. Their two orphaned observers were terminated only after exact process/start-time checks; all audit observers are now exited. No workload process was signalled.

## Evidence and verification

Three stable subtrees matched GNU `du -x -B1` **exactly**: AnyViewPhys models, the closed Stage 1 PPO/GRPO bundle, and the active rebuttal initializer/raw-RGB bundle. Category totals reconcile exactly. Full raw transfer passed independent remote/local SHA256 equality. No model computation or tests ran on macOS.

- [Compact measured summary](SUMMARY.json), [validation](VALIDATION.json), [latest capacity](capacity-latest.json).
- [Raw directory table](DIRECTORIES.csv) — 40.7 MB, overlapping parent/child prefixes; do not sum arbitrary rows. This is the original census before the one-file refresh.
- [Full raw census](census.json) — 115,446,413 bytes, SHA256 `b2afe7af459e99d1d2a8b7dd7fb5865d41897d9f5c6dc9e48f1d0a13244ac225`. The same immutable raw file remains at `/tmp/storage-audit-baige-20261107-r1/census.json` on the development machine; compact records are retained in Git, bulk raw files locally.
- [Volatile-file refresh](volatile-large-file-refresh.json), [staged-input metadata](staged-input-metadata.json), [initial scheduler snapshot](scheduler-snapshot.json), [additional task evidence](additional-live-dependency.json).
- [Audit scope](PLAN.md), [census implementation](census.py), [summary implementation](analyze.py), [transfer verification](transfer-verification.json), [bounded-attempt limitations](supplemental-limitations.json).
