# Baige storage audit and archival

Completed **2026-11-04 16:49 Asia/Shanghai**. **19 sources cold-archived**, covering **118.11 GiB** of source-directory allocation; **51.75 GiB** of duplicate Flora shards consolidated. PFS has **596.89 GiB free** of 2 TiB. No GPU allocation or model computation was used.

| Completed category | Sources | Source allocation, GiB |
| --- | ---: | ---: |
| Closed RoboTwin / LIBERO run folders | 6 | 22.81 |
| Stale standalone environments | 4 | 24.85 |
| Unrelated RoboMME / state-SFT / evaluation assets | 3 | 8.65 |
| Retired ManiSkill / LIBERO-carry / WAM / RoboTwin model assets and caches | 6 | 61.80 |
| Total | 19 | 118.11 |

The four retired environments are old `fluffy-bassoon/envs/policy`, `simulator`, `flora-bootstrap-20261028-r1`, and superseded `fluffy-bassoon-2/envs/dsrl-pi0-task57-20261101-prep-r3`. Their complete contents are recoverable. The 19 archives occupy **87.05 GiB on BOS** under `/mnt/bos/adgfyl/strong-plant-archives/storage-audit-baige-20261104-r1/`.

Both Flora model directories remain usable. All eleven shard pairs were independently SHA256-hashed before the legacy directory entries were atomically changed to share the canonical current inodes. No model contents or model paths changed. Keep these frozen shards immutable.

`uv cache clean` completed without `--force`: diagnostic cache 5,847 files / 266.1 MiB; DSRL download cache 69,616 files / 170.4 MiB. Installed environments and their hardlinked packages remain available.

## Verification and recovery

Every cold move passed exact archive-member coverage, GNU tar content/metadata comparison against the source, stable source rechecks, complete BOS SHA256 readback, and local process-dependency checks before reclamation. Final reconciliation verified all 19 recovery pointers, archive sizes, sidecars, receipts and matching PFS/BOS inventory hashes. Both archive workflows passed Linux CPU restoration fixtures; the parallel workflow preserved hardlinks, symlinks and literal filenames and detected same-size corruption with unchanged mtime.

Each original source has an adjacent `<name>.bos-archive.json` recovery pointer. Use its exact command and checksum. **The `workspace-r2` format requires `tar --ignore-zeros`** to restore every tar stream. See [restoration instructions](RESTORE.md) and the full [manifest](MANIFEST.json), which records every object, digest, original path and recovery command. The deployed audit scripts match the local [SHA256 record](SCRIPT-HASHES.json). Raw inventories and command logs remain on PFS/BOS; compact evidence is mirrored here.

## Retained dependencies

- Current RoboCasa setup, assets, caches, weights and source pins; current Flora and LIBERO weights and package environments; seven protected interpreter paths still resolve to their original targets.
- Shared ManiSkill site-packages and all Python bases. MetaWorld, RoboTwin and Flora/LIBERO `.pth` files still load that older ManiSkill environment, so its age is insufficient reason to remove it.
- Both held steering event005 recovery lineages, held critic-comparison state, source checkpoints/corpora, and checkpoint-selection inputs.
- Final18 teacher plus the old Flow-SDE raw bundle: following `SOURCE-CHUNKS.json` found 3,072 retained references into r4/environment, so those three bulk-directory candidates were withdrawn before reclamation. Their unused scratch archive was removed.
- Source checkouts and user changes. Historical scripts for cold assets require their recorded restore operation before reuse.

Stage1 PPO `job-bylh99ylx9ie` advanced to **16/40** during cleanup, with its storage checkpoint-exit flag false. All audit archive/cache workers have finished. No training job was stopped or changed.

## Space accounting and audit limits

At bulk-archival start, free PFS was 479.47 GiB; the final reading is 596.89 GiB, a **net increase of 117.41 GiB while PPO continued writing**. Source-directory totals can count hardlinks shared with retained environments and must not be added as an exact attribution of physical free-space gain. The entry check was earlier than the recorded bulk-start baseline.

Large metadata censuses were stopped once candidates were identified; scans excluding journals or unfinished scans are partial, not a complete atomic directory census. Filesystem readings and finalized per-source archive inventories are authoritative for their timestamps. Earlier VIA/action-critic/WAM projects were already archived on Nov3; eight top-level pointers were checked for source absence and object presence and are **excluded from this run's totals**.

The Mac checkout was only about 394 MiB with about 310 GiB disk space free; no unrelated local files were removed. Routing and credentials remain outside Git.
