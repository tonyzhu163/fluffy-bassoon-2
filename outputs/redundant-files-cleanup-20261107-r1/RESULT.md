# Redundant-file cleanup — verified complete

**Reclaimed another 17.21494 GiB (18,484,406,272 allocated bytes) by releasing exactly 323 approved files.** All original bytes remain recoverable from verified BOS copies. Final reconciliation passed at **2026-11-07 15:22:28 Shanghai**. Together with the earlier outputs archival, this chat has reclaimed **224.91539 GiB**.

| Released material | Files | Reclaimed GiB | Recovery |
| --- | ---: | ---: | --- |
| Duplicate `verified-cache.tar.zst` | 1 | 11.78482 | Canonical BOS archive independently rehashed to the original published SHA256. |
| Extracted object download `objects.zip` | 1 | 3.48108 | New exact cold recovery archive; extracted simulator families retained. |
| Truncated earlier `model.safetensors` download | 1 | 1.10764 | New exact cold recovery archive, including its original incomplete bytes. Valid extracted motor retained. |
| Retained-Git deployment bundles | 320 | 0.84140 | New exact cold recovery archive plus retained commits and verified object closure. |

Cold recovery root: `/mnt/bos/adgfyl/strong-plant-archives/redundant-files-cleanup-20261107-r1/`. The new `selected-files.tar.zst` contains 322 files, totals **5.24144 GiB**, and has SHA256 `4c7a0c58d162f6c0f234c78af5b00b4e02315c5781b97b4dfe744be9cae3da31`. The already-cold duplicate cache is recovered from its original canonical object named in PLAN.json. [Exact source inventory](SOURCE-INVENTORY.json), [all released paths](RELEASED.json), [restore instructions](RESTORE.md).

## Verification and preserved data

The Linux fixture passed complete archive restoration, hardlink counting, unselected-file preservation and same-size/same-mtime changed-file rejection. Runtime source **`ba9eec509c1e887705cd815439d7f34250ca62e9`**, SHA256 **`e276281d63750c98a06a1f6817d701fcbd59d821745d9dcb9bcca026294b05fd`**, was deployed through a verified Git bundle; the remote repository HEAD stayed unchanged.

Before releasing any original: hash every selected file; rehash the existing BOS duplicate; reconcile the ZIP with its installed manifest; verify the earlier model is truncated; check all 320 bundle tips remain reachable and their complete object closure exists; compare the new archive against sources; independently hash all cold bytes; restore into empty staging and verify exact member paths, byte hashes, modes/owners/mtimes and hardlink topology. Each source was checked again for unchanged metadata, SHA256 and scheduler/development command/FD/mmap readers before unlink. Only exact approved regular-file paths were removed; original directories and compact records remain.

**All 3,303 protected entries retained their original stamps, and every protected regular file was rehashed unchanged after release.** This includes the valid extracted motor, restored simulator families, installed manifest, compact failed-download records and all unselected deployment bundles. The original 12 bundles without retained tips were preserved. The 127 GiB source corpus, 86 GiB scientific training rollouts, 38.77 GiB predecessor recovery state, held runs, active outputs, shared models and environments were outside this batch.

Shared PFS free space was **266.00 GiB** at cleanup start and **261.10 GiB** at final verification. These snapshots include concurrent work; exact reclaimed allocation above counts only this batch and only the final hardlink. No GPU was allocated or workload stopped. All own staging was removed and the worker exited.

[Final reconciliation](FINAL-VERIFICATION.json) · [Execution result](RESULT.json) · [Recovery receipt](RECOVERY.json) · [Git closure](GIT-CLOSURE.json) · [Qualification](QUALIFICATION.json) · [Deployment](DEPLOYMENT.json) · [Verifier deployment](VERIFIER-DEPLOYMENT.json). The protected inventory is retained byte-for-byte as [compressed JSON](PROTECTED-INVENTORY.json.gz); [evidence manifest](EVIDENCE-MANIFEST.json) records original uncompressed hashes and remote locations. [Full audit](../storage-audit-baige-20261107-r1/RESULT.md) · [Earlier 207.70 GiB archival](../outputs-retention-20261107-r1/RESULT.md).
