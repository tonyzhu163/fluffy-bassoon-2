# AnyViewPhys entire-project cold archive — verified complete

**Archived the entire `/mowenjie/anyviewphys` tree and released 288.841 GiB (310,140,596,224 allocated bytes) from PFS.** All 511,379 original entries have complete cold recovery. The final reconciliation passed at **2026-11-08 02:34:40 Shanghai**; PFS then had **355.95 GiB free**. The original path now contains only `COLD-ARCHIVE.json` and `RESTORE.md`, with an adjacent recovery pointer.

| Measure | Verified value |
| --- | ---: |
| Archived / restored / released entries | 511,379 / 511,379 / 511,379 |
| Source unique regular logical content | 288.356 GiB |
| Cold archive | 232.514 GiB (249,659,603,127 bytes) |
| Physical allocation released | 288.841 GiB |
| Prior two verified cleanups | 224.915 GiB |
| Combined verified allocation released | 513.756 GiB |
| New GPU allocations | 0 |

The archive was formed as eight cold objects under the prior R3 run after the single-object attempt failed. This continuation reused that fully verified cold archive and extracted tree after a scratch-metadata stability guard stopped the prior verifier. It synced scratch and required every restored reference SHA256 and archived logical attribute/inode/link identity. The resumed pass recorded zero filesystem-local allocation/ctime changes. The original PFS source also passed its full strict metadata/ctime check before release. See [MIGRATION.json](MIGRATION.json), [PLAN.md](PLAN.md) and [RESTORED-STORAGE-METADATA.json](RESTORED-STORAGE-METADATA.json).

The scope includes models, Python environments, caches, code and complete Git state, run outputs, snapshots, tools, third-party trees and staging. The original AnyViewPhys repository was clean at `834804455987004b3ca69d1dac91ccfc49288871`. No project subset was excluded.

The private cold manifest lists eight independently verified objects (at most 32 GiB each):

`/mnt/bos/adgfyl/strong-plant-archives/anyviewphys-cold-archive-20261108-r4/anyviewphys.tar.zst.parts.json`

The reassembled archive's SHA256 is `f8cb2a601d4a00aef9dbe5315d491f3a3b1879e9d720b2eff362863d6c64065e`. The manifest has its own checksum in [RECOVERY.json](RECOVERY.json). The eight data objects remain under `anyviewphys-cold-archive-20261108-r3`; this run's cold manifest and metadata reference those exact preserved objects. Keep both the data objects and recovery metadata. The losslessly compressed full content inventory, source, receipt, restore instructions and complete release journal accompany the manifest. [BULK-EVIDENCE.json](BULK-EVIDENCE.json) records exact byte counts, SHA256 and PFS/BOS locations for the large inventories and journal; those raw bodies are kept out of Git.

Before releasing the hot tree, the workflow independently read back every cold part and matched its SHA256, reassembled and verified the complete archive digest, extracted that readback, synced scratch, and compared every restored path, regular-file size/SHA256, mode, numeric owner, nanosecond mtime, symlink target and internal hardlink relationship against the complete original source inventory. The Linux fixture also qualified ACL/xattr-capable tar options and verified user-xattr restoration. A full source metadata/ctime recheck, fresh scheduler attestation and reader checks preceded the atomic source quarantine. Release checked each inventoried inode and recorded every removed entry; final reconciliation matched all journal paths and released-byte totals, cold metadata and both original-path pointers. Scratch staging, quarantine and the archival fence were removed, and the worker exited.

Physical reclaim counts file allocation only at the last actual hardlink and directory allocation once. It can differ from gross directory estimates, and the observed PFS free-space change includes other jobs' concurrent writes. The prior 224.915 GiB cleanup total is separate from this project's measured reclaim.

The only AnyViewPhys controller was already paused before scoring began. Its prepared plans and completed CPU inputs are preserved; [QUIESCE.json](QUIESCE.json) records the exact boundary. Restoration must reconcile its original admission/hold and create an explicit continuation before scoring. The archive contains 6 symlinks to shared Python interpreters under `/root`, with no symlink to outside project data; [EXTERNAL-LINKS.json](EXTERNAL-LINKS.json) lists those dependencies; [EXTERNAL-RUNTIME-FINGERPRINT.json](EXTERNAL-RUNTIME-FINGERPRINT.json) records their two resolved binaries, exact SHA256 and Python versions (3.11.17 and 3.12.8). Restoring files requires sufficient PFS capacity and Linux GNU tar/zstd, followed by validation of outside dependencies and the Python/accelerator runtime. This is not a process-memory, base-OS or driver snapshot.

Follow [RESTORE.md](RESTORE.md) to verify the complete cold digest, extract into fresh staging, compare the full inventory and restore the original path without overwriting newer work.

Evidence: [execution result](RESULT.json), [cold receipt](RECOVERY.json), [final verification](FINAL-VERIFICATION.json), [evidence manifest](EVIDENCE-MANIFEST.json), [qualification](QUALIFICATION.json), [runtime](archive.py), [final verifier](verify_final.py), [full storage audit](../storage-audit-baige-20261107-r1/RESULT.md). Execution source `0c782dd7f4ef7c8092b7884ca3d488a6c44cb544`; script SHA256 `5418232b8bb9078f50f99b838e4c623631a5c7d1bda75a2672e8e420d7b10631`. No active computation worktree was edited.
