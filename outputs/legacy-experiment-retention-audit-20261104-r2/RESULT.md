# Legacy experiment retention audit

Completed 2026-11-04T22:26:19+0800 on Baige. Audited all **315** selected legacy experiment folders, then removed **235.65 GiB of obsolete payload allocation across 658,903 files**. The whole-bundle archiver was stopped when the human changed the task to audit useful code/results and delete the space-occupying portions. Its nine already completed full backups remain in BOS.

| Family | Folders audited | Files removed | Removed allocation, GiB | Retained regular-file allocation, GiB | Earlier full BOS copies |
| --- | ---: | ---: | ---: | ---: | ---: |
| robotwin | 81 | 392,183 | 123.30 | 3.33 | 3 |
| mikasa | 17 | 1,284 | 2.93 | 0.99 | 3 |
| libero | 190 | 265,415 | 109.08 | 1.75 | 0 |
| dsrl | 27 | 21 | 0.34 | 3.78 | 3 |

Removed raw training/replay/chunk arrays, obsolete model/checkpoint payloads, generated videos/frames, compressed observation/RPC snapshots and regenerable bytecode. Nineteen runtime copies of one-off scripts were removed only after matching their exact content to pinned Git blobs.

Retained configurations, logs, scalar/per-case evaluation records, numerical references, unknown code and selected terminal recoveries behind the reported FULL/RESET results. Seven RoboMimic recovery ZIP files remain for the reported comparison. The initial code review found 125 of 127 local experiment scripts recoverable from a pinned Git commit; the two unmatched launchers, totaling about 5 KiB, were preserved. Canonical reusable repository code, shared models/environments, and current/held/queued work were outside the deletion scope.

The scientific retention decisions use the existing [Findings](../../docs/FINDINGS.md): DSRL provides useful single-seed controls; the LIBERO/RoboTwin records retain their negative outcomes, headroom limits and lack of a demonstrated memory advantage. This cleanup adds no new scientific claim or simulator/model run. Unique or unrecognized code and compact evidence were kept instead of being inferred obsolete from age.

Verification reconciled every directory receipt, checked 925,458 retained record identities, rehashed 139 unique code files, rechecked 3,309 canonical result records and checked 40,114 deleted paths absent. All nine earlier archive pointers, sizes, SHA sidecars and inventory hashes reconcile. Their complete compressed streams passed extraction integrity checks; **3,296 compact code/config/log/result files (0.272 GiB)** were recovered to the audit's `cold-evidence/` on Baige and independently rehashed. Both prune passes finished with zero errors and zero skipped sources.

PFS free space at final reconciliation: **452.57 GiB** of 2048 GiB. Payload totals can count hardlinks; concurrent jobs change free space, so these are not an exact attribution of filesystem gain. No active-folder storage census was refreshed. No shared environment/model was moved and no GPU job was launched.

[Verified manifest](MANIFEST.json), [initial classification](AUDIT-INITIAL.json), [reviewed classification](AUDIT.json), [retention decisions](REVIEW.json), [first deletion receipt](PRUNE-SUMMARY.json), [remaining trace receipt](ADDITIONAL-TRACE-SUMMARY.json), [recovered compact records](COLD-RECOVERY.json), [deletion-boundary qualification](PRUNE-QUALIFICATION.json).

Full file inventories, review excerpts and per-directory receipts remain in the Baige audit folder. Each selected legacy source has an adjacent `.retention.json` marker pointing to this audit. Full recovery of the nine earlier cold copies uses the adjacent `.bos-archive.json` pointer and its recorded `tar --ignore-zeros` command; all four tar streams must be read.
