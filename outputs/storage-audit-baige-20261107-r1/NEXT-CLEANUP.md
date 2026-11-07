# Additional cleanup review — November 7, 2026

**Next small cleanup candidates: 17.21 GiB. No payload was deleted in this follow-up.** Current PFS free space was 277.91 GiB at the fresh directory check. The earlier 207.70 GiB archival is complete and separate.

| Priority | Candidate | GiB | Evidence and boundary |
| --- | --- | ---: | --- |
| 1 | `cache/robotwin-ppo-pilot-20261105-r1-restore02/verified-cache.tar.zst` | 11.78 | Both PFS and canonical BOS copies fully rehashed now and equal the original published SHA256. Preserve the separate extracted motor directory. No visible development reader found. |
| 2 | `cache/robotwin-sidney-objects-20261105-r1/objects.zip` | 3.48 | Fresh full SHA256 matches the pinned installed asset manifest; extracted families remain. Keep those actual simulator assets and manifest. Preserve/revalidate a cold copy or pinned download recovery before eviction. |
| 3 | Old `cache/robotwin-ppo-pilot-20261105-r1/pi05-motus-effc6e1a-r1/model.safetensors` | 1.11 | Incomplete download: 1,189,316,096 actual bytes versus 7,473,091,464 required by its safetensors header. The valid restored motor is a different path. Keep compact download metadata. |
| 4 | 320 old deployment `.bundle` files in the project root and `bundles/` | 0.84 | All advertised commits are reachable from retained repository refs. Keep bundle files with missing/unretained tips; check full object closure and no in-flight deployment before deleting eligible copies. |

All paths above are relative to `/mowenjie/fluffy-bassoon-2/`. Exact files, byte totals and qualifiers are in [NEXT-CLEANUP.json](NEXT-CLEANUP.json). This is a review list, with final execution checks stated explicitly; no new deletion authorization was inferred from the user's question.

The biggest further opportunity is **127.22 GiB** of staged source members in `inputs/steering-source-members-20261105-pfs01`, backed by canonical BOS corpus archives. Its closeout explicitly keeps it protected, so it is excluded until its downstream dependency is released and restaging mapping revalidated. Another **86.18 GiB** of matched PPO/GRPO training rollouts could be cold-archived, preserving scientific history and final recovery states. The **38.77 GiB** deferred predecessor checkpoints also require complete recovery archival, rather than direct deletion. Those sizes are from the earlier full census, not a new full scan.

Shared HF/model caches, active AnyViewPhys material, held critic/joint-steering state, all extracted simulator assets and environments remain excluded. A bounded attempt to count package-cache last-link reclamation timed out, so its gross cache size is not presented as reclaimable bytes. The identified read-only observer was terminated and confirmed exited; no workload was signalled. No GPU/model computation ran.

[Cache duplicate hashes](next-cleanup-cache-verification.json) · [Download/header and retained-Git-ref evidence](next-cleanup-downloads.json) · [Fresh metadata/capacity](next-cleanup-preflight.json) · [Full audit](RESULT.md) · [Completed archival](../outputs-retention-20261107-r1/RESULT.md).
