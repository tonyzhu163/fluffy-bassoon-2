# Two native PPO iterations completed

Baige Pro6K job`job-rxih2makcwv4`, immutable source`bb4c905a`, finished
successfully at2026-11-06 03:29:28Shanghai. All four final actor sidecars contain
40native optimizer updates. Both iteration boundaries have12worker sidecars
and full native actor/Adam/scheduler DCP files. All17final data-file SHA256s
match (11,177,195,539bytes); eight frozen-VLM after-checks are unchanged.
Four initial old-logprob replay errors are exactly0. Eight canonical rollout
receipts retain the two unshuffled source versions. No new SFT panel ran.

The common initializer is original broad Sidney SFT, not a task-specific SFT or
RL warmstart. Execution kept micro64/GPU×accum8×4=global2048, five PPO epochs,
H400/H50/K5/.3, native mixedBF16/fullFP32vision+norms and the qualified original
loss/schedule. Native batch-end trimming stayed enabled, experimental per-scene
trimming stayedOFF. Host RAM peaked492.45GiB within actual1024GiB.

| Native phase | Iteration1 | Iteration2 |
| --- | ---: | ---: |
| Collection seconds |2778.9 |2706.2 |
| Actor-training seconds |532.2 |530.6 |
| Training success_once |0.17773438 |0.20800781 |

These success metrics come from different training cases and policies. They
are not a paired evaluation or a comparison to Sidney's published SFT results,
which remain the baseline reference. This engineering run is not the matched
group8 PPO arm or six-task/student-corpus admission.

All journal entries resolved. Train07 charges819,200requested/conservative
motor commands plus23,040constructor/reset allowances =842,240slots. Including
old lineage charges gives1,128,960<1,200,000. No measured executed-motor or physics
integration-substep count is claimed; see[the accounting limitation](CONTROL-ACCOUNTING.md).

Scheduler6871sec×4=27,484GPU-sec (7.63444GPUh); creation-to-release6883sec×4=
27,532conservative seconds; worker6862sec×4=27,448. The current maximum41,040
reservation is closed and replaced by this actual charge. Final output uses
114.9701GiB, below125GiB. Protected initial and complete recovery checkpoints,
canonical histories and raw audit evidence remain in place.

[Final integrity](CHECKPOINT2-INTEGRITY.json), [boundary](checkpoint2/pilot-boundary.json),
[result](RESULT.json), [exit](EXIT.json), [native metrics](all-iteration-metrics.txt).

Post-close retention: the eight canonical rollout PTs were cold-copied with full byte/SHA readback and complete staged restoration before releasing 43.08858 GiB from PFS. Original metadata, restore pointers and all 36 checkpoint/boundary/worker files remain hot; the reported 114.97 GiB is the pre-archive snapshot. [Verified archive and restoration receipts](../../robotwin-sidney-matched-baselines-20261106-r1/train07-archive01/RESULT.md).

A second post-close retention pass cold-copied all 256 simulator trace files with full cold and staged SHA256 readback before release, reclaiming another 43.53453 GiB. All 36 checkpoint/boundary/worker files stayed hot and unchanged. Original trace paths have restore pointers; see [the complete restoration inventory](../../robotwin-sidney-matched-baselines-20261106-r1/train07-traces-archive01/RESULT.md).
