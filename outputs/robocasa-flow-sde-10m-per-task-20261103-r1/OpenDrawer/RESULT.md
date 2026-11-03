# OpenDrawer: native readout after 130 updates

Verified Nov4 2026 on Baige Pro6K, `job-9j5eqjo0dfzj`, fixed runtime
`0988b5f9e9bf31ac3db673cc2860959611b9ae12`. Eight GPUs, 16 TRAIN and 8 EVAL lanes
per GPU, two actor ranks, KL penalty off. Native process exited 0.

**The run finished its target but failed both preregistered teacher gates.**
Its corpus is retained as reference data, without admission as an improving
teacher. No extra training or gate changes were made.

| Readout | Before / early | After / late |
| --- | ---: | ---: |
| TRAIN first success, iterations 1–10 vs 121–130 |823/1280 (64.30%) |882/1280 (68.91%) |
| TRAIN success at end, same windows |799/1280 (62.42%) |861/1280 (67.27%) |
| Paired 64 first success |40/64 (62.50%) |47/64 (73.44%) |
| Paired 64 success at end |37/64 (57.81%) |43/64 (67.19%) |

TRAIN first success improved 4.609375 percentage points, below the required 10.
The held-out panel had 13 improved cases and 6 regressions; exact two-sided
McNemar p=0.1670684814453125, above 0.05. The observed positive difference does
not meet the registered evidence threshold.

All 130 TRAIN cohorts have 128 complete episodes of 600 controls with unique,
correct seeds: 16,640 episodes and **9,984,000 TRAIN controls**, plus 76,800 controls
in the two evaluation panels. Both actor ranks completed 3,120 logical optimizer
steps; rank counts are not added together. Initial likelihood gaps were zero
on both ranks throughout; losses were finite and the frozen VLM stayed unchanged.
The final full checkpoint's model files and all 18 worker sidecars were hash
verified. All 64 before/after cases matched in identity, native physical state,
camera/asset bank and public state; image differences remain diagnostic.

All 130 numerical corpus cohorts and their joins and raw-file manifest hashes
now pass. The full source corpus and QC metadata are verified in BOS;
interrupted cleanup was completed using the guarded recovery receipt. Final
130 remains fully resumable on PFS. See [campaign results](../RESULT.md),
[readout](FINAL-READOUT-R1.json), [action ledger](FINAL-ACTION-ACCOUNTING-R1.json)
and [archive recovery](../CORPUS-ARCHIVE-OpenDrawer-R1-RECLAIMED.json). Offline
initial-policy likelihood reporting remains pending.
