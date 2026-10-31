# Task57 DSRL due check — 2026-10-31 16:16 Shanghai

Runtime was not queried before the due time. The existing Baige job `job-iudscqfntb4e` remains Running at the original source/grant, cases, recipe and limits. No source edit, test, job control, new allocation or journal traversal occurred.

At 2026-10-31T16:16:28.771093+08:00: 665/666 training successes, 9,285 replay chunks, 175,840 updates and 120,473 all-in actions. Losses are finite. There is no error exit or retry.

| New threshold | Actual updates | Success | Primitives | Settling |
| ---: | ---: | ---: | ---: | ---: |
| 160,000 | 160,240 | 10/10 | 1,261 | 100 |
| 170,000 | 170,160 | 10/10 | 1,245 | 100 |

Native remains 10/10 with 1,451 primitives plus 100 settling actions. Newest query seeds, nominal states and actual initial-observation hashes match native; query parameters/replay remain unchanged with zero query SGD. The descriptive cost trend continues without a success-rate, overall sample-efficiency or meta-learning claim.

The first training failure is a completed 400-step timeout at zero-indexed episode 576, state16. It is an RL outcome, not a runtime fault. A selective saved-metadata follow-up bound the failed trace and nearby visits:

| Training episode | State | Success | Primitives | Updates before → after |
| ---: | ---: | --- | ---: | --- |
| 536 | 16 | True | 143 | 140,540 → 140,840 |
| 576 | 16 | False | 400 | 151,260 → 152,060 |
| 616 | 16 | True | 133 | 162,580 → 162,860 |
| 656 | 16 | True | 134 | 173,160 → 173,440 |

All four visits have equal nominal-state and actual initial-observation hashes. The failure raw trace rehashes correctly. Later successes use different training RNG and changed actor weights; they do not establish causal recovery, memory or adaptation. No extra rollout or physical failure-mode analysis was performed.

The 271,527,629-byte current full recovery, 264,341,179-byte newest immutable milestone, newest completed raw trace and RPC input/outcome pair all match their hashes. Producer cost counters sum internally with zero uncertain actions; exhaustive journal reconciliation remains deferred. The earlier partial tail was not reused as a verified prefix.

Storage is 10,430,706,619 bytes against 64GiB. One attempt still reserves 36,000 aggregate GPU-seconds; 7775.8 active elapsed seconds is not a final charge. The action cap remains 1M.

Reader bytes are bound to immutable Git commit `f75b46b55479637959c8e6e280f334e49d8bf14f` and blob `d1391d0abf95f64dbef007028722f2d0e7f18092`. The next-check argument was `2026-10-31T16:36:00+08:00`.

Next owned check: 16:36 Shanghai. No further polling this turn.

![Existing job Running](screenshots/gpu-check-1616.jpg)
