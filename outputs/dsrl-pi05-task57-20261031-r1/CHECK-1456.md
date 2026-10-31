# Task57 DSRL due check — 2026-10-31 14:56 Shanghai

The owned Baige job `job-iudscqfntb4e` is Running. Source `d001556780c78893cb7ad0310ec46e5d817e5698`, grant `48acb746…`, cases and all cumulative limits are unchanged. No job controls, source edits, tests or new allocations occurred.

The first read at 14:59 showed 280/280 successful training episodes, 71,120 optimizer updates and 53,108 all-in actions. The verification snapshot at 2026-10-31T15:00:54.337567+08:00 contains 293/293 successful training episodes, 4223 replay chunks, 74600 updates and 54893 actions. Latest actor/critic/entropy/temperature metrics are finite. No failure or recovery example has appeared.

| Newly completed panel | Actual updates | Success | Primitives | Settling | Paired shorter/equal/longer vs native |
| --- | ---: | ---: | ---: | ---: | ---: |
| learned-040000 | 40160 | 10/10 | 1320 | 100 | 10/0/0 |
| learned-050000 | 50080 | 10/10 | 1343 | 100 | 10/0/0 |
| learned-060000 | 60000 | 10/10 | 1304 | 100 | 10/0/0 |
| learned-070000 | 70260 | 10/10 | 1279 | 100 | 10/0/0 |

Native remains 10/10 with 1,451 primitives plus 100 settling actions. Every completed panel uses the same recorded states, stochastic query seeds, nominal-state hashes and actual initial-observation hashes. Query weights/replay remain unchanged and query SGD is zero. This is success preservation with shorter observed query trajectories on this small bank; it does not establish a success-rate gain or overall sample-efficiency benefit after training cost.

The cursor-bound latest full recovery was hashed before the metadata walk and matched (143476901 bytes). The newest immutable 70k-threshold checkpoint also matched its saved SHA. The newest completed-panel and latest-training raw traces match; the latest completed training episode's RPC input/outcome pair matches. Already verified immutable prefix artifacts were reused rather than rehashed.

The producer counters sum to 54893 actions: 43246 training, 11637 query and 10 qualification settling actions. All settling is included; the saved counters report zero uncertain actions. The prior independently reconciled prefix is bound by SHA. A bounded 120-second walk checked 1506 later journal records and found no unresolved intents in that read range; it did not finish the whole new tail, so no complete independent tail reconciliation is claimed. Exhaustive reconciliation is deferred to termination.

The single worker still holds the same 36,000-second aggregate ceiling. Active elapsed time at the verification snapshot was 3241.3 seconds, not a final charge. Saved output is 4252256567 bytes against 64GiB; the action ceiling remains 1M. The original motor before-hash is unchanged evidence; its final after-hash awaits termination.

Next owned check: 15:16 Shanghai. No further polling this turn.

![Existing job Running](screenshots/gpu-check-1456.jpg)
