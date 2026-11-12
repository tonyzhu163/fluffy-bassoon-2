# Additional PPO teacher pilots

All three authorized pilots completed ten PPO updates and preserved 1,280 training episodes per task, full resumable checkpoints, and verified private BOS archives. Cabinet resumed from the original step-8 optimizer, sampler and RNG checkpoint and completed steps 9–10. The before/after evaluation uses the same 64 cases at the native 600-control horizon.

| Task | Before | After | Paired gains / losses | Exact two-sided McNemar p |
| --- | ---: | ---: | ---: | ---: |
| OpenCabinet | 38/64 | 40/64 | 9 / 7 | 0.804 |
| SlideDishwasherRack | 42/64 | 41/64 | 4 / 5 | 1.000 |
| PickPlaceCounterToStove | 40/64 | 40/64 | 7 / 7 | 1.000 |

These short, single-seed pilots demonstrate working collection and optimizer execution. They do not establish improved teachers. The three tasks should not yet be counted as effective new sources. Preserve the collected histories and checkpoints; avoid scaling this teacher recipe on the basis of these endpoints alone.

All initial physical and public states matched the canonical cases exactly. Initial image comparisons were diagnostic: maximum RGB deltas were 1, 1 and 2 intensity levels for Cabinet, Rack and CounterToStove, respectively. Every environment JSON used in the coverage calculation was read from its archive and checked against its recorded SHA256. Each task has 128 training episodes in each of the ten iterations. Cabinet's two archive segments preserve the complete chronology.

Explicit DPT teacher-query actions were not collected by these behavior rollouts. The four saved transformer comparison still uses the original Drawer/Toaster/Coffee corpus; no six-task transformer retraining is claimed.

[Coverage evidence](evidence-final/TEACHER-COVERAGE.json) · [Execution and checkpoint provenance](EXECUTION.json) · [Shared conservative compute accounting](../robocasa-kv-comparison-20261112-r1/BUDGET-RECONCILED.json).
