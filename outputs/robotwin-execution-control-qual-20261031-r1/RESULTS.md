# Allocated execution-control qualification

Baige job-k1upj2ycy60q ended Success at2026-10-31 00:44:14 Shanghai. Actual allocation was0GPUs/4CPUs/16GiB; scheduler running time71seconds. Process exit0 and all123 tests passed without skips:

| Package | Exact commit | Tests | Pytest time |
| --- | --- | ---: | ---: |
| Source owner control | c62be99615e797fbec108e1f1b5e232e4a591494 | 45 | 6.80s |
| Future frozen-panel guard | c166171b847ada9a20a9629a07d5a0100b517097 | 31 | 1.97s |
| Trained paired launcher | c353970a316837b13277ec65587e58e5cb25ce56 | 47 | 19.40s |

The launch artifact was3ea7a97b33ca0ce9fccc8c649a63465363439931, based on reviewed latest main acca706d5e75d1da099ff7ce6fddd727dace0fba. Source and native model code were not changed. The tests exercised only owned temporary fake processes and tiny synthetic CPU Adam fixtures. Actual source process signals, robot actions, VLA queries and production optimizer updates were all0.

This qualifies the software gates; trained GPU collection, current-owner release and the actual new checkpoint/ledger bindings still require their separate live admission. No serial source owner was paused or stopped by this test job.

Raw phase logs and receipt are under receipts/. [Scheduler proof](receipts/scheduler-terminal.png).
