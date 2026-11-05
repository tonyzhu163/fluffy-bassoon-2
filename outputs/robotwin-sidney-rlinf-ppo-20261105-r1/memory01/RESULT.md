# RoboTwin reset memory attribution

Baige job `job-gczjjcuo53sq` completed a model-free, 16-scene diagnostic on
the pinned RoboTwin runtime. No VLA or learner was loaded. This is a host-memory
diagnostic, with zero scientific PPO updates.

| Phase | Process RSS (GiB) | Live glibc allocations (GiB) |
| --- | ---: | ---: |
| Constructor, including temporary setup scenes | 1.14 | 0.29 |
| First reset | 15.34 | 13.62 |
| After 400 fixed-hold controls per scene | 15.39 | 13.79 |
| Second reset, after Python GC | 27.93 | 13.79 |
| Second reset, after `malloc_trim(0)` | 15.13 | 13.79 |
| Closed, deleted, collected and trimmed | 2.17 | 0.72 |

The second reset leaves approximately 13.34 GiB of freed allocations in glibc's
arenas. Trimming returns about 12.79 GiB of resident pages while the measured
live allocations stay unchanged. Python tracing is about 87 MiB and arrays
reachable from task attributes total about 46 MiB. The fixed-hold rollout does
not show a comparable growth in live allocations. This attributes the dominant
reset-to-reset increase to allocator retention of native simulator allocations;
it does not establish that every native allocation is necessary.

The retained raw RGB/state observation hashes remain identical across GC and
trimming, and every inspected camera is nonblank. The newer 64-scene diagnostic
must still qualify reset-time peaks and paired reset/motor behavior before a
production retry. A batch-end trim can reduce steady RSS while leaving a large
peak during the reset itself.

All 7,200 physical fixed-hold controls and 480 setup/reset allowances were
charged to the original shared action ledger under the fresh `memory01`
attempt. Scheduler allocation was 411 GPU-seconds; creation through release was
424 GPU-seconds, and the worker clock recorded 395 GPU-seconds. Source commit:
`6f0262ff02305383f071c2f27dda1e1a289c1146`.

Raw evidence: [RESULT.json](RESULT.json), [EXIT.json](EXIT.json),
[allocated GPU probe](gpu-probe.json).
