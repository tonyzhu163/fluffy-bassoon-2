# Second trained source round pair

Baige job-d0e0hq6jgang completed Success at01:32:27Shanghai on2026-10-31. Actual scheduler time was391GPU-seconds (01:25:56start); parent runtime379.547seconds and pair phase334.662seconds are separate timing boundaries. Same qualified57016ba runtime and c353970 launcher; no additional qualification rollouts or changed optimizer/model contract.

| Learner | Final round | CompleteK4 | Source actions | PPO updates / SGD |
| --- | ---: | ---: | ---: | ---: |
| Persistent PPO | 54 | 324 | 517,443 (+8,903) | 54 /270 (+5) |
| Trained reset PPO | 56 | 336 | 535,670 (+10,045) | 56 /280 (+5) |

This adds18,948 new source actions,614 recorded decisions and48 complete episodes. Both children exited0; original canonical heads have zero pending/uncertain charges. All12 likelihood/value replay errors were exactly0.0. Reported updateKL was0.0002397/0.0001739, below0.1. Frozen motor, raw-corpus/native masks, durable acknowledgments, full checkpoint/Adam/sampler/RNG/evaluation-cursor recovery and owned cleanup passed allocated and independent audits. Both263-member checkpoint archives pass CRC. The100-round source goal remains unfinished.

Full scheduler throughput was48.46actions/GPU-second; this is another measured workload, not a paired scalar speedup ratio. Peak VRAM70,387,761,152bytes=65.55GiB including actual SGD. No retry or memory abort occurred. Unused2,309operationalGPU-seconds are sealed. No extra trials are collected to fill unused action capacity, and future continuation stays within original canonical source caps.

[Independent audit](receipts/independent-final-audit.json) · [Allocated audit](receipts/audit.json) · [Scheduler](receipts/scheduler-terminal.json). Root continued a uniquely named R3 original-goal unit while the sole integration owner prepares and qualifies the continuous supervisor; no learner was reinitialized.
