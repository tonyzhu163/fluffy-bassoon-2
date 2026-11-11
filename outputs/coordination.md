# Live coordination

[November 10 board snapshot](archive/coordination/coordination-20261110-before-clear.md) · [Verified findings](../docs/FINDINGS.md)

| Work | Cluster / job / source | Last check, progress and next action |
| --- | --- | --- |
| Synthetic learning histories exploration | Baige 4090 / `job-guy3jxur9ic0` / eval code `6974ae82` | Nov 11, 14:40 Shanghai. Completed three-seed paired diagnosis; all six original curves replay exactly. Held-out random histories: real 54.65%, masked 43.94%, shuffled 38.83%; feedback helps every seed. Training-actor histories recover only 1.77 points. Next: bounded toy optimization/supervision comparison. No active allocation. Cumulative scheduler usage 7.530 GPU-hours; conservative bound 7.768. [Results](learning-histories-feedback-20261111-r1/RESULT.md) |
| Repeated-attempt collection for the in-context learner gate | Baige, not submitted; code `d9e4b4eb` | Nov 10, 20:07 Shanghai. Scripts committed, syntax-checked only. Waiting on console access from this session; the queued retry job has priority. Next: OpenDrawer qualification, 128 episodes on one GPU. [Plan](retry-history-gate1-20261110-r1/PLAN.md) |
