# Live coordination

[November 10 board snapshot](archive/coordination/coordination-20261110-before-clear.md) · [Verified findings](../docs/FINDINGS.md)

| Work | Cluster / job / source | Last check, progress and next action |
| --- | --- | --- |
| Synthetic learning histories exploration | Baige 4090 / `job-guy3jxur9ic0` / code `6974ae82` | Nov 11, 14:33 Shanghai. Three-worker frozen-checkpoint feedback evaluation submitted; exact original replay gates each worker before fresh comparisons. Next: verify probes/replays, then inspect paired outcomes. Additional cap 2.25 GPU-hours; prior conservative bound 7.620. [Protocol](learning-histories-feedback-20261111-r1/PLAN.md) · [Original results](learning-histories-20261111-r1/RESULT.md) |
| Repeated-attempt collection for the in-context learner gate | Baige, not submitted; code `d9e4b4eb` | Nov 10, 20:07 Shanghai. Scripts committed, syntax-checked only. Waiting on console access from this session; the queued retry job has priority. Next: OpenDrawer qualification, 128 episodes on one GPU. [Plan](retry-history-gate1-20261110-r1/PLAN.md) |
