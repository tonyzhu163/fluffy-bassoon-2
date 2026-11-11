# Live coordination

[November 10 board snapshot](archive/coordination/coordination-20261110-before-clear.md) · [Verified findings](../docs/FINDINGS.md)

| Work | Cluster / job / source | Last check, progress and next action |
| --- | --- | --- |
| Synthetic learning histories exploration | Baige 4090 / `job-tar8hons981r` / code `50508c86` | Nov 11, 13:10 Shanghai. All 15 fits passed distinct-GPU probes and reached 1,500/20,000 updates. Five conditions × three seeds, K=6. Checkpoint recovery is bitwise exact. Next check 13:30; expect completion around 13:40. Cumulative allocation upper bound 1.23 GPU-hours at last check; all allocation caps sum to 23.5 GPU-hours. [Plan](learning-histories-20261111-r1/PLAN.md) · [Interpretation](learning-histories-20261111-r1/SCIENCE-REVIEW.md) |
| Repeated-attempt collection for the in-context learner gate | Baige, not submitted; code `d9e4b4eb` | Nov 10, 20:07 Shanghai. Scripts committed, syntax-checked only. Waiting on console access from this session; the queued retry job has priority. Next: OpenDrawer qualification, 128 episodes on one GPU. [Plan](retry-history-gate1-20261110-r1/PLAN.md) |
