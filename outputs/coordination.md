# Live coordination

[November 10 board snapshot](archive/coordination/coordination-20261110-before-clear.md) · [Verified findings](../docs/FINDINGS.md)

| Work | Cluster / job / source | Last check, progress and next action |
| --- | --- | --- |
| Synthetic learning histories exploration | Baige 4090; preparing qualification; code `8bbee7fd` | Nov 11, 12:58 Shanghai. Five conditions × three seeds; K=6, 20,000 updates. Static checks passed; deployed Git bundle for CUDA qualification. Next: verify checkpoint recovery and time 500 updates, then launch the grid. Maximum planned allocation 36.5 GPU-hours. [Plan](learning-histories-20261111-r1/PLAN.md) |
| Repeated-attempt collection for the in-context learner gate | Baige, not submitted; code `d9e4b4eb` | Nov 10, 20:07 Shanghai. Scripts committed, syntax-checked only. Waiting on console access from this session; the queued retry job has priority. Next: OpenDrawer qualification, 128 episodes on one GPU. [Plan](retry-history-gate1-20261110-r1/PLAN.md) |
