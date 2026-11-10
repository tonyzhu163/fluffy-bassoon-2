# Live coordination

[November 10 board snapshot](archive/coordination/coordination-20261110-before-clear.md) · [Verified findings](../docs/FINDINGS.md)

| Work | Cluster / job / source | Last check, progress and next action |
| --- | --- | --- |
| Compact-bowl DPT action labels and students | Baige development RTX 4090, `pipeline-01`, collection `78adcc26` | Nov 10, 23:23 Shanghai. Qualification finished at 0.406 worker GPU-hours. Both releases on 7/8 layouts, public-history decoder 15/16. Training-split geometry is running. Next collect 24 training and eight validation scenes, then compare frozen adapters. [Evidence](robocasa-bowl-dpt-labels-20261110-r1/QUALIFICATION.md) |
| Repeated-attempt collection for the in-context learner gate | Baige, not submitted; code `d9e4b4eb` | Nov 10, 20:07 Shanghai. Scripts committed, syntax-checked only. Waiting on console access from this session; the queued retry job has priority. Next: OpenDrawer qualification, 128 episodes on one GPU. [Plan](retry-history-gate1-20261110-r1/PLAN.md) |
