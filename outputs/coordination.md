# Live coordination

[November 10 board snapshot](archive/coordination/coordination-20261110-before-clear.md) · [Verified findings](../docs/FINDINGS.md)

| Work | Cluster / job / source | Last check, progress and next action |
| --- | --- | --- |
| Compact-bowl DPT action labels and students | Baige development RTX 4090, `common-handoff01` at `ed87f13c` | Nov 11, 09:34 Shanghai. The instructed motor reproduced 2/8 training goals at 128 updates and 3/8 at 1,024. A recorded failed replay kept lifting the grasped fruit and never approached the bowl. The exploratory phase check now hands over after the common scripted lift on those same eight training goals. [Training cases and observed failure](robocasa-bowl-dpt-labels-20261110-r1/RESULT.md) |
| Repeated-attempt collection for the in-context learner gate | Baige, not submitted; code `d9e4b4eb` | Nov 10, 20:07 Shanghai. Scripts committed, syntax-checked only. Waiting on console access from this session; the queued retry job has priority. Next: OpenDrawer qualification, 128 episodes on one GPU. [Plan](retry-history-gate1-20261110-r1/PLAN.md) |
