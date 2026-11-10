# Live coordination

[November 10 board snapshot](archive/coordination/coordination-20261110-before-clear.md) · [Verified findings](../docs/FINDINGS.md)

| Work | Cluster / job / source | Last check, progress and next action |
| --- | --- | --- |
| Compact-bowl DPT action labels and students | Baige development RTX 4090, `student02` at `bf64c1ef` | Nov 11, 01:10 Shanghai. Both students reached update 8 with finite gradients. They use 956 training chunks and 374 development chunks. RLinf input packing was verified on all 1,539 saved branch observations after student01 stopped before its first update. Next score the 128, 512 and 1,024-update checkpoints in the simulator. [Evidence](robocasa-bowl-dpt-labels-20261110-r1/corpus-evidence.json) |
| Repeated-attempt collection for the in-context learner gate | Baige, not submitted; code `d9e4b4eb` | Nov 10, 20:07 Shanghai. Scripts committed, syntax-checked only. Waiting on console access from this session; the queued retry job has priority. Next: OpenDrawer qualification, 128 episodes on one GPU. [Plan](retry-history-gate1-20261110-r1/PLAN.md) |
