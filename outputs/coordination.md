# Live coordination

[November 10 board snapshot](archive/coordination/coordination-20261110-before-clear.md) · [Verified findings](../docs/FINDINGS.md)

| Work | Cluster / job / source | Last check, progress and next action |
| --- | --- | --- |
| Compact-bowl DPT action labels and students | Baige development RTX 4090, `student02` at `bf64c1ef`, final development panel `resume-01` | Nov 11, 02:33 Shanghai. The 128- and 512-update DPT panels each achieved 1/16 placements. A linear probe of those encoders read 12/16 and 8/16 hidden goals. A transient GPU utilization reading stopped the 1,024-update job before its first episode. A fresh probe passed, and this final panel has restarted. Next lock the two checkpoints. [Protocol](robocasa-bowl-dpt-labels-20261110-r1/PLAN.md) |
| Repeated-attempt collection for the in-context learner gate | Baige, not submitted; code `d9e4b4eb` | Nov 10, 20:07 Shanghai. Scripts committed, syntax-checked only. Waiting on console access from this session; the queued retry job has priority. Next: OpenDrawer qualification, 128 episodes on one GPU. [Plan](retry-history-gate1-20261110-r1/PLAN.md) |
