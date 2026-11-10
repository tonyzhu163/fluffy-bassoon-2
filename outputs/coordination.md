# Live coordination

[November 10 board snapshot](archive/coordination/coordination-20261110-before-clear.md) · [Verified findings](../docs/FINDINGS.md)

| Work | Cluster / job / source | Last check, progress and next action |
| --- | --- | --- |
| Compact-bowl DPT action labels and students | Baige development RTX 4090, collection `78adcc26`, student `52cab354` | Nov 11, 00:38 Shanghai. Training collection finished. Nineteen of 24 scenes supplied both correct continuations, yielding 956 masked action chunks. The public-history decoder read 43/48 hidden goals across held-out pairs. Next complete eight validation scenes and verify the first optimizer updates. [Protocol](robocasa-bowl-dpt-labels-20261110-r1/PLAN.md) |
| Repeated-attempt collection for the in-context learner gate | Baige, not submitted; code `d9e4b4eb` | Nov 10, 20:07 Shanghai. Scripts committed, syntax-checked only. Waiting on console access from this session; the queued retry job has priority. Next: OpenDrawer qualification, 128 episodes on one GPU. [Plan](retry-history-gate1-20261110-r1/PLAN.md) |
