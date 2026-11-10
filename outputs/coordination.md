# Live coordination

[November 10 board snapshot](archive/coordination/coordination-20261110-before-clear.md) · [Verified findings](../docs/FINDINGS.md)

| Work | Cluster / job / source | Last check, progress and next action |
| --- | --- | --- |
| Compact-bowl DPT action labels and students | Baige development RTX 4090, `student02` at `bf64c1ef`; `confirmation-01` at `78adcc26` | Nov 11, 03:08 Shanghai. All 72 development rollouts finished. DPT and memoryless each selected 1,024 updates with 3/16 correct placements. The checkpoints were locked before collecting 16 disjoint confirmation scenes. Next verify their scripted branch coverage and evaluate all locked policies. [Selection and per-episode outcomes](robocasa-bowl-dpt-labels-20261110-r1/development.json) |
| Repeated-attempt collection for the in-context learner gate | Baige, not submitted; code `d9e4b4eb` | Nov 10, 20:07 Shanghai. Scripts committed, syntax-checked only. Waiting on console access from this session; the queued retry job has priority. Next: OpenDrawer qualification, 128 episodes on one GPU. [Plan](retry-history-gate1-20261110-r1/PLAN.md) |
