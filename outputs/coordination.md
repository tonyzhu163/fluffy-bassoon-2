# Live coordination

[November 10 board snapshot](archive/coordination/coordination-20261110-before-clear.md) · [Verified findings](../docs/FINDINGS.md)

| Work | Cluster / job / source | Last check, progress and next action |
| --- | --- | --- |
| RoboCasa matched retries and training duration | Baige retries `job-bhvb6x6iy4cg`; curves `job-u602g5vn95np` | Nov 10, 20:15 Shanghai. Preserved all 96 supports from `job-6clge1qh9hk0`. Corrected process reuse after an XML mismatch. Curves reached four epochs. Next verify strict reset identity and native retry scores. [Plan](robocasa-three-ppo-retry-20261110-r1/PLAN.md) |
| Repeated-attempt collection for the in-context learner gate | Baige, not submitted; code `d9e4b4eb` | Nov 10, 20:07 Shanghai. Scripts committed, syntax-checked only. Waiting on console access from this session; the queued retry job has priority. Next: OpenDrawer qualification, 128 episodes on one GPU. [Plan](retry-history-gate1-20261110-r1/PLAN.md) |
