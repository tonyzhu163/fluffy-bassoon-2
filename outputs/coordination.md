# Live coordination

[November 10 board snapshot](archive/coordination/coordination-20261110-before-clear.md) · [Verified findings](../docs/FINDINGS.md)

| Work | Cluster / job / source | Last check, progress and next action |
| --- | --- | --- |
| Synthetic learning histories exploration | Baige 4090 / previous `job-guy3jxur9ic0`; recipe screen awaiting qualification | Nov 11, 15:21 Shanghai. Paired diagnosis complete: random-history real feedback 54.65%, masked 43.94%, shuffled 38.83%. Reviewed incoming five-recipe plan; fixed old-checkpoint defaults and added fresh evaluation RNG control. Prepared CUDA numerical/resume/capacity qualification; no new job launched. Next: qualify one GPU, then launch the 15-fit grid if timing permits. Used 7.530 scheduler GPU-hours / 7.768 conservative; reserve at most 23.25 more for this screen. [Plan](learning-histories-fix-20261111-r1/PLAN.md) |
| Repeated-attempt collection for the in-context learner gate | Baige, not submitted; code `d9e4b4eb` | Nov 10, 20:07 Shanghai. Scripts committed, syntax-checked only. Waiting on console access from this session; the queued retry job has priority. Next: OpenDrawer qualification, 128 episodes on one GPU. [Plan](retry-history-gate1-20261110-r1/PLAN.md) |
