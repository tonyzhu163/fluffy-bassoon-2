# Live coordination

[November 10 board snapshot](archive/coordination/coordination-20261110-before-clear.md) · [Verified findings](../docs/FINDINGS.md)

| Work | Cluster / job / source | Last check, progress and next action |
| --- | --- | --- |
| Synthetic learning histories exploration | Baige 4090 / `job-tar8hons981r` / code `50508c86` | Nov 11, 13:06 Shanghai. Grid submitted: 15 one-GPU workers, five conditions × three seeds, K=6, 20,000 updates. CUDA, bitwise checkpoint continuation and evaluation chunking passed; 500 updates took 45 s. Expected ~30 min training plus evaluation; each worker capped at 90 min. Qualification scheduler usage 220 GPU-seconds; all allocation caps sum to 23.5 GPU-hours. Next: verify advancing updates, then check every 20 minutes. [Plan](learning-histories-20261111-r1/PLAN.md) |
| Repeated-attempt collection for the in-context learner gate | Baige, not submitted; code `d9e4b4eb` | Nov 10, 20:07 Shanghai. Scripts committed, syntax-checked only. Waiting on console access from this session; the queued retry job has priority. Next: OpenDrawer qualification, 128 episodes on one GPU. [Plan](retry-history-gate1-20261110-r1/PLAN.md) |
