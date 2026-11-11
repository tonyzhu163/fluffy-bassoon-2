# Live coordination

[November 10 board snapshot](archive/coordination/coordination-20261110-before-clear.md) · [Verified findings](../docs/FINDINGS.md)

| Work | Cluster / job / source | Last check, progress and next action |
| --- | --- | --- |
| Synthetic learning histories exploration | Baige 4090 / `job-tar8hons981r` / code `50508c86` | Nov 11, 13:37 Shanghai. Completed all 15 fits and evaluations; scheduler succeeded 13:35:58. Held-out attempts k=6: 57.05%, same-history reference 86.50%; all seeds miss near-reference screen. Next: diagnose toy learning and causal feedback use before robot expansion. Total scheduler accounting 7.424 GPU-hours, conservative bound 7.620; no active allocation. [Results](learning-histories-20261111-r1/RESULT.md) · [Interpretation](learning-histories-20261111-r1/SCIENCE-REVIEW.md) |
| Repeated-attempt collection for the in-context learner gate | Baige, not submitted; code `d9e4b4eb` | Nov 10, 20:07 Shanghai. Scripts committed, syntax-checked only. Waiting on console access from this session; the queued retry job has priority. Next: OpenDrawer qualification, 128 episodes on one GPU. [Plan](retry-history-gate1-20261110-r1/PLAN.md) |
