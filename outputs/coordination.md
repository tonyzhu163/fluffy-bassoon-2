# Live coordination

[November 10 board snapshot](archive/coordination/coordination-20261110-before-clear.md) · [Verified findings](../docs/FINDINGS.md)

| Work | Cluster / job / source | Last check, progress and next action |
| --- | --- | --- |
| Synthetic learning histories exploration | Baige 4090 / grid `job-h9o49h1vu3zt`, confirmation `job-obkpwzu2057v` / trainer `4b06413e` | Nov 11, 16:19 Shanghai. Complete: 15/15 fits and 12/12 fresh evaluations, all exit 0. Selected corrections 75.64% versus original 67.40%; same-history reference 94.94%. Attempts 60.50% versus 57.34%; substantial gaps remain. All checkpoints preserved. Toy cumulative 16.212 scheduler / 16.606 conservative GPU-hours. Next: finalize evidence; continue separately owned LIBERO feasibility work, next status 16:39. [Result](learning-histories-fix-20261111-r1/RESULT.md) · [Execution](learning-histories-fix-20261111-r1/EXECUTION.json) |
| Repeated-attempt collection for the in-context learner gate | Baige, not submitted; code `d9e4b4eb` | Nov 10, 20:07 Shanghai. Scripts committed, syntax-checked only. Waiting on console access from this session; the queued retry job has priority. Next: OpenDrawer qualification, 128 episodes on one GPU. [Plan](retry-history-gate1-20261110-r1/PLAN.md) |
| LIBERO-Gen history pilot | Baige CPU admission passed / no GPU job | Nov 11, 16:41 Shanghai. Selected 8.45 GB assets and source verified; current-owner import fixed. Private GL/EGL libraries resolve missing imports; all BPP/runtime imports and both prompt schemas pass. Next: deploy matching launcher, then two native motor episodes on one GPU. Usage 0 / 5 GPU-hours. [Plan](libero-gen-history-pilot-20261111-r1/PLAN.md) · [Status](libero-gen-history-pilot-20261111-r1/PREPARATION-STATUS.json) |
