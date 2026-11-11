# Live coordination

[November 10 board snapshot](archive/coordination/coordination-20261110-before-clear.md) · [Verified findings](../docs/FINDINGS.md)

| Work | Cluster / job / source | Last check, progress and next action |
| --- | --- | --- |
| Synthetic learning histories exploration | Baige 4090 / `job-h9o49h1vu3zt` (15 GPUs) / pinned `4b06413e` | Nov 11, 15:32 Shanghai. CUDA qualification passed: dense numerics, bitwise resumes, exact old-curve replay; projected fits 26–30 min. Grid has all 15 GPU probes passed; two workers at 1,000 updates, 13 still starting. Next: sub-agent checks startup around 15:34 and monitors every 20 min while preparing LIBERO; retain all checkpoints. Prior conservative usage 7.768 GPU-hours + qualification 0.0692; grid cap 22.5. [Execution](learning-histories-fix-20261111-r1/EXECUTION.json) |
| Repeated-attempt collection for the in-context learner gate | Baige, not submitted; code `d9e4b4eb` | Nov 10, 20:07 Shanghai. Scripts committed, syntax-checked only. Waiting on console access from this session; the queued retry job has priority. Next: OpenDrawer qualification, 128 episodes on one GPU. [Plan](retry-history-gate1-20261110-r1/PLAN.md) |
| LIBERO-Gen history pilot | Baige preparation only / BPP `58125e68`; no GPU job | Nov 11, 15:23 Shanghai. Verified demonstration-only BPP interface and exact same-scene two-bowl motor pair; static inventory 183 seen / 10 held-out BDDL files. Isolated official source clone completed; LIBERO submodule/runtime/checkpoint still unqualified. Next: finish runtime/data preparation, then bounded motor qualification before reader training. Additional usage 0 / 5 GPU-hours. [Plan](libero-gen-history-pilot-20261111-r1/PLAN.md) · [Evidence](libero-gen-history-pilot-20261111-r1/INTERFACE.md) |
