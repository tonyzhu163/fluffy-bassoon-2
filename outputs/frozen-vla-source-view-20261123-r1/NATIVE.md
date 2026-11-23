# Frozen π₀.₅ source-view evaluation on Kinova3

Baige Distributed Training shows `job-69hluhi5wgb7` queued at the first check, 19:12 Shanghai, November 23. It was submitted at 19:12:00 for one RTX 4090, 12 CPUs and 48 GiB. The launcher is pinned to `2b59cff7e173ee8064905256b98a1dd3fd0f79d5` with a one-GPU-hour cap. The first PFS check found no launcher directory. Queue time is tracked separately from GPU use.

The monitor watches `baige-dev:/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-source-view-20261123-r1/gpu01` for device qualification, the first matched proposal and the first real native episode. The next scheduled status check is 19:32 or earlier upon allocation. Frame comparisons use raw, Kinova render, Panda render and raw repeat on starts 14/15. Native scores use separate same-job raw, Kinova and Panda branches on starts 16/17. [Plan](PLAN.md) · [Renderer qualification](RESULT.md) · [Launch contract](HANDOFF.md)

## Monitor checkpoints

At 19:52 Shanghai the scheduler still showed this job queued, 40 minutes after submission. The PFS launcher directory was absent. Next check 20:12 or on receipt of the launch files. [Timestamped observations](native-readback/monitor.tsv)
