# Parallel native-horizon collection is running

Verified 2026-10-30T16:25:09.832135+08:00 Asia/Shanghai. Source `db01840`, with latest origin/main
`acca706` reviewed before deployment. All three allocations independently
passed 112 Linux tests, actual GPU/CUDA probes, and native-noise parity 0.0.
Their GPUs are distinct 96GB NVIDIA RPBZZZ6 devices in Baige's Pro6K pool.

| Method | Baige job | Run output identity |
| --- | --- | --- |
| Persistent PPO | job-6phpm6rnvuf0 | robotwin-wholetrial-ppo-20261030-r2 |
| SAC | job-yd05kvzuqlh9 | robotwin-wholetrial-sac-20261030-r2 |
| Trained-reset PPO | job-d8moi1k3ufye | robotwin-wholetrial-ppo_reset-20261030-r2 |

The first shared initial cases matched across all three GPU processes:
8/8 exact frozen-public-feature hashes.
The snapshot rates are approximately 29–32 actions/s per GPU during collection,
excluding startup and subsequent optimizer/evaluation work. Their sum is a
rough concurrent-rate estimate, not a matched benchmark of the pending
batched-motor optimization. No learning-performance gain is established by startup.

The first prepared f8c89b3 worker stopped in preflight before any VLA or robot
interaction: a test requested a 1.1M source allocation while leaving its total
campaign allowance at 1M. A bounded fail-fast allocation captured the rejection.
The test allowance was corrected while retaining the production budget guard;
the corrected suite passed. The 100-round test runs on job-local shared memory
to avoid tens of thousands of temporary journal writes to a slower filesystem.
Old attempts remain immutable and their scheduler time is metered separately.

[Protocol](PROTOCOL.md), [launch receipts](LAUNCH-AUDIT.json),
[parent manifest](parent-manifest-r2.json), and [first-failure traceback](preflight-failure/traceback.log).
The 20-minute monitor continues all three workers and their resumable quotas.
