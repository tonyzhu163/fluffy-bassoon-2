# Stage 1 utilization and timing check

2026-11-04, approximately 16:24 Shanghai. PPO job `job-bylh99ylx9ie`,
immutable source `d9a1c616`; seven completed updates at measurement.

The Baige column shown to the human is **GPU average utilization** (16.67%
observed in its task table), with eight GPUs allocated. Its page had automatic
refresh off; its displayed aggregation window is not established. The job-local
15-second nvidia-smi samples over the latest ten minutes average 8.52%, with
55.94% of sampled device readings at zero. These sampled means are not an
integrated throughput or cost metric. The trace shows bursts on all rollout
GPUs, then SGD on GPUs 0 and 1 while the other six wait.

Median completed-update interval over updates 2 through 7 is 168.85 seconds,
0.3752 allocated GPU-hours for 76,800 controls (128 episodes x 600). The
previous Flow-SDE PPO reference measured 257.32 seconds, 0.5718 GPU-hours for
the same controls on eight GPUs. The observed reduction is 34.38% per rollout
batch, but **this is not a matched implementation-speed comparison**: Stage 1
uses one optimizer epoch / six Adam steps, while the historical recipe used
four epochs / 24 steps. Current actor optimization takes about 34.06 seconds
versus the historical roughly 115 seconds. The saved Stage 1 plan explicitly
requested the epoch reduction to test stability; it is not yet established as
optimal after correcting the critic.

The latest native timing rows show roughly 131-137 seconds generating rollouts,
including environment interaction, within roughly 172-second iterations;
rollout/predict is around 15-16 seconds and model sync around 1.3 seconds.
These are existing nested/worker timing metrics and must not be added as
independent exclusive phases. They support substantial time outside GPU model
execution; the exact split between simulator resets/stepping, serialization,
I/O and synchronization has not been isolated. Low utilization is a real
execution-efficiency concern even though the changed recipe is cheaper per
collected batch. No active contract, GPU count or batch schedule was changed
in response to this read-only check.

Evidence: [measured intervals and sampled utilization](PERFORMANCE-20261104-1624.json),
[historical timing](../robocasa-flow-sde-10m-per-task-20261103-r1/PERFORMANCE-NOTE.md),
[Stage 1 epoch specification](../robocasa-ppo-grpo-meta-pipeline-20261104-r1/PLAN.md).
