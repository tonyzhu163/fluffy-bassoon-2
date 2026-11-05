Four-GPU native PPO actor profiling selected microbatch64 with eight accumulation steps and global batch2048. This maximizes measured actor throughput after accounting for the old-probability reference pass required when actor and rollout BF16 batch partitions differ. RLinf's learning rates, PPO loss, clipping, global batch and five epochs remain unchanged.

| Per-GPU microbatch | Accumulation | Median native update (s) | Estimated update including reference overhead (s) | Peak allocated (GB) | Peak reserved (GB) |
| --- | --- | --- | --- | --- | --- |
| **64** | **8** | **26.41** | **26.41** | **19.87** | **27.28** |
|128|4|25.14|29.83|31.26|41.04|
|256|2|25.47|30.13|54.05|69.04|
|512|1|35.83|42.72|94.57|100.97|

GB denotes decimal bytes/1e9. Each device exposed101.974GB (94.97GiB).512 completed but left only1.00GB outside the caching allocator's peak reservation;256 was the largest tested batch meeting the8GiB headroom criterion. Filling memory did not improve throughput.64 delivered77.54 global examples/s in the actor update. Its first saved-rollout replay error was zero on all four ranks.

Pinned RLinf recomputes reference logprobs once before five PPO epochs when `algorithm.recompute_logprobs=true`. Larger actor batches need that path because rollout per-rank batching is64 and different BF16 partitions change logprobs. The estimate adds one measured reference pass divided by five; the benchmark reference pass also requested values, so this is a conservative estimate rather than a directly measured full training iteration.64 matches rollout64 and does not need the extra pass.

Every candidate completed four real FSDP PPO/Adam updates on four ranks: one warmup and three steady updates. Inputs used saved scene-aligned source images/state and actual saved Gaussian chains, repeated to global2048. Advantages/returns were synthetic; updates operated on scratch weights and were discarded. Candidate order64,128,256,512 reused one actor group with `PYTORCH_ALLOC_CONF=expandable_segments:True`, native per-update cache cleanup and `NCCL_P2P_DISABLE=1`. These are actor measurements, not end-to-end simulator throughput or scientific PPO results.

Baige Pro6K job `job-jd3wvntcngt2` completed successfully at2026-11-05 19:12:23 CST. Source `99c0393b65d5f2db14bf838e37574582f21e19d1`; scheduler duration671s on four GPUs charged2684 GPU-seconds. Cumulative qualification/profiling charges are5.255 GPU-hours including0.711 legacy qualification; the current18 additional-GPU-hour envelope has used4.544. No scientific PPO optimizer update has occurred.

[Machine-readable selection](SELECTION.json), individual `micro-*/rank-*.json` records, [allocation probe](gpu-probe.json), and [scheduler receipt](../THROUGHPUT03-COMPLETE.jpg) retain the evidence. The full step0 recovery checkpoint and shared action journal remain protected. New PPO writes await combined shared-storage admission; see [execution state](../EXECUTION.json).
