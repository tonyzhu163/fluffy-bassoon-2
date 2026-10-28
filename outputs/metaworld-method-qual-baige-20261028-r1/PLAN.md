# ML10 method qualification, 2026-10-28

User-approved scope: CTM T1, RL²-PPO, ECET and AMAGO1; one seed; one RTX6000-class allocation per policy; common10M training transitions and10PPOepochs, AMAGO10sampled replay passes explicitly accepted. This qualification measures runtime/capacity before the full80-round runs.

Four isolated one-GPU jobs,16CPU/64GiB each,45-minute hard cap each; no retries without diagnosis. Source is the exact committed SHA passed to the launcher, recorded in each config and scheduler name. Reuse the clean detached chat worktree for editing; deploy a new immutable worktree through Git bundle. No tests/model computations on macOS or Baige development container. Baseline overlay environment is separate and borrows qualified heavyweight libraries read-only.

Each job probes its allocated GPU and records environment pins. CTM runs common CPU/CUDA/simulator/recovery/parity tests then one50env/K5H500,10epoch,cached100 update. RL² tests exactCPU/CUDA recovery and fusedGRU parity then the same real collection/epoch count. ECET verifies its pinned official encoder, real10family H500 replay, then a fullnative50env update. AMAGO runs upstream identity, cache/reference/recovery tests, real simulator and defaultB24L2000 one-update capacity/timing. Throughput-only cells omit evaluation and cannot be called learning curves or resumed into training.

Inspect finite metrics, replay errors, optimizer/sample counts, memory, checkpoint hashes and costs. CTM collector phase timers split inference/action transfer, simulator stepping and observation/reset bookkeeping. Optional parallel simulation must pass seed/trajectory parity before use. Preserve failed receipts and previous checkpoints. Mirror small evidence and update only this work's live coordination row.

Hardware confirmed by the user: use one NVIDIA RPBZZZ6/96GiB per policy in the existingBaigequeue. Parallel simulation is explicitly requested: qualify exactseed/trajectory parity, then use8persistentCPUworkers within16allocatedCPUs. Default-shaped timing uses this backend after parity passes.
