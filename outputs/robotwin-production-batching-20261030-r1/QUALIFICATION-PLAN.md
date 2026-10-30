# Directly authorized packed PPO systems test

Human requested 70–80 GB use on a 96 GB GPU, asked about 12 processes as two
batches, and accepted the explained configuration: six independent simulator
processes for persistent PPO and six for separately trained reset PPO on one
exclusive GPU. This replaces the earlier unlaunched scalar/N2 comparison.

Each learner retains global batch six complete K4 trials, source-task native
600/500/400 cutoffs, H32 frozen motor, LR1e-6, five PPO epochs and original loss
weighting. Controllers and Adam are initialized fresh with seed20261030; this
is an isolated one-round qualification, never a replacement for either active
100-round production recovery or its trained iteration-10 snapshot.

| Arm | Trials | Maximum new source actions | PPO attempts | SGD steps if accepted |
| --- | ---: | ---: | ---: | ---: |
| Persistent PPO, six processes |6 K4|10,800|1|5|
| Reset PPO, six processes |6 K4|10,800|1|5|
| Total on one GPU |12 K4|21,600|2|10|

Maximum scheduler time is45minutes/2,700GPU-seconds, including loading, corpus
writes, updates, checkpointing and shutdown. Record this NEW reservation once;
the previous12k-action/1,800s qualification hold stays separate and unchanged.
No scalar reference or held-out/source-validation episodes are added. Native
reset policy-action bound remains zero; internal setup physics is disclosed.

One allocation probe runs before either worker. Both owned driver processes
then use the same allocated CUDA device, separate parent VLA copies, controllers,
simulator pools, corpora, ledgers and recoveries. Measure whole-device VRAM once
per second, retaining both parent identities. Target70–80GiB; interrupt only
owned parents above85GiB or90% ofactualdevicecapacity, allowing quarantine,
checkpointing and process cleanup. Actualhardware/totalbytes are attested, not
inferred from a marketing name. The measured result may leave more headroom;
process count is selected by throughput and memory, not memory occupancy alone.

Before launch, exact-pin Linux CPU tests must pass for native cutoff/feedback,
B1 controller replay, raw corpus preservation, intent-before-RPC, returned-count
resolution, lane failure, copy-safe campaign fences and coordinator acknowledgment.
A complete native group stays blocked until the pending rollouts, counters,
sampler/RNG and journal identity are durably present in the joint checkpoint.

Report source actions, complete histories, source updates, replay errors,
original optimizer gate outcome, corpus/physical-ledger consistency, allocated
GPU peak/utilization, reset/collection phases and whole-allocation walltime.
Collection-phase timing alone excludes replay/checkpoint acknowledgment. Inspect
representative nonblank RGB and retain pre-existing OIDN runtime errors.

Existing production jobs and their worktrees remain immutable. No automatic
scientific migration, global-batch expansion or fresh retry budget is authorized.
