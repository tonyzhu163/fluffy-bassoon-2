# Finish the existing second PPO cycle from checkpoint 1

The previous allocation completed one fresh collect/update cycle,15 Adam steps
on each of8 ranks, and the fixed16-case DEV panel. It stopped at its fixed worker
time limit while collecting cycle2. Checkpoint1 contains58 files/9,958,082,409
bytes and passed full cold readback and local staging. Its manifest is
`dc81e593f08fd630dd4206761229f9f1793338e7536c310b941f2cfa9d97ad2c`.

The proposed job restores that completed boundary on the same8-rank layout. It
loads native actor/Adam/scheduler DCP state, then all24 actor/rollout/environment
sidecars. Every actor must match its saved owned-parameter, optimizer-semantic,
scheduler, scaler and RNG fingerprint before further collection. Environment
samplers resume at reset ordinal4. Fresh scenes repeat the interrupted second
cycle; partial physics is discarded. Its original charged actions and unresolved
intents remain in a copied accounting ledger under the unchanged500,000 ceiling.
No closed run files are modified.

The output identity is `motusresume01`; code and model bytes stay read-only and
all new outputs go to job-local NVMe/project BOS. Shared train/eval lifecycle,
intermediate-render suppression, native offload, bounded metadata publication,
local safety monitoring and NCCL P2P settings remain unchanged. Native train/dev
batches stay32/2, actor microbatch256/accumulation1, global batch2048, five epochs,
Flow-SDE/GAE/losses, learning rates, seed groups and optimizer schedule remain
unchanged. There are15 new optimizer steps per rank, ending at30, followed by
checkpoint2 and the same16-case DEV panel. The native owned-state restore itself
is a runtime gate; CPU byte/tensor audits alone do not qualify GPU restoration.

Request: raise the cumulative cap from61 to66 GPU-hours. Closed conservative use
is215,849 GPU-seconds (59.9581 hours);3,751 seconds remain under the old cap.
Reserve21,600 GPU-seconds for this job:8 GPUs times a45-minute creation-to-release
watchdog, including startup and cleanup. Worst cumulative use is237,449 seconds,
below the proposed237,600 cap by151 seconds. This requests5 additional cumulative
GPU-hours and reserves6 job GPU-hours using the existing remainder. The worker
limit is40 minutes, scheduler43 minutes, retries disabled. Check live capacity,
actual devices, storage and write probes again immediately before launch.

The timing estimate uses the completed cycle: actor training took196.8 seconds
on rank0 and checkpoint publication/readback111.75 seconds. The preceding48-minute
worker also paid SFT initialization, checkpoint0, DEV0 and part of cycle2. A
40-minute worker allows one resumed cycle, but completion is still conditional on
actual simulator and storage throughput. Stop within the reservation if it does
not finish and preserve checkpoint1. Do not launch from the prepared contract
until the explicit cap increase is recorded.

Required evidence before launch: independent cold CPU audit of all58 checkpoint
files and24 worker tensors, source bindings and composed resume configuration;
per-rank action escrow sufficient for a complete new cycle; exact launcher
staging body checked against actual stored artifacts. Runtime completion requires
all24 worker restore receipts, eight exact actor fingerprints, all8 learner
reference checks,15 new updates/rank, checkpoint2 cold/local verification and DEV2.

This finishes a bounded restoration check only. Matched joint-Motus GRPO, sustained
source training, six-source/six-held-out transfer, chronological AD histories,
DPT labels and causal adaptation curves are still outside this job's results.
The base Motus model's previous task exposure remains disclosed. No learning,
adaptation, transfer or convergence claim follows from the16-case DEV prefix.
