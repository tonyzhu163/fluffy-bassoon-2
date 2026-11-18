# Stage 1: OpenDrawer fixed VLM-critic PPO versus critic-free GRPO

Requested by the human on 2026-11-04, including pushing verified results to
`main`. This executes only Stage 1 of the
[pipeline plan](../robocasa-ppo-grpo-meta-pipeline-20261104-r1/PLAN.md).
No Stage 2 source campaign or student fitting is included.

Both arms start from the original Human-300 SFT weights on the qualified
native environment and patched RLinf commit
`8f9bfa7661928ef8c17494b04f4d7bf3125634e1`. Use a clean, verified latest-main
source checkout. The preflight source is
`ad918c9f57f132838e54afa79d4c14474c931b0a`; record the actual launch pin.

Each arm uses eight allocated GPUs, 168 CPUs, 1,280 GiB RAM and 32 GiB SHM,
128 training episodes per iteration, horizon 600, action chunk 50, 40
iterations, one optimizer epoch, noise 0.3 and `ignore_last=true`.
PPO has a VLM-prefix value head and group size 1. GRPO has no value head
and uses 16 scenes with eight attempts each. Use the same fixed DEV64 bank
at steps 0/10/20/30/40; save complete checkpoints at 0/1/10/20/30/40.
The first-update checkpoint is produced by the existing runner.

Each arm plans 3,072,000 TRAIN controls, 192,000 DEV controls and 696,320
reset-reserve controls, totalling 3,960,320. A separate global action escrow
of 4,100,000 applies to each arm. Keep full sampling, raw environment and
actor-batch records. No duplicate-field drop is enabled for this pilot.

The concrete compute reservation is **at most 64 GPU-hours total**:
two eight-GPU jobs with four-hour wall limits, including startup,
qualification and evaluation. The human explicitly approved this ceiling in this chat on 2026-11-04.
No automatic retries or optional four-epoch ablation are included.
The approval covers Stage 1 only. CPU preflight uses
the existing allocated Linux development container with CUDA hidden.

At preflight the shared PFS has about 515 GiB available. Prefer serial arms
to control storage occupancy. After an arm exits and its exact scheduler
allocation is released, archive its closed bulk artifacts to the existing
BOS mount, verify archive contents and hashes before reclaiming PFS, and
retain its latest complete resumable checkpoint locally. Never reclaim
another run's data. Keep the existing 200 GiB PFS floor and checkpoint-exit
guard; recheck capacity before each submission. If the floor prevents
completion, preserve the boundary and report the storage condition before
an explicitly recorded continuation with a new attempt identity.

Inside each actual job, require the existing eight-device GPU probe and
model/normalization hash checks. Verify iteration-1 likelihood parity
(and PPO value parity), finite losses, unchanged frozen VLM, and GRPO's
identical physical/public reset groups. Verify advancing updates after
launch, then monitor every 20 minutes. Record cluster, job IDs, last check,
progress, next action, measured update time and cumulative GPU usage.

Report the complete TRAIN and DEV64 curves, KL and clipping diagnostics,
PPO explained variance, all boundary/evaluation bindings and action/cost
receipts. Apply the original Stage 1 gate: parity <=1e-3, all 40 updates
finite with the VLM unchanged, approximate KL <=0.03 without saturated
clipping, and >=10 percentage-point upward TRAIN/paired-DEV learning
without collapse. A failed gate remains a result; it does not authorize
changing hyperparameters or launching Stage 2. Push compact evidence and
the verified result to `main`; checkpoints, videos and raw trajectories
remain on cluster storage with hash/path references.
