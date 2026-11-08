# Proposed next source-learner experiment

Status: prepared scientific/resource proposal; not launch-admitted and not funded
above the current100 GPU-hour cumulative cap. This file does not authorize a job.

## Question and scope

Does continued fresh-data binary-reward GRPO improve the joint Motus phone-task
policy beyond the weak two-cycle result? Continue the existing validated GRPO
checkpoint2 to a fixed checkpoint10, retaining source-task scope and the
untrained held-out tasks. Choose GRPO because it won the predeclared method DEV
selection, while retaining its negative checkpoint2 confirmation as a negative
result. The checkpoint1 diagnostic is exploratory and does not reselect the
starting checkpoint or erase that confirmation.

Eight additional fresh1024-episode collections add8192 episodes and120 Adam
updates per rank. Final optimizer counter must be150. Every collection uses
128 scene groups×8 rollouts with independent native sampling, native binary
reward, zero reward filtering, no critic and the pinned GRPO sample-standard-
deviation formula. Preserve trainable action expert/frozen VLM, H32/192,
Flow-SDE0.3/five steps, actor LR5e-6, global2048/five epochs and native
micro32/accum8/world8. No simultaneous reward, noise, learning-rate, horizon,
camera or critic change is proposed. This is a longer Motus port, not an exact
released RLinf `adjust_bottle` reproduction.

Use the original1000-scene TRAIN bank only. Extend the recorded two-iteration
schedule with a deterministic committed group schedule; retain its prefix
exactly and rotate through the source bank before repeating scenes. Fresh
policy rollouts are required even when scene IDs recur. Preserve all optimizer,
sampler and RNG state on continuation. Record the explicit migration of the
schedule and limits; never modify the original active/closed run worktree.

## Measurements and stopping

- Complete cold checkpoints and16-scene diagnostic panels at every iteration.
  These panels show a curve; they do not select a new effectiveness endpoint.
- Independently verify original behavior likelihoods, GRPO advantages/tied
  groups, finite gradients and frozen parameters every iteration. Stop on an
  invariant failure. Preserve the latest fully verified resumable boundary.
- Paired64-scene DEV readouts at fixed checkpoints6 and10 versus the original
  SFT, retaining native B2 and actual shared Gaussian draws. A decline remains
  reportable; do not change the checkpoint10 primary endpoint based on DEV.
- Retire the already consumed confirmation bank. Qualify and predeclare64
  new source-task scenes with no overlap with TRAIN, existing DEV or consumed
  confirmation before inspecting policy outcomes on them. Evaluate SFT versus
  fixed checkpoint10 once. Keep the same≥10-point and exact-p<.05 gate.
  If a valid new bank cannot be qualified within the allowance, report DEV only
  and leave effectiveness unresolved; never relabel reused scenes as fresh.
- This one training-seed continuation cannot establish cross-seed stability.
  More student/held-out training remains paused until the source gate passes.

## Resource envelope and why approval is needed

Corrected GRPO's two-cycle job used32880 conservative GPU-seconds (9.133h),
including initialization, all three checkpoints and small DEV panels. Eight
additional cycles project approximately36.5GPUh by linear extrapolation; this
is an estimate, not a guaranteed runtime. The following bounded envelope covers
that estimate with startup/cleanup margin and independent endpoint readouts:

| Stage | Maximum reserved GPU-hours | Bound |
| --- | --- | --- |
| Eight new GRPO collections, possibly split only at verified boundaries |40|8 GPUs; total creation-to-release allocation across training jobs≤5h|
| Two fixed DEV readouts |6.667|2×12000GPU-s including startup/cleanup|
| New source-scene qualification |5|Physics/planner validity only; predeclare the bank before policy outcomes|
| One fixed confirmation readout |3.333|12000GPU-s including startup/cleanup|
| Total additional envelope |55|Historical usage remains charged|

Reconciled current closed use is98.943889GPUh; the full55h envelope would
reach153.943889GPUh. No GPU jobs remain active.

Proposed cumulative cap: **155 GPU-hours**, up from100. Final admission must
use the reconciled current ledger and preserve the total155h hard cap.
Reserving stages is not permission to overrun any job's declared watchdog.
Keep scene qualification and the confirmation reservation separate. If any
complete stage cannot fit after reconciliation, stop before allocating it. CPU
qualification costs no GPUh.

## Required engineering before launch admission

This is a concrete proposal, not a claim that the existing two-cycle launcher
already supports ten cycles. Before any accelerator launch:

1. Generalize run-scoped schedule/boundary validation and the complete training
   readback from two to ten iterations, preserving the old two-cycle contracts.
   Qualify every schedule rank/reset and the unchanged native arithmetic on
   allocated Linux CPU. Verify checkpoint2 manifests and all state descriptors.
2. Audit native runner resumed counters, trace ordinals, action escrow, constant
   scheduler and restored RNG. Ensure the first new collection belongs to policy2
   and no stale saved batch is substituted for fresh experience.
3. Check actual BOS capacity/quota/write/read probe. Stage code/checkpoints and
   caches on project NVMe; retain latest two complete local checkpoints after
   verified cold copies to bound staging. Never move active environments/caches.
   Set explicit cold output cap from measured per-iteration data size.
4. Review latest main, commit qualified code/contracts and deploy through Git.
   Recheck queue capacity and actual devices. Bind render/inference to allocated
   GPUs, retain scheduler visibility and job-local NCCL_P2P_DISABLE=1.
5. Admit each job only after the current ledger plus all outstanding reservations
   fits the approved cap; disable automatic retries and set worker/scheduler/
   creation-to-release limits. Verify advancing updates, then monitor≤20min.

No additional GPU allocation is launched by this proposal. Approval of the
155h cumulative envelope would permit the above source-task work, after these
checks, without repeated per-job launch questions. It would not approve new
held-out-task training, a new reward objective or student experiments.
