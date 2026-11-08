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
| One fixed confirmation readout |3.333|12000GPU-s including startup/cleanup|
| Total additional envelope |50|Historical usage remains charged; fresh bank already qualified|

Reconciled closed use is99.906944GPUh after independent qualification of the
fresh64 bank; the remaining50h stage envelope would reach149.906944GPUh.
No GPU job is active. The earlier5h scene allowance is no longer a future
reservation; its actual within-cap costs are already included in the ledger.

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

## Implemented continuation and remaining admission

The first segment is implemented in `motus_curve.py`, `motus_curve_entry.py`
and `train_motus_curve01.sh`: checkpoint2→6, four fresh collections,60 new
Adam updates/rank, final counter90. The prepared contract reserves at most
20GPUh including startup/release. Its worst cumulative total is119.906944h,
which exceeds the approved100h and therefore remains unlaunchable.

Actual Linux CPU qualification at commit
`403920c2601842e524c6831bc862bb38f528befa` passes full checkpoint2 cold hashes,
all24 worker sidecars/eight strict owned-Adam descriptors, all48 original
checkpoint2 DEV reference files, all40 schedule rounds at four/eight ranks,
the exact256-group prefix and all1000 TRAIN scenes before repeats. Actor,
algorithm, rollout and environment sections match the original configuration.
The pinned native loop runs with RPC stubs and correct fresh-data tags/counters
for2→6 and6→10. This qualifies routing, not actual resumed SGD or the later
checkpoint6 input. [CPU evidence](MOTUS-CURVE-GRPO-CPU02-READBACK.json).

The original common prediction-offset assumption failed the first CPU check.
Native terminal-value calls make training prediction counts worker-dependent.
The corrected contract binds measured checkpoint2 DEV starts
`[83,79,78,82,79,78,80,80]`; all six files per worker must be eval-policy2,
nonterminal B2 predictions. The actual GPU baseline must then match original
observations, motor actions and both RNG states exactly before fresh collection.

After budget approval, perform the remaining storage/queue/device admission
above, pin the admitted commit and start only this first segment. Verify live
updates and preserve every complete cold boundary. Run the independent
`readback_motus_curve_training.py` on the closed new output before accepting
checkpoint6. The bounded launcher keeps two local checkpoints after verified
cold copies and preserves the imported source checkpoint separately.

The second2.5h allocation is a new contract, not an implicit retry: consume the
actual checkpoint6 manifest, original configuration and measured per-worker
final DEV references from segment1; independently qualify those inputs before
6→10. Keep total training allocation≤40GPUh. The paired readout now has
explicit fixed6/10 DEV support, as described below; qualify its actual future
inputs separately and retain its unchanged B2/noise/restore checks.
Qualify/predeclare the new confirmation bank before inspecting its policy
outcomes. None of these future inputs or full-model resumes is claimed passed
by the present CPU receipt.

## Fixed DEV readout preparation

`MOTUS-SOURCE-GRPO-CURVEDEV06-CONTRACT.json` and
`MOTUS-SOURCE-GRPO-CURVEDEV10-CONTRACT.json` are prepared contracts for the
existing64-case DEV bank. Each uses the original SFT checkpoint0 versus the
fixed endpoint, eight GPUs, native B2,128 total episodes, zero optimizer
updates and a12000GPU-second creation-to-release reservation. Actual endpoint
manifest hashes are deliberately unset; CPU qualification alone cannot admit
these jobs. Checkpoint10 remains the primary longer-curve endpoint.

Two provenance details are explicit in the implementation. SFT checkpoint0
retains the original two-cycle schedule, while the new trained checkpoint
uses the ten-cycle schedule. Only the read-only C0 path accepts the verified
256-group prefix; the native sampler/RNG restore is retained. SFT calibration
reads original SFT sampling files under `motusbehaviorgrpo01/state`, never the
continuation's initial checkpoint2 recordings. The readout clears implicit
training resume and restores only its two declared inputs.

Linux qualification at `7226b8dbf0089dd3a63513ca041bbb91909b5baf` passes both
endpoint schemas, actual pinned native ODE output/final-RNG references with a
deterministic suffix stub, paired initial draws, original selection guards,
all48 original SFT files, and the new source/actor restore audit against the
real original checkpoint0/2 GPU records and complete cold sources. These are
code/reference checks; no checkpoint6/10 model or simulator was constructed.
[Verified CPU receipt](MOTUS-CURVE-READOUT-CPU02.json). The updated independent
reader also passes all1857 files, eight physical/noise/outcome panels and actor
before/after restore invariants on the completed old checkpoint1 DEV run. Its
recorded6→10/64,p=.125 result is unchanged; this is compatibility evidence, not
a new learning experiment. [Reader receipt](MOTUS-CURVE-READOUT-LEGACY01.json).

After the appropriate segment finishes:

1. Use `readback_motus_curve_training.py` with its actual training commit and
   a fresh `<training-identity>-readback01` output. Require its full cold
   boundaries/owned Adam checks and all fresh-collection NumPy references.
2. Populate the matching prepared DEV contract from that verified result,
   including actual endpoint manifest/configuration and training-readback hash.
   Run `qualify_motus_curve_readout_cpu.py --contract <contract-basename>
   --identity cpumotuscurvereadoutNN` with a fresh numeric suffix on Linux
   with CUDA hidden. This
   mode requires the real endpoint, all24 worker sidecars at each input,
   counters90/150, unchanged numerical settings and original SFT references.
3. Record that actual-input receipt separately from the present code/schema
   receipt. Reconcile the current ledger, admit within the authorized
   cap, commit on main and deploy the exact commit through Git. The launcher
   rejects missing input qualification, training verification or budget approval.
4. Run `readback_motus_source_readout.py <evaluation-identity>` on Linux after
   release. It checks every output file, physical transition and actual paired
   Gaussian, plus all worker restores and actor before/after signatures. New
   endpoints additionally require their complete cold source checkpoints.

The present extension rejects checkpoint6/10 confirmation on the consumed
bank. The explicit fresh-confirmation route binds the qualified,
predeclared64-scene manifest and retain the existing native sampler inventory
for checkpoint restoration; choose the new scenes through the explicit readout
reset hook. Changing the native inventory would fail exact sampler restoration.
Qualify that physical path before policy evaluation. The fresh64 bank now has independent expert/reset/device qualification. The
explicit fresh-bank route is now Linux CPU-qualified; actual C10 inputs and
the fixed checkpoint10 GPU confirmation remain outstanding.

## Qualified fresh confirmation bank and remaining route

`MOTUS-FRESH-CONFIRMATION-BANK.json` is byte-identical to the independently
verified Linux artifact (SHA256
`76518cff54b658f299f60ecbcaae914a252566674b468449d7e43aed288f1283`).
It contains the first64 valid scenes from84 ordered candidates100100242
through100100325, excluding every native1000TRAIN and150oldEVAL seed.
All20 expert rejections remain in the raw qualification. The bounded scan04
prefix of83 candidates/63valid is audited and copied unchanged; scan05 adds
only the next successful scene. Native cuRobo expert success, exact physical
seed/reset, all three cameras, allocation binding and cleanup pass. There were
zero policy calls or optimizer updates. Original sampler inventories remain
unchanged. [Closeout and independent hashes](MOTUS-FRESH-SCENES05-CLOSEOUT.json).

`MOTUS-FRESH-C10-CONFIRMATION-PROTOCOL.json` predeclares this bank, C0 versus
fixedC10,128 policy episodes, native batch2, four panels/eight ranks,
31000+100*panel+rank common RNG and the unchanged effectiveness gate. It is
not a launch contract. The explicit new-bank route is now implemented and
Linux CPU-qualified at7f09bde7; it preserves the prohibition on consumed
confirmation. See `MOTUS-FRESH-READOUT-CPU01.json` and the prepared
`MOTUS-SOURCE-GRPO-FRESHCONFIRMATION01-CONTRACT.json`. Retain exact full C0/C10
restores and the separate original SFT calibration. Fresh scenes cannot be
required to match old DEV images/actions. Verify paired physical states and
actual initial Gaussian tensors across policies before interpreting outcomes.

The expert backend is isolated cuRobo0.7.8 with native table-only collisions;
only run-local stale asset-file locations are repaired. Earlier MPLib controls
failed and remain preserved in their receipts. Expert restoration does not
explain the completed weak learning result: joint-action VLA execution does
not call this expert planner. No collision relaxation, held-out task training,
reward change or policy-dependent scene selection is authorized by this plan.


The CPU qualification uses192 reset-argument references and384 calls through
the pinned ODE loop with a deterministic suffix stub; all192 paired draws,
outputs and final CPU RNG match. It also checks the actual fresh-bank proof,
all48 original SFT files, existing C0/C2 cold restores and DEV6/10 schema.
No actual future C10 checkpoint, fresh policy episode or learning gain is
qualified by this receipt.

After C10 training completes, bind its actual manifest SHA and independently
verified training readback in the prepared fresh-confirmation contract. On the
allocated Linux CPU, run `qualify_motus_fresh_readout_cpu.py --identity
cpumotusfreshreadout02 --actual-inputs` with a new unused output identity and
the pinned runtime. Bind that successful actual-input receipt separately from
the code/schema receipt. Reconcile remaining reservations, storage and queue;
only then admit the contract and deploy its exact commit through Git.

The GPU launch uses `motus_source_readout.sh <commit>
<project-BOS>/motus-source-grpo-freshconfirmation01-signals
MOTUS-SOURCE-GRPO-FRESHCONFIRMATION01-CONTRACT.json`. After closure, run
`readback_motus_source_readout.py motus-source-grpo-freshconfirmation01`
on Linux CPU. The reader verifies every cold file,24worker restores/policy,
strict owned Adam/RNG invariants, actual physical scene bindings, all trace
transitions and actual paired initial Gaussian tensors before reducing the
fixed64 paired outcomes. Report the predeclared gate unchanged. Neither code
qualification nor budget approval alone admits missing C10 inputs.

The new independent reader also passes the complete existing C1 DEV evidence:
1857 cold files,128 episodes,128 actual scene-binding records and all384 actual
Gaussian draws. It reproduces the previous6→10/64 result and p=.125 exactly;
see `MOTUS-FRESH-READOUT-LEGACY01.json`. This compatibility check adds no GPU
cost and does not consume the fresh confirmation bank.
