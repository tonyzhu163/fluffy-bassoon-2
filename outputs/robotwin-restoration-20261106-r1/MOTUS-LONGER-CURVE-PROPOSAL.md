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

Reconciled current closed use is99.308889GPUh after the bounded scene diagnostics;
the full55h future envelope would reach154.308889GPUh. No GPU jobs remain active.

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
20GPUh including startup/release. Its worst cumulative total is119.308889h,
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
bank. A future fresh-confirmation implementation must bind a newly qualified,
predeclared64-scene manifest and retain the existing native sampler inventory
for checkpoint restoration; choose the new scenes through the explicit readout
reset hook. Changing the native inventory would fail exact sampler restoration.
Qualify that physical path before policy evaluation. The new bank, its GPU
qualification and the fixed checkpoint10 confirmation remain outstanding.

## Fresh confirmation scene preparation

The candidate scan is now committed in `MOTUS-FRESH-CONFIRMATION-SCAN.json`.
It excludes all1000 native TRAIN seeds and all150 native EVAL seeds, including
the22 unused seeds after the old128-case split. Scan candidate integers
100100242 through100101265 in ascending order and stop at the first64 valid
scenes. These numbers continue the source task's EVAL seed range; they are
candidates, not a claim of already valid scenes. Record every attempted seed
and rejection in the contiguous prefix. No VLA, rewarder, policy success or
checkpoint-dependent signal may enter selection.

`motus_fresh_bank.py` calls the unchanged pinned native `check_seeds`: scene
setup, expert `play_once`, `plan_success` and `check_success`. It additionally
requires the expert's actual physical seed and a fresh inference-style reset
to equal the candidate. Validate finite state, each of three camera images,
instruction/scene metadata and allocated renderer PCI. Abort on a reset or
identity mismatch. A planner's ordinary invalid candidate is recorded and the
scan advances; no substitute seed or incomplete bank is published. Keep the
native success-seed inventory intact. Publish `BANK.json` only after64 valid
scenes and joined/synchronized/offloaded simulator cleanup. The independent
`--readback` reconstructs the exact selection from the full ordered records.

`MOTUS-FRESH-SCENES01-CONTRACT.json` uses a narrower first attempt which fits
the existing100h authorization: one GPU for at most3600 creation-to-release
seconds,3100 worker seconds,57 scheduler minutes, no automatic retries,8GiB
local staging and64MiB BOS output. Historical356198 plus3600 reserved equals
359798GPU-seconds (99.943889h), below360000. The original proposal retains a
5GPUh total scene allowance only if the155h cumulative extension is approved;
this first attempt does not authorize spending the other4h. If fewer than64
scenes qualify before the watchdog, preserve the records and publish no bank.

Before this within-cap attempt, review latest main and qualify/pin its source files. Bind the
actual original configuration SHA and verified Linux CPU receipt in the
contract, retain all existing TRAIN/EVAL inventories, and stage its unchanged
configuration and ordinary asset links on project/job NVMe. Check storage
capacity/quota/write-read and actual queue resources, create fresh signal and
output identities, and use the standard allocated `gpu_probe.py` evaluation
check with expected-count1. Set `LYCO_BAIGE_GPU_MAP` to that actual job's probe;
the collector verifies allocation mapping and renderer identity. Run
`motus_fresh_bank.py --collect <admitted-contract>` under the declared worker,
scheduler and creation-to-release watchdogs. Check progress and monitor at
most20 minutes apart. Do not create another allocation to compensate for
invalid scenes or timeout without reconciling budget and preserving evidence.

If collection passes, run its `--readback <actual-output-directory>` on Linux
CPU and commit the verified manifest/hash before either SFT or C10 policy is
evaluated on it. The later confirmation readout needs an explicit new-bank
flag and that verified manifest; it must keep old confirmation guards closed,
restore complete C0/C10 state with unchanged sampler inventories, and use the
explicit physical reset hook. It must not assume fresh scenes reproduce old
DEV SFT pixels/actions. Shared B2 ODE/noise, full restores, paired outcome
reader and the original effectiveness gate remain mandatory. Actual new
scene qualification and that input-specific readout are still GPU work.


The first visibility-guard attempt charged61GPU-seconds. The corrected second
attempt passed its actual one-GPU probe, constructed48 exact requested scenes,
and recorded48 failed expert checks. Stop/release at11:27:36Shanghai charged
618GPU-seconds; no bank, VLA call or optimizer update was produced. Closed
usage is now356877GPU-seconds (99.132500h), leaving3123 seconds under100h. All
raw records/logs and their hashes are preserved in
`MOTUS-FRESH-SCENES02-CLOSEOUT.json`. The saved intermediate RESULT retains
its last `collecting` status; the scheduler stop receipt governs the outcome.

MPLib reports invalid-start self/table collisions. Source inspection shows its
wrapper passes only arm DOFs to an articulation planner, whose installed padding
method fills omitted DOFs from its internal articulation state. A full measured
qpos repair is isolated in `motus_planner_full_state.py`; its Linux numerical
reference passed; subsequent paired GPU controls remain negative. This is a hypothesis about the failed qualification,
not a demonstrated cause of the completed PPO/GRPO learning result. cuRobo is
not installed in this runtime. Do not relax collisions or call these rejected
scenes a valid confirmation bank.

Before another scene scan, require Linux qualification of the actual installed
padding/wrapper functions and GPU positive controls on predeclared old native
validity seeds. Record both original and corrected expert behavior, actual joint
vectors and collision diagnostics; verify unchanged initial physical state and
allocated renderer. Stop early on a failed control. Only then scan the original
fresh candidates in order, with no policy outcomes and no sampler-inventory edit.
At that boundary, a later GPU contract could reserve at most3123 seconds
and use a fresh identity. No new reservation is active. Longer source training
continues to require the pending cumulative-cap extension.

The paired old-seed planner control completed at12:07:17, charging265GPU-s.
Native and full-state each fail all4 expert episodes, with identical initial
observations and38-DOF articulation traces. All88 full-state planner calls
retain exact measured non-arm state; the native6-DOF route does not. The repair
alone is insufficient:84 individual full-state solver calls succeed and4 IK
calls fail, while every whole expert episode fails. No further scene scan is
admitted. Closed357142GPU-s leaves2858 under100h. See
`MOTUS-PLANNER-CONTROL01-CLOSEOUT.json`. Inspect actual planning-world
synchronization and task failure stages before allocating another diagnostic.

Subsequent world-sync control charged186GPU-s and failed before solving
because the phone was absent from the early planning-world snapshot. Native
post-setup reconstruction charged184GPU-s and failed before completing because
scene-to-FCL conversion rejects task mesh scale0.078. Both have paired initial
observations and independent readbacks; neither published a bank or used
VLA/SGD. Closed357512GPU-s leaves2488 seconds, with no active reservation.
See `MOTUS-PLANNER-CONTROL02-CLOSEOUT.json` and
`MOTUS-PLANNER-CONTROL03-CLOSEOUT.json`. Qualify actual mesh-scale geometry on
Linux CPU before another bounded GPU positive control. These bridge failures
remain separate from the completed weak PPO/GRPO learning experiment.

The opt-in scaled-convex converter now passes actual CPU PhysX/FCL geometry
references at040b3202:9 analytic distance cases and18 collision checks, with
exact native unit equivalence and maximum analytic error3.505e-6. See
`MOTUS-SCALED-MESH-CPU03.json`; no task expert or triangle-mesh qualification
is implied. Before another allocation, reconcile collision scope with the
recorded native cuRobo/static-world behavior: adding all task objects changes
grasp constraints. Qualify intended contact semantics, complete geometry and
actual four-seed expert success before scanning fresh candidates. Restore the
native backend in an isolated uv environment if needed; preserve shared active
environments. No further GPU contract or fresh-bank scan is admitted yet.

Isolated native-backend preparation is now recorded in
`MOTUS-CUROBO-CPU03.json`: cuRobo0.7.8 atd64c4b0 builds successfully and
all5 extensions pass CPU ABI imports. Full MotionGen import requires a visible
CUDA device, so it remains unqualified. Run-local copies repair only two stale
URDF paths and verify all30 mesh files; shared assets/runtime stay unchanged.
Native table-only collision semantics and paired expert success must pass an
actual bounded device control before scene selection. These CPU checks add no
GPU cost:357512 seconds closed,2488 remaining. They do not change the training
recipe, its negative confirmation, or the pending155h request.
