# RoboTwin restoration and shared in-context learner

Owner: chat `01a1b024-f4ac-7181-af87-bac9fbee72bb`. Started 2026-11-06,
Asia/Shanghai. This is the execution plan for the user's two pasted notes and
the requirement to carry RoboCasa engineering into **both training and eval**.
Record changing state in `STATUS.json` and only this owner's coordination row.
Keep earlier runs, sources, checkpoints, reservations and human holds intact.

## Intended result and decisions

Restore a qualified RoboTwin training/evaluation path; collect independent
single-task PPO and GRPO experience from one common joint SFT initializer;
train a shared causal history transformer; measure whether prior experience
improves a frozen deployment policy without test-time gradients. Simulator
reward is the initial RoboTwin arm. Flora requires separate reward qualification.

Decisions already resolvable without another approval:

* Use the existing six source/six held-out split in `configs/robotwin_vlrla.json`.
  Source: handover_mic, pick_diverse_bottles, place_a2b_left,
  place_object_stand, place_phone_stand, scan_object. No held-out histories,
  normalization, labels, gradients or checkpoint selection.
* Use **one joint SFT base for all tasks in any comparison**. Each task and
  algorithm gets an independent teacher initialized from that exact base;
  teacher weights may change, deployment base weights may not. “Teacher
  branch” means a separate lineage/output, not a long-lived Git branch.
* Motus joint pi0.5 is the intended new base from the supplied notes; retain
  the already qualified Sidney H50 path to close existing work and as a
  separately named reference. Do not mix these motors in one dataset/claim.
* PPO is the primary source algorithm; GRPO is a matched comparison. Preserve
  useful mixed-quality data even when average teacher success is flat. Only
  an AD claim of distilling improvement requires improving teacher traces.
  DIT and offline value learning require informative rewards/coverage instead.
* First controlled objective comparison: action residual interface, same
  history architecture/data/bounds for AD, DIT-style and offline IQL-style
  actor-critic. Add DPT only with qualified query-state teacher labels.
  Retain the existing learned-KV bridge as an interface comparison; do not
  conflate interface and objective changes. These user notes refine the prior
  bridge-first route for this new RoboTwin run only.
* Primary execution profile fills available device memory with **useful real
  execution batches**, retaining measured operating reserve. Also report the
  fastest measured profile; high memory occupation alone is not a speed result.
  Keep the optimizer's global batch, epochs, loss weighting and schedule fixed.

## Facts to start from

1. RLinf's recorded pi0.5 single-task recipe is available in
   `outputs/robotwin-rl-baselines-20261105-r1/UPSTREAM.json`, pinned upstream
   `77a521298f1d215971246c2d7a423720d9e76652`. Its live main file was checked
   again through the GitHub connector on Nov 6 and agrees on the fields below.
   Qualified local runtime: RLinf c605b9d55ed93a180119b099e6f836abff293b08,
   RoboTwin dca9ec682688821c42944eacd7f5dd9bfb15396f. Updating upstream requires
   an explicit migration; a latest config is not a qualified installed runtime.
2. Sidney matched PPO/GRPO already completed two iterations and 40 Adam steps
   per rank. Read `outputs/robotwin-sidney-matched-baselines-20261106-r1/RESULT.md`.
   Training collections at policy steps 0/20 are not final-policy evaluation.
   Both final complete actor/optimizer restores passed. Preserve these files.
3. Existing controller/prompt diagnosis and final-policy export/readout are
   owned by the re-entry work. Read their current EXECUTION/CONTRACT/CLOSEOUT
   and scheduler state before any new launch. Export01 failed before model
   load because validate_cfg created an output folder before exclusive mkdir;
   export02 has a prepared correction. Do not duplicate or edit a running job.
4. The Motus model card declares native **H32**, adapt_to_pi and delta arm
   joints, three cameras, 50 tasks/27,500 clean+randomized demonstrations.
   The repository already has Motus transforms and a pinned revision in
   `src/fluffy_bassoon/rl/robotwin_noise.py`; reuse and requalify those instead
   of inventing another adapter. Sidney uses H50, absolute joints, mean/std,
   and checkpoint-specific activations/precision. Neither adapter substitutes
   for the other. Audit all twelve tasks' base SFT exposure.
5. The native fast RoboTwin controller plans/interpolates an entire chunk.
   Calling take_action once per command is a different controller, even with
   the same targets. Requested targets, completed target boundaries, partial
   targets and physics integration steps are different measurements. Existing
   fast training lacks measured execution masks; it is not yet student data.

## Reference PPO and GRPO math

The preserved RLinf pi0.5 reference uses 256 environments, four collection
rounds, H200 episode horizon, H50 predicted/executed chunks, five Flow-SDE
steps, noise .3, global minibatch 2048, five update epochs, per-GPU micro32,
gamma .99, lambda .95, actor LR 5e-6, critic LR 1e-4, Adam (.9,.95), eps1e-8,
weight decay .01, grad clip1, value clip .2, Huber delta10, zero critic warmup,
zero entropy/KL bonus, normalized advantages and chunk-level likelihood/reward.
It is task-specific adjust_bottle SFT in the release. The new joint initializer
and source tasks are explicit changes, not a reproduction of its success score.

For PPO, use native RLinf GAE and actor-critic loss. With the exact native
termination/timeout bootstrap convention, delta_t = r_t + gamma*b_t*V(next)
- V(current), and A_t = delta_t + gamma*lambda*c_t*A_next. b/c encode the
separately qualified bootstrap and continuation masks; do not infer both from
one combined done. rho_t = exp(logp_theta-logp_old), policy objective
mean(min(rho*A, clip(rho,1-.2,1+.2)*A)), plus native clipped value/Huber loss.
Use RLinf's actual reduction, action/denoising masks and normalization, with an
independent numerical reference. Verify chosen-step value replay and frozen
vision train/eval behavior; do not blindly apply a different RLinf patch.

For pi0.5 GRPO, reuse the same Flow-SDE likelihood and chunk representation;
change advantage to native grouped trajectory score and loss to actor-only,
remove the critic. Group size8, identical initial physical scene and prompt
within each group, independent declared policy randomness. For a single
episode per lane, R_i is the native masked sum of rewards (the qualified native
GRPO score does not discount it); A_i = (R_i-mean_group(R))/(sample_std_group(R)
+1e-6), broadcast over valid decisions. std uses denominator G-1. All-tied
groups give exactly zero finite advantage. With multiple episodes in one lane,
use the native done-reset return recurrence, not a naive whole-buffer sum.
Audit later normalize_advantages and reduction as well as this group formula.
Record tied groups, reward spread, finite gradients, KL, clip fraction and cost.

The shipped OpenVLA-OFT GRPO YAML uses other rewards/clips/batches. It is a
formula/implementation reference, not a pi0.5 reproduction. For the controlled
pi0.5 comparison keep the PPO reference's reward scale, .2 clips, global2048,
five epochs and optimizer values; group8 is used for both methods so scene
replication does not confound the optimizer comparison. Report PPO group1 as
the separate released-layout reference. No reward filtering in the matched
comparison; retain tied-group failures as context. Treat changing filtering,
reward shaping or gamma as separate conditions.

Two named protocols avoid an H mismatch:

* `rlinf_joint_h50`: qualified Sidney H50, H200, remaining reference fields.
  The prior Sidney H400 pilot stays a separately named VLRLA-style protocol.
* `motus_native_h32`: Motus H32/delta transforms and the same declared PPO
  optimizer/formula. Start with H192 (six full decisions), explicitly changing
  the H200 reference. A true H200 version requires qualified eight-action last
  chunk, masks, timeout and likelihood handling; never silently run 224 actions
  or change a positional embedding from32 to50. Do not compare H192 with H200
  as action-budget matched. All paired Motus methods use the same H192.

## Phase 0 — recover access and reconcile existing work

Read docs/README, PAPER, FINDINGS, FLORA-PPO, METHODS, robotwin baselines,
coordination and the exact prior contracts. On macOS use the configured proxy
function in the same interactive shell before Git. If absent, recover the
documented local route; do not commit routing/authentication or invent a proxy.
Use configured SSH or an existing authenticated Baige development browser tab.
No browser JavaScript/security preference changes are needed by this plan.

Fetch/review latest main, preserve other owners' dirty edits and deploy a
verified commit into a fresh detached source. Read actual queue capacity,
limits, GPU model/VRAM, current jobs and ownership. Reconcile old 50-GPU-hour
cumulative cap including failures and still-live reservations **before** using
remaining compute; a new folder does not reset that cap. Prefer finishing the
existing endpoint/controller diagnostics if still pending. Do not stop another
owner's job, modify its source, reuse its run ID, or consume its reservation.

Storage: project NVMe code/env/cache/output roots, quota/capacity and write+fsync
probe. Reconcile live future growth with the latest shared floor rather than
copying a stale free-space number. Retain original data, normalization, initial/
selected/final weights and complete latest resumable states. Use existing cold
archives only after full integrity/stage-back qualification. Bound new artifact
growth before admission; no shared cache relocation or speculative cleanup.

Exit: exact source/model/runtime IDs, live-state/budget/storage receipt, usable
allocated Linux route. If unavailable, continue static/config/core work and
record the exact missing capability; never report a GPU run as launched.

## Phase 1 — qualify the shared motor and simulator

Pin Motus revision (existing effc6e1a99bd4d0edf9447e8ba313def67649cfe pending
fresh remote verification), weight/config/norm/tokenizer hashes, joint/gripper
mapping, quantile vs mean/std, delta/absolute conversion, resize/padding, prompt,
native H32 and dtype. Compare native reference actions and the RLinf boundary
on recorded actual source observations using explicit full sampler noise/RNG.
Freeze VLM/vision; train only source action expert and PPO critic. Student
deployment freezes base, representation, normalization and history model.

Use a fresh uv overlay; preserve vendor/shared packages. Run core software/
numerical tests in allocated Linux CPU. In a GPU step run
`scripts/[train]gpu_probe.py --expected-count N --output <fresh path>` and real
simulator episodes. Bind CUDA and SAPIEN to allocated PCI/UUID; retain scheduler
visibility. Check three nonblank RGBs, 14D state/commands, finite native chunks,
actual physics motion, scene/instruction consistency and weight immutability.
Qualify reset, terminal/timeout, render warnings, camera type and controller.

Audit native source-task competence for coverage, ceiling/floor and reward
variation. This is a small qualification panel, not a full SFT rebenchmark.
Use existing source seed banks when they cover a task. Generate planner-valid
banks for pick_diverse_bottles/place_object_stand/scan_object on allocated Linux;
never substitute another task or silently fall back to random seeds. Hash whole
TRAIN/DEV/TEST banks; compare actual reset identities, not just integer seeds.

Exit: one joint motor qualified with real episodes and informative source
outcomes; a zero-reward/unsupported motor triggers boundary/coverage diagnosis.

## Phase 2 — port RoboCasa engineering and fill RTX memory

Follow ENGINEERING.md for source locations, integration and evidence per item.
Training **and evaluation** call the same qualified preprocessing, batched motor,
native controller, action decoding and passive execution accounting. Do not
leave evaluation on the legacy serial wrapper. Fresh source only for changes.

First fix passive fast-controller measurement without changing original PPO
loss masks: record requested targets, completed boundaries, terminal partial
target, physics steps and native timestep in a separate diagnostic field. Keep
old official executed_action_count behavior unchanged so RLinf does not acquire
new masks silently. Verify against the full-observation native reference and
physical trajectories for success, failure, timeout and already-successful no-op.
Use the separate measured prefix mask for corpus transitions. A changed PPO
mask policy would be a separately qualified method migration.

Profile both actor and inference on the actual allocated hardware, which may
be48/96GB or another RTX6000 variant. Do not infer VRAM from its product name.
For fixed global2048, choose divisors m with world*m*accum=2048. Sweep increasing
microbatches from the qualified value through the largest safe candidate, one
scratch lineage per OOM retry. Keep actor/critic LR, five epochs, reduction,
sampling, global batch and optimizer schedule fixed. Recompute old logprobs
once before all epochs whenever partition/kernel parity requires it; otherwise
PPO ratio1 must hold on unchanged weights. Include that cost in update timing.

The existing 4-GPU ~95GiB-device sweep found micro64 fastest, micro256 largest
with8GiB reserve, micro512 completing with only~1GB free. This is prior evidence,
not a measurement for the restored protocol. Default memory-oriented candidate
is256 on that same device/runtime; refit for H32, camera batch and actual VRAM.
Reject unsafe512. No dummy tensors to inflate utilization. Report largest safe
batch and fastest end-to-end batch; use the former for the user's requested
memory profile unless it violates runtime/budget/replay constraints.

Sweep inference lanes/batch separately (1,2,4,8,16,32,64,...), keeping the saved
global256 environment collection and rounds fixed; schedule in execution waves
instead of changing data quantity. Keep groups contiguous/complete. Measure
CPU scene/reset throughput, host RAM/SHM, GPU allocated/reserved/NVML peaks,
prefix cache, inference/update seconds, I/O and full iteration wall time.
Keep max(8GiB,8% total VRAM) reserve including rendering/communication peaks.
Bound host memory and artifact queue bytes as well as GPU memory.

Use case/episode/chunk-addressed RNG independent of rank/lane arrival; record
BF16 batch-shape differences and qualify tolerances versus eager reference.
Compile only fixed hot inference shapes after eager numerical qualification,
with warmup/time-to-amortize and eager fallback; no silent precision, TF32,
attention-kernel, denoiser-count or chunk-length changes. Carry mixed precision,
frozen eval, shared prefix reuse, process/thread affinity, bounded async I/O,
selective weight export and recovery sidecars into both execution modes.

Exit: actual-device report and the chosen train/eval profiles; passive counts,
likelihood/value replay, frozen gradients and independent full recovery pass.

The actual Motus reference diagnosis is recorded in BATCHMATH01-COMPLETE.json.
Native B1 and native batched execution are distinct numerical conditions, even
with identical per-case noise and preprocessing. Do not relax the original
singleton equivalence gate or call changes to actual batch shape transparent.
For new residual training/evaluation, record actual batch size/row and tail
policy and share that native protocol across methods. Log case-addressed noise
separately: it preserves draws, not cross-batch action equality.

## Phase 3 — independent source histories, bounded pilot first

Start place_phone_stand (existing scene bank); use independent PPO/GRPO
lineages from the selected original joint SFT, paired scenes/prompts/noise
protocol, equal measured interaction budgets. Two iterations first; inspect
actual optimizer advancement and independent source DEV at checkpoint0 and
each completed iteration. Retain initial, first-trained, DEV-selected, final
and latest recovery; checkpoint at every pilot iteration. Extend only after
measured time/storage and remaining cap allow it. Later tasks follow in the
declared source order; add training seeds when budget supports uncertainty.

Record full chronological streams including failures, actions actually sent,
executed-prefix masks/durations, current/next public observations, reward,
termination/truncation, trial/episode IDs, policy version/optimizer step,
checkpoint/source/config IDs and sampler RNG. Persist before SGD. No abandoned
prefix becomes a completed trajectory; charge every attempt. Whole teacher
stream/case split before fitting or window extraction. Keep policy version,
teacher iteration, hidden simulator state and source-only diagnostics out of
student inputs. They are audit metadata, not privileged shortcuts.

At each recorded query observation, reconstruct the original **frozen base**
proposal B using saved base sampler noise, then label residual A_teacher-B in
the same decoded action units/schedule. Include B in the student input. Teacher
noise alone is not a frozen-base steering label. Cache immutable public prefix
features/base proposals with motor/transform hashes; never cache an action-expert
state that changes with teacher SGD as if it were the fixed deployment base.

Exit is based on data information: improving histories support AD; flat but
mixed rewards/coverage support DIT/offline learning. All-zero or all-ceiling
data with no useful variation triggers coverage/reward/initializer diagnosis.
No fabricated corruptions, success-only filtering or final-checkpoint relabeling
of AD. PPO/GRPO improvement does not gate the entire project.

## Phase 4 — shared history learner

First establish residual controllability on one source task. Use per-coordinate
bounds in physical joint/gripper units, clipped/limited without changing base
transforms; record saturation and residual/base magnitudes. A trained static
residual is a control for headroom. If it cannot improve, diagnose interface
and coverage before attributing failure to the meta-learning objective.

Use the same causal architecture for objective comparisons: initial candidate
6 layers, width512, 8 heads,2048 decision tokens, separate modality projections,
source-only training normalization; tune only on source DEV under equal update
budgets. Tokens include public visual/state features, instruction, current base
proposal, previous executed action, previous reward, terminal/timeout and
episode boundary. Current outcome is never a current-action input. Memory
survives episode resets within an adaptation trial and clears between trials.
Bind both memory and Bellman boundaries explicitly; they need not be identical.

* AD: imitate the action used by the chronological teacher at that point.
* DPT: relabel only the current query with a qualified final/strong teacher at
  that same observation; report extra label compute/quality, never call it optimal.
* DIT-style: learn in-context values/advantages and weight action prediction
  by clipped exp(A/temperature); initial temperature1, cap20. Preserve failures
  as context even when action-target weight is low. Name flow-matching weighting
  through a KV bridge as a proposed adaptation, not the original DIT objective.
* Offline IQL-style: history-conditioned Q/V and residual actor, expectile .7,
  target EMA .005, advantage temperature3/cap100 as explicit candidate defaults.
  Use measured decision durations and chosen terminal/timeout bootstrap rules;
  a gamma per physics tick is a new reward-time convention and must be declared.
  Start with per-decision gamma .99 as the source convention. Report coverage
  and value extrapolation error; a critic cannot infer missing success evidence.

These are candidates for a controlled first fit, not paper-established optimal
settings or an approved large campaign. Add a matched context-free student,
episode-reset student and final-policy distillation control. KV/velocity/noise
interfaces remain separately named comparisons. A frozen expert can propagate
action-supervision gradients into a KV bridge; detaching it entirely is wrong.

Preserve the full planned action chunk alongside the measured delivered prefix:
native whole-chunk TOPP planning can make the undelivered tail affect the
executed trajectory. Do not zero unexecuted request coordinates in a critic
without qualification. Keep actual physics duration separate from wall time
and request length. Retain zero-duration terminal no-ops for audit; they are
not extra physical decisions. Before student fits, declare one shared Bellman
clock for DIT/IQL and verify episode/trial boundaries; native teacher masks
are not automatically physical corpus masks. The held-out normalization ban
applies to newly fitted student statistics; the released joint base model's
pretraining and fixed normalization may already cover those tasks and must be
disclosed as base exposure.

## Phase 5 — frozen online adaptation and claim checks

Freeze all model weights/normalization and use only actual accumulated feedback
at deployment. Report separate-query success after S=0,1,2,5,10,20 completed
support episodes; context resets each new trial. Never count the query as its
own support or report at-least-once success as query adaptation. This run's
support grid is explicit and differs from older0/1/2/4/8 contracts.

Primary paired methods share motor, interface, architecture, cases, execution
profile and measured action budget: native SFT, context-free residual, AD,
DIT-style, offline IQL; DPT when labels exist; online PPO/GRPO separately charge
their gradient and interaction cost. Learned-KV comparison is a separate factor.
Use trained no-history/reset baselines plus full-history reset, shuffled reward,
shuffled episode order, wrong-task history and reward-ablated interventions.
Shuffling reward alone does not remove visual outcome feedback.

Use source DEV before locking transfer; six held-out tasks remain untouched
until final. Candidate locked test:128 paired independent cases/task,3 training
seeds if affordable, task/trial-clustered uncertainty and per-task curves;
size/seed count finalized using measured cost before launch. Same task-family
conditions persist across supports/query while scene reset policy is explicit.
For the VLRLA reconstruction retain within-task scene resampling; report any
fixed-scene variation as a different protocol. Disclose base SFT task exposure.
Count every support/query action, value/teacher label and online update.

Exit: query curves versus real support count with actual coverage and source/
deployment costs. Static gains alone do not establish feedback adaptation.
Negative results remain useful evidence. Publish verified results in FINDINGS.

## Operations, budget and recovery

Do not assume the entire plan fits the prior50GPUh cap. Import all earlier
closed/failed charges and outstanding reservations; initially reserve at most
2GPUh for new qualification within the **remaining** balance. No new full-six-
task or20-support test launch until its measured cost fits. Above the existing
cap, prepare exact profiles/time/episodes/storage and request only the necessary
increment. CPU tests/static work need no budget approval. Elapsed time is not
approval. Never create a separate50h budget by renaming work.

Before launch: reviewed main SHA, detached source, output identity, scheduler
resources/limits, actual device type, storage+quota+write/fsync, lifetime budget
reservation, deadline/cleanup margin and GPU probe. Use job-local
NCCL_P2P_DISABLE=1 for6000D. Verify advancing updates or completed artifacts,
then check every20min; meaningful task updates include cluster, job ID, phase,
last check, next action and cumulative usage. Disable automatic retries; diagnose
before a fresh identity. No automation is needed to watch this specific work.

Checkpoint boundaries include actor/critic, owned Adam/groups, LR/scaler,
normalization, global/per-worker RNG, sampler/case cursors and pending histories.
Test independent fresh-process resume and the next completed iteration/case,
not only load_state_dict. No claimed in-flight physics resume. Before stopping,
verify exact ownership/job/step and retain the latest complete boundary. After
completion verify scheduler release/cost and artifact hashes, prune only safely
recoverable unused payloads under retention policy and close reservations.

## Missing details and autonomous resolution rules

* Motus H32 versus Sidney H50, transforms and native controller: resolve with
  explicit protocol and reference qualification, not an assumed compatible load.
* Fresh Baige state and usable compute access: recover the existing authenticated
  surface/configuration; do not infer liveness from local snapshots.
* Fast-controller execution masks and durations: implement passive metadata and
  qualify before corpus export, without retroactively rewriting old evidence.
* Missing three source-task scene banks: generate qualified disjoint banks.
* Full campaign cost and primary uncertainty target: measure pilot, propose a
  concrete extension only if it exceeds remaining authorization.
* Official VLRLA weights/code and matched data: remain reconstruction labels
  until verified. Motus's larger training dataset is not an equal-data PACE result.

None of these requires a new scientific decision from the user at planning
time. A missing authenticated compute capability or required budget extension
is reported precisely while all independent work continues.

## Native Motus first-iteration launch boundary

`motus_configuration.py` and `motus_teacher_entry.py` preserve native
H32/H192,256 environments,four collection rounds,global2048 and five epochs.
This gives6144 decisions/15 Adam updates per iteration. Native actor probability
recomputation handles micro256/accum2 before all epochs. PPO retains native
K-step mean rollout critic values and selected-step actor values, with an
independent capture of both; it does not import Sidney transforms or another
project's critic patch. An independent NumPy reference checks the actual saved
PPO/GRPO learner batch before SGD. Group8 and the shared source case schedule
apply to both algorithms.16 source DEV cases use the first16 pinned eval-bank
seeds without outcome filtering and never select a best-of-panel checkpoint.

The initial allocation stops at the first whole iteration of the intended
two-iteration lineage, retaining15 updates and a complete resumable boundary.
It is a measured prefix, not the completed paired comparison or source corpus
admission. Checkpoint0/1 include native DCP,owned Adam/groups,scheduler/scaler,
all worker RNG/samplers and case ordinal. Full cold readback and local stageback
precede their receipts; independent new-process restoration remains required.

Both training and evaluation bind the qualified passive controller after every
reset, including scenes recreated by offload. Measured terminal masks and
physics duration follow native terminal final_info without changing PPO masks.
The native worker reads mode-specific enable_offload flags; explicitly enable
both modes, since the historical top-level flag alone leaves scenes resident.
Use native batch-end trimming, with per-scene close trimming disabled.

All new worker writes use job-local staging or the existing project BOS archive;
shared source/model/cache paths are read-only. Per-rank request journals have
one writer per rank. Their500k-action whole-run envelope is precommitted; if an
attempt's journal is unrecoverable, conservatively charge the entire envelope
and admit no incomplete corpus boundary. Bound I/O to2jobs/512MiB, retain full
chronological canonical batches before SGD, and drop duplicate sidecar inputs
only after canonical SHA verification. Archive state/traces remain separate
from the old Sidney corpus. New scientific launches still require latest-main
review, actual four-GPU probe, capacity/quota/write checks and measured progress.

MOTUSPPO01 reserves15600GPU-s including creation/release allowance against the
closed153404/180000GPU-s cumulative cap. Worker timeout3300s,scheduler60min,
creation-to-release watchdog3900s. Refresh actual scheduler creation time and
watchdog when submitting; no whole six-task/student campaign funding is implied.

### OpenPi reference compatibility correction

Prefix02 exposed c605's explicit OpenVLA-only probability-reference guard
before model loading. The original reference diagnostic reshapes flattened
action tokens and cannot directly accept continuous OpenPi [N,H,A] outputs.
Do not bypass that validator or call its token reshape on Motus. The owned
actor now has an explicit `native_pilot.recompute_openpi_logprobs` extension:
retain [N,H,A], use the native two-level post-shuffle SGD partition and native
OpenPi forward arguments, replace old probabilities once before all5 epochs,
and report a masked per-decision mean coordinate gap. The actor temporarily
selects this owned hook at the native reference location; the driver/native
configuration remains `algorithm.recompute_logprobs=false`. The learner,
loss masks, GAE, optimizer and sampler remain pinned/unmodified. Numerical
CPU cases must verify exact shuffled partitions, continuous coordinates,
mask preservation, finite/all-masked behavior, failure atomicity and RNG.
Actual FSDP unchanged-policy replay is still required before first SGD.

Explicit DEV panels require `runner.val_check_interval=1` to construct native
evaluation workers. The owned runner suppresses native automatic panels and
runs only its declared initial/iteration DEV panels, avoiding duplicate queries.
Full native schema validation on CPU substitutes only the declared four-rank
placement metadata; every real job still runs actual Cluster/GPU validation.


## Residual corpus and student handoff

The existing `robotwin_history*.py` path is a different 1024-dimensional noise
controller. Its action labels, token layout and horizon/clock checks must not
be silently reused as decoded 14-dimensional residual labels. Reuse the causal
decoder and numerical DIT/IQL primitives where their contracts agree; qualify
new action/feedback tokenization and online reset handling on Linux.

The pinned OpenPi training sampler records `forward_inputs.chains` as
`[batch,6,32,32]`: entry0 is the original float32 Gaussian draw, followed by 5
native denoising states. The original frozen Motus ODE proposal can therefore
be reconstructed from that saved entry0 and the same public observation. Use
the original Motus weights, transforms and fixed normalization, not the updated
teacher expert. Verify sampled teacher decoded actions against the actually
submitted full 32x14 command, and retain their link to chronological canonical
batches before shuffle. The residual label is full `A_teacher - B_frozen` in
decoded command units; denoising indices/noise are audit inputs, not residual
steering labels. Every reconstruction records actual batch/row as well as
motor/transform/noise hashes. Changing native batch shape is a numerical
condition; the observed singleton drift rules out assuming arbitrary
repartitioning is transparent. Lock one qualified profile for all student
objectives and deployment comparisons before exporting the student dataset.

Split the source case groups before normalization, window extraction or
objective fitting. All 8 replicas of one scene belong to the same partition,
and paired PPO/GRPO uses the same partition. A collection-slot split that puts
replicas of the same seed on opposite sides is leakage. Precommit a grouped
stream view by task/rank/local-group, ordered by teacher iteration, rollout
round and replica index. Within a simultaneous vector round, replica order is
a deterministic offline history view, not a claim that the teacher consumed
one replica's feedback before acting in another. Four rounds x8 replicas give
32 complete source episodes per version per group; 2 teacher versions give 64.
Keep every failed episode. Source DEV panels never become student fit data.
This explicitly retains the teacher/student exposure mismatch for the AD
comparison instead of pretending that a student saw the teacher's whole pool.

Each new decision token contains only current public visual/state/instruction
features and the current frozen base proposal, plus shifted previous issued
commands, measured delivered-prefix mask, real reward, separate termination/
truncation, actual physics duration and the current episode-start bit. Keep the
whole previous issued chunk available alongside its measured delivered mask:
its tail can affect native TOPP execution. Current outcomes, teacher version,
optimizer step and hidden simulator state never enter the current query.
Full history preserves memory and terminal feedback across physical resets
within a trial; episode-reset clears both; static training clears every
history field and state on every decision. Merely clearing attention while
leaving prior feedback embedded in an observation vector is not a static
control. These behaviors need core causality/reset tests, not just manifests.

The initial shared offline Bellman convention is finite physical episodes,
gamma 0.99 per positive-duration issued chunk, no bootstrap across either
termination or the declared H192 timeout, and no cross-episode Bellman edge.
Memory may still carry across that boundary. Retain zero-duration terminal
no-ops in the audit stream, but do not count them as extra physical decisions
or fabricate next observations. Use actual environment rewards from the trace,
not the actor's timeout-bootstrapped GAE reward tensor. Unexpected nonzero
feedback on an unexecuted no-op requires diagnosis before export. Both DIT and
IQL use this same convention; physics seconds are recorded separately rather
than silently exponentiating gamma by integration steps.

Before first fitting, derive and lock residual bounds using source TRAIN
commands and native decoded arm/gripper units; report label coverage,
saturation, residual/base magnitudes and all clipping. Keep the full requested
chunk in actor/critic action inputs; do not zero the undelivered tail without
qualifying its effect on the controller. A one-version corpus may support
mixed-reward DIT/IQL and a static residual controllability check, but it cannot
be admitted as AD learning-history data. The complete controlled AD/DIT/IQL
comparison still requires its declared matched data and policy versions.


## Saved-collection recovery boundary

Prefix04 reached a complete chronological6144-decision collection and native
pre-SGD probability-reference checks before its fixed worker limit. It did not
produce a complete updated checkpoint. COLLECTION-AUDIT02-COMPLETE.json binds
all4 cold-read batches, original per-rank pre-SGD RNG and independent GAE.
A fresh learner-only recovery must restore checkpoint0 actor/Adam/scheduler,
then restore those exact RNG states immediately before the native learner.
Do not recollect, shrink global2048/five epochs, rerun GAE on a shuffled batch
or continue from partial uncheckpointed updates. Require all15 native updates
per rank, unchanged frozen parameters, complete owned Adam/scheduler/scaler/RNG
and logical sampler state, full cold/local readback and independent recovery.
The original pilot has no post-collection environment snapshot: a later online
continuation needs an explicit new-boundary collector/RNG migration. Never call
this an exact all-worker simulator resume or admit the first fixed policy as an
AD learning history. Preserve the original failed contract and charged prefix.

`replay_motus_learner.py` implements that learner-only boundary in a fresh
allocation. It reads the original resolved configuration and requires exact
actor/algorithm/rollout equality after full native validation. Each rank checks
its collection hash and independent GAE, restores the recorded pre-SGD RNG,
then executes the unchanged native shuffle and all15 optimizer updates. Keep
micro256/accumulation2/global2048 and all5 epochs. No simulator actions are
charged again or invented as new data.

At update15, save full native DCP, owned Adam/scheduler/scaler/RNG sidecars,
the logical sampler boundary, scheduled plain inference weights and a cloned
512-row numerical-control fixture per rank. Complete the cold and local
stageback checks before the unsaved update16 control. Fresh worker processes
must restore update15 exactly, qualify the plain inference export at the same
B4 partition and reproduce the same unsaved next update. The control never
becomes a retained teacher version. This does not replace an updated-policy
simulator DEV panel or the explicit collector migration needed for iteration2.
The prepared recovery reserves9000 GPU-seconds against9176 remaining; validate
the code, queue and exact deployment commit before admitting that allocation.

MOTUSREPLAY01-COMPLETE.json now records completion of that boundary at
c79bdc1d: all15 native updates/rank,19.085GB full checkpoint readback,
independent owned-state restore and identical next-update control, with exact
B4 plain-export likelihood/value calibration. The allocation closed with
174840/180000 GPU-seconds consumed;5160 remain and none are reserved.
Next qualify updated-policy simulator DEV on its fixed16 cases, then address
the combined collection memory/initialization gate. This does not admit the
existing single-policy collection as AD history or supply policy1 rollouts.


## Paired updated-policy DEV admission

CPUDEV01.json binds Linux-only import/configuration qualification at b05f2efa:
full native evaluation schema, unchanged actor/algorithm/eval-environment, the
complete effective rollout model and all4 actual original B4 inputs.
CPUDEV-INPUTS01-CLOSEOUT.json preserves the rejected raw-vs-transport hash
assumption; corrected cpudev-inputs02 verifies native EnvOutput packing and
per-rank saved RNG. No model or simulator was constructed during these checks.

MOTUSDEV01-CONTRACT.json admits4GPU/64CPU/256GiB/SHM64 for at most10 scheduler
minutes,570 worker seconds and660 creation-to-release seconds. The2640GPU-s
reservation gives worst177480/180000 and2520 unreserved. Use the complete
update15 plain weights, original16 seeds, native B4/rank and saved baseline
RNG. Require static original-SFT action calibration and exact first simulator
observations before reporting a paired result. Native environment/rollout
channel calls, measured controller, bounded writer and metric aggregation are
shared with training DEV; launch no actor workers or training scenes. Require
16 completed trajectories, zero SGD, frozen parameter bytes, I/O drain, at
least8GiB sampled GPU reserve and independent trace readback. Keep this panel
out of student fitting and held-out transfer claims. Close the scheduler's
actual creation-to-release charge before allocating any remaining allowance.


MOTUSDEV01-COMPLETE.json closes that panel: update15 policy2/16 versus
baseline4/16, with1 paired gain and3 losses. All4 original SFT first-action
checks were exactly0, all4 fresh initial observations matched, frozen weights
were unchanged and evaluation used0 SGD. Independent CPU readback verified
all166 new state files and both panels'24 traces/24 sampling files each,
including actual commands/public observations, physical/terminal packets and
identical first Torch/CUDA RNG states. This small source DEV panel shows no
improvement; it does not select a checkpoint or qualify held-out transfer.
The4GPU allocation closed09:19:52, charged2128 conservative GPU-seconds.
Cumulative176968/180000 leaves3032, with none active or reserved.

The next bounded repair should first qualify deferred setup against the16
retained updated-policy trajectories on one allocated GPU, using fresh
processes and the original4-row partitions. Replay the exact issued chunks,
compare public observations/language and measured physics/feedback across
the whole H192 episode, and retain default setup as the immutable reference.
Measure constructor/reset time and memory. Keep actual intermediate render
semantics, controller, native loss masks and finite reset guards unchanged.
A proposed2400-second creation-to-release ceiling would fit3032 remaining,
but is not admitted until its code, Linux checks, storage/action bounds and
current resources are verified. A component-only64-scene measurement would
not qualify a co-resident actor/rollout/environment training allocation.

After setup parity, qualify the actual combined profile with the8GiB reserve
and exact native B64 likelihood/value checks before collecting policy1. A
pipeline-stage/batch change needs explicit numerical qualification rather than
being described as transparent. Online continuation also requires the recorded
new-boundary collector/RNG migration. The remaining3032GPU-s do not fund the
matched six-source PPO/GRPO and student campaign; prepare a measured extension
before exceeding the user's hard50GPU-hour cap.


SCENESETUP01-CONTRACT.json now admits that bounded1GPU replay after
CPUSCENE02 at3f7ac49a: all24 retained chunks cold/staged/read, exact original
config/manifest/readback, shell syntax and6 allocated-Linux reset/setup cases
passed with CUDA uninitialized. CPUSCENE01 preserves a rejected mistyped
readback-hash literal; the archived receipt itself remained unchanged.
Run a default rank0 control first, then all4 deferred reference ranks in
separate processes. Preserve native constructor/offload/reset ordering, B4,
intermediate-render profile and full requested chunks; compare all5 native
output parts, including raw feedback and final/reset observations.
The5120-action ceiling includes explicit temporary/real setup allowances.
Require8GiB sampled reserve and complete child closure. The2400GPU-s
creation/release reservation gives worst179368/180000,632 unreserved.
No model, SGD, student data or large-scene/combined-training claim is involved.


SCENESETUP01 failed before simulator construction because the shared shell
inserted a4-GPU fallback into a1-GPU container; the actual job probe rejected
this and the allocation released10:33:41, charged62GPU-s. Preserve the failed
source/outputs. SCENESETUP02 uses a fresh output/source and a fallback of0 only
when scheduler visibility is unset. CPUSCENE03 atd0799d46 passed4 actual shell
visibility cases, all24 source chunks/config/reference hashes, and6 reset tests
on Linux with CUDA uninitialized. Its admitted2400GPU-s reservation gives
worst179430/180000 against177030 closed, leaving570 unreserved.
CPUCHUNKMODEL01 separately qualified11 decoded-history/model cases at4c7b9c24;
student export/trainer/evaluation integration remains pending.

SCENESETUP05 at2b8824fb now qualifies opt-in deferred setup at native B4
against the retained updated-policy DEV commands: default4 cases plus all16
deferred cases,30 chunks, all5 native output parts and measured execution
exact. Five fresh processes completed explicit final cleanup after joining the
executor and synchronizing CUDA before native offload. Independent CPU
readback at149a134d cold-verified64 JSON files and24 unique source traces.
This final-shutdown order is not a replacement for reusable per-iteration
Offload: repeated training offload, long reset stress, larger scene counts and
the combined actor/rollout/environment memory profile remain unqualified.
See SCENESETUP05-COMPLETE.json. No model calls or SGD ran. Constructor/reset
timings are default-first with shared job caches; do not claim an isolated
speedup or end-to-end training throughput from this replay.


`meta_rl/chunk_value.py` now connects the decoded-chunk encoder and actor/Q/V
models to AD, DIT-style and offline IQL losses. Complete positive-duration
windows use the same full/episode/static resets as online queries. Both
termination and H192 timeout stop Bellman edges; return-label clocks must
match the declared gamma. Whole requested residual chunks remain action
labels, including undelivered tails. IQL uses expectile0.7, temperature3,
cap100 and target-Q EMA0.005; DIT-style uses temperature1/cap20 and the
explicit0.1 Bellman auxiliary inherited from the existing value learner.
Global valid-decision weighting and optimizer execution reuse the qualified
shared helper. CPUCHUNKOBJECTIVES01 at9bb0437b passed37 Linux CPU cases,
including4 new decoded-objective numerical/gradient/batch-partition checks.
These tests construct small CPU models, with CUDA uninitialized; no native
Motus or simulator is loaded. Corpus export/admission, complete fitting,
checkpoint/online integration and real student evaluation remain pending.
A one-version corpus is still inadmissible for chronological AD, and DPT is
rejected until qualified same-query labels exist.


LIFECYCLE01 at61e099b3 now qualifies reusable native offload at B4 in the
actual shared train/eval setup hook. Four default/deferred x train/eval
profiles each ran two collections of two episodes:64 episode replays on4
unique retained cases,96 chunks and12 native offloads. Offload joins old
executor threads and synchronizes allocated CUDA before native cache close,
then creates an executor that subsequent resets/replays can use. All4 fresh
processes exited0 with final executors closed. Evaluation matched all5 retained
native output parts; default/deferred commands, outputs and measured physical
execution matched within both modes. Independent CPU readback at9bb0437b
verified126 JSON files,6 original traces, GPU/renderer binding and12720 charged
conservative actions. Sampled minfree was94358208512 bytes; no model or SGD
ran. Released12:53:16 Nov7, charged966GPU-s; cumulative178782/180000 leaves1218
with none reserved. This qualifies opt-in repeated B4 offload, not the full
four-round G8 collector,64-scene co-resident memory or long reset stress.
See LIFECYCLE01-COMPLETE.json and LIFECYCLE01-READBACK01.json.
