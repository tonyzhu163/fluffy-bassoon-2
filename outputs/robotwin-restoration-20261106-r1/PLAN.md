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
