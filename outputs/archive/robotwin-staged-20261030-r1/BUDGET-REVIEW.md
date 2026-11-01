# Review of staged RoboTwin collection under the shared action cap

Read-only planning review, 2026-10-30 Asia/Shanghai. No model computation,
tests, allocation, or edits to the driver or shared coordination were made.
The earlier 272k-per-arm pilot remains unlaunched. The figures below describe
the proposed new campaign; they do not change any existing run contract.

The 32k / 64k / 128k schedule fits the confirmed **one-million-action total
RoboTwin cap**, including all three training arms and the requested evaluations.
It is a cumulative continuation schedule: the final source cost is
3 x 128,000 = **384,000**, not 3 x (32,000 + 64,000 + 128,000).

## Exact maximum action arithmetic

Assumptions: three source learners, K4 training, 400-control-action episode
cap; at each milestone, one K4 validation trial per learner on every one of
the six source tasks, and one K8 trial per target condition. Target conditions
are persistent PPO normal / memory-reset / reward-blind, frozen SAC, online
SAC, and separately trained reset-PPO normal. No extra trained-reset lesion
is necessary. One native episode per selected target is run before and after
the target evaluations. All early successes can lower actual cost.

| Component | Maximum per milestone | Maximum across three milestones |
| --- | ---: | ---: |
| Source training, cumulative final 128k per arm | incremental |384,000|
| Source validation: 6 tasks x K4 x 3 arms x 400 |28,800|86,400|
| Target: 2 tasks x K8 x 6 conditions x 400 |38,400|115,200|
| Native before/after: 2 tasks x 1 episode x 2 passes x 400 |1,600|4,800|
| Previously executed r4 collection plus capture qualification | imported once |13,026|
| **Maximum charged total** | |**603,426**|
| **Unspent ceiling for retries, further qualification and later decisions** | |**396,574**|

The imported total is 11,475 actions from r4 plus 1,551 from the camera
qualification. Those feature/hash artifacts do not provide the new raw RGB
corpus and receive no new source-training credit. Earlier failed startup
attempts recorded zero executed policy actions. Explicit future settling or
no-op robot commands must count; reset-internal PhysX integration ticks are
not separately instrumented robot-control commands.

| Milestone, source actions per arm | New source actions since prior milestone | Evaluation maximum | Cumulative campaign maximum, including prior 13,026 |
| --- | ---: | ---: | ---: |
|32,000|96,000|68,800|177,826|
|64,000|96,000|68,800|342,626|
|128,000|192,000|68,800|603,426|

Use the first two targets in the existing canonical held-out list,
`adjust_bottle` and `move_stapler_pad`, fixed before outcomes are inspected.
This is a two-task diagnostic subset; it cannot support a six-task transfer
claim. A stronger affordable plan is to use those two at the first two scales,
then evaluate **all six targets at 128k**. The additional four targets and their
native references cost at most 80,000 actions, giving **683,426 total** and
316,574 remaining. Either choice must be frozen in the initial contract.

Do not fill the remaining ceiling automatically. One million is a shared
upper bound, not a collection target or a per-method entitlement. Repeating
all source training from scratch at each scale would increase the three-arm
source cost to 672,000 and total cost to 891,426 for the two-target plan; that
also fits numerically but answers a different question and discards the
intended continuation curve.

## What the checkpoints can establish

Keep all six source tasks from the first stage. Uniform shuffled **trial**
blocks give each task recurring exposure; they do not guarantee equal actual
actions per task because success terminates episodes early. Record both trial
and action counts per task at every snapshot. With H400/K4, a trial costs at
most 1,600 actions, so 32k requires at least 20 trials per arm: at least three
full six-task blocks, apart from an explicitly recorded final budget cut.
There is no need to narrow training to two source tasks to fit this cap.

With eight K4 trials per PPO update and no artificial flush at intermediate
milestones, minimum cumulative PPO update counts are approximately:

| Source actions per PPO arm | Minimum trials from H400/K4 | Full eight-trial updates | Optimizer steps at five epochs and minibatch >=8 |
| --- | ---: | ---: | ---: |
|32,000|20|2|10|
|64,000|40|5|25|
|128,000|80|10|50|

Early success can increase trials and updates for the same actual-action cap.
The final source-boundary flush can add a partial-batch update. These are
data/accounting bounds, not measured optimization throughput or evidence of
convergence. Eight-trial PPO is an explicit pilot departure from the original
256-trial recipe. Do not claim that every intermediate snapshot follows a
complete PPO batch if pending on-policy trials are retained there.

For SAC, let N be the actual saved source chunk decisions. Current code makes
20 SAC gradient updates per decision after the first 500 warmup decisions,
so cumulative updates are **20 x max(0, N - 500)**; actor, critic and temperature
each step once per SAC update. Each decision executes no more than 32 actions,
so at the three exact source caps N is at least 1,000 / 2,000 / 4,000. This
implies at least **10,000 / 30,000 / 70,000 SAC updates**, with three times as
many optimizer.step calls. Actual episode-final partial chunks increase N.
No useful empirical upper bound or update-time estimate has yet been measured
for this real batch256/UTD20 pipeline. Report N and updates explicitly.

The 32k checkpoint is therefore a sensible first learning/throughput gate,
but only about two full PPO updates is a weak scientific test. A smaller 16k
source stage could sit near SAC warmup and provide little post-warmup data;
an 8k stage may not train SAC at all. A brief infrastructure slice can validate
raw I/O and recovery, but should not replace the first learning milestone or
be treated as evidence for or against meta-learning.

K8 reports at indices 1/2/4/8 are episode-index adaptation curves. With one
trial per condition on two targets, each reported index has only two target
outcomes per condition. Such a curve is diagnostic and has high uncertainty.
Source and target evaluators must restore independent source snapshots and
must not let online target SAC updates enter later source collection.

## Compute estimate and the remaining original allowance

The only qualified faster-collector measurement is the two-scene camera gate:
26.96 executed actions/second, from 517 actions in 19.180 seconds. Reference
versus candidate traces matched, including the 117-action early success and
400-action timeout. That measurement excludes raw corpus writing and training
updates, and does not cover throughput on all six source tasks or targets.
See [capture results](../robotwin-capture-qual-20261029-r1/RESULTS.md).

At that measured rate, the following are **collection-only extrapolations**:

| Work | New maximum actions | Time at 26.96 actions/s |
| --- | ---: | ---: |
| First 32k source milestone, all three arms |96,000|59.3 minutes|
| Its source/target/native evaluations |68,800|42.5 minutes|
| First milestone plus evaluations |164,800|1.70 GPU-hours|
| Increment from 32k to 64k plus evaluations |164,800|1.70 GPU-hours|
| Increment from 64k to 128k plus evaluations |260,800|2.69 GPU-hours|
| All new collection, two targets at all scales |590,400|6.08 GPU-hours|
| All new collection, all six targets at final scale |670,400|6.91 GPU-hours|

These are not guaranteed lower bounds or authorized resource budgets. Task
variation may move simulator throughput either way, while actual UTD20
optimization, immutable policy snapshots, raw NPZ compression/fsync, repeated
model setup and checkpoint I/O add costs not measured by this gate.

The old **170-second** remaining qualification allowance can at most correspond
to about 4,583 actions at that reference rate before startup/I/O. It cannot
reach a 32k-per-arm learning milestone. With long episodes and serial three-arm
interleaving, the first eight-trial PPO batch alone spans roughly 35.2k aggregate
actions before its update; SAC's 500-decision warmup is roughly 15–16k own
actions and around 45–48k aggregate actions before training. Exact first-update
times must be measured. A larger run must be proposed in GPU-hours, separately
from the already confirmed one-million-action scientific cap.

A two-hour resumable first slice is a defensible **proposal** for the 32k
learning milestone and its evaluations, but may not finish both after the
unmeasured optimizer/raw-I/O costs. It should advance real source learning,
save at a safe boundary and report observed actions/updates/bytes/seconds;
it is not authorization for all three scales or a promise of completion.

## Storage estimate: raw corpus has not yet been measured

The pinned D435 camera configuration is 320x240 for main and wrist cameras;
the adapter uses three uint8 RGB views. A public frame contains 691,200 RGB
bytes before compression, about 8,248 feature bytes and a small proprioceptive
array. The recorder content-addresses frames so a next observation is reused
as the next decision's current observation. It does not store two independent
full frame copies for each adjacent chunk.

In a **long-episode planning scenario** with 13 chunk decisions plus one reset
frame per 400 actions, the first full milestone/evaluation block has about
5,768 frame references, or **4.03 GB uncompressed**. All three two-target scales
have about 20,664 frames, or **14.45 GB uncompressed**; expanding the final scale
to six targets gives about **16.41 GB**. These are arithmetic scenarios, not
measured NPZ sizes, compression ratios or upper bounds. Short successful
episodes can raise frame counts per action; content deduplication can lower
actual stored bytes. Decision metadata/noise/actions/native outcomes add files.

Immutable SAC acting weights also matter. The actor has 562,048 float32
parameters, about 2.25 MB before serialization. At the same long-episode
scenario, 128k SAC source actions produce about 4,160 acting decisions; after
warmup, about 3,660 distinct actor versions could be saved, roughly **8.23 GB**
of parameter data before file overhead. Online target adaptation adds further
versions. PPO versions, complete source/target recoveries, replay and journals
add more. Do not estimate working storage from compressed RGB alone or assume
the earlier hash-only run measured this corpus cost. A capacity/write probe
and actual bytes/decision measurement are required before extrapolating size.

## One ledger and safe continuation

Create **one fresh shared ledger** for the entire staged campaign. Import
the two pinned prior receipts once; subsequent allocations call the same
idempotent imports. Use total cap1M and final source caps128,000 per arm from
the first allocation. Keep source milestones and evaluation lists in the
initial immutable contract. Do not create a separate ledger per scale or
change method caps from32k to64k to128k on resume: the ledger correctly rejects
changed contracts/caps. The same applies to changed source commits under the
current exact driver contract; routine wall-clock continuations should use
the same pinned driver unless a separately recorded migration is implemented.

The current driver needs the complete stage loop, snapshot preservation and
stage-qualified evaluation-unit identities; merely passing different
--source-steps to three existing launches fails contract equality. Keys such
as source_validation:method:task and target:method:task:repetition must include
the milestone, otherwise later scales would be skipped as already completed.
Record exact snapshot identities and cumulative retained source costs with
each evaluation. Keep pending PPO trials, source optimizers/replay/samplers
and all RNG state across stages. Preserve/restore source interaction RNG around
evaluation so milestone tests do not silently perturb later training draws.

Prefer completed-trial boundaries for intermediate milestones. If exact
action thresholds intentionally cut a K4 trial, retain its declared bootstrap
and external_stage_budget_cut marker; do not describe it as a complete trial.
Carrying old-policy pending trials through an evaluation is valid if the
source policy does not change before their ordinary PPO update. A deliberate
partial-batch flush at every stage instead changes the optimization schedule
and must be explicit.

The ledger reserves before every simulator call, refunds only a resolved
unused upper bound, and keeps unresolved crash intents charged in full.
Completed evaluation units should be checkpointed at trial boundaries so
resume skips them. A partially executed evaluation may need to restart under
a fresh identity; its previous physical actions remain charged, even when
excluded from the retained result. Likewise distinguish source actions charged
by the ledger from source actions retained in learner recovery. Do not refund
or double-credit lost experience after a crash.

Numerically the proposed plan has ample retry allowance. Scientifically its
first stage is useful for measuring real learning and runtime, while the
later scales and broader final target set provide the actual small learning
curve. Neither action-cap feasibility nor camera parity establishes trained
steering, useful memory, or convergence.
