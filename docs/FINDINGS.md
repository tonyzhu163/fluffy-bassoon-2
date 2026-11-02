# Findings to date

Consolidated 2026-11-01. Only the Meta-World ML10 comparison and the DSRL runs
are well-controlled enough to build on. The other threads are summarized here
so they are not repeated. Their run folders were deleted, except RoboTwin's
(restored under `outputs/archive/`); other links point to GitHub permalinks at
the pre-cleanup commit. Single-seed results are
descriptive, and none establishes a CTM, memory or in-context-RL advantage.

## Headline

- **No experiment so far shows in-context adaptation or a memory benefit**, on
  Meta-World, RoboMimic, LIBERO or RoboTwin. Reset and reward-blind controls
  match or beat full-history controllers wherever they were run.
- **Ordinary DSRL works when there is headroom.** Original Pi0 on task 57 goes
  4/10 → 10/10, and OpenPI40 pi0.5 on task 44 goes about 5/20 → 20/20. Both
  are single-seed. On saturated tasks (pi0.5 task 57, already 10/10) it only
  shortens episodes.
- The main blockers were **headroom and source learning**, not freezing the
  motor. Weak source learners (~11–17% success, flat curves) gave the
  meta-learners nothing useful to imitate.

## Meta-World ML10 (CTM question)

10M training transitions per run, seed 1000.
[Evaluation](../outputs/metta-common-ppo-chart-20261029-r1/ML10-EVALUATION.md),
[common-PPO bundle](../outputs/metta-common-ppo-chart-20261029-r1/README.md),
[native-method bundle](../outputs/metaworld-ml10-matched-baige-20261028-r1/README.md).

| Core (common PPO, ≤250k params) | Train success | Held-out success |
| --- | ---: | ---: |
| GRU | 20.8% | 0.8% |
| Transformer-XL | 20.0% | 8.8% |
| CTM | 16.0% | 0.0% |
| CMM | 10.0% | 22.4% (reset 26.4%) |

The CMM lead is not a memory effect: reset scores higher. The native-method
comparison (CTM, RL²-PPO, ECET, AMAGO1) shares the CTM checkpoint and also
shows no CTM advantage. CMM few-shot calibration failed (4.55% vs paper
30.63%) from optimization instability.

## RoboMimic Can

Two seeds × 8 held-out scenes × 4 attempts, 28,800 training steps per arm.
RL² scores 41/64 and frozen DSRL-NA 31/64. RL² with memory reset scores 40/64,
and an episode-reset-trained controller 39/64, so the gain is not from learning
across attempts.
[Evaluation](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/741064b4d8df00555f92a7a8449055c0edb50707/docs/robomimic-can-dsrl/EVALUATION.md).

## RoboCasa365

The original Human30075000 pi0.5 completed the predeclared OpenDrawer headroom
bank with **3/8 terminal successes**, full native750-step episodes. The first
episode verified all15 chunk seeds against exact zero-residual action replay
before executing each chunk once. The task was frozen and the fallback was
not run. On exactly those eight native videos, official Flora expected scores
ranked10of15 success/failure pairs correctly: **AUROC0.6667**, below the
predeclared0.70 gate. Both integrated simulator- and Flora-reward residual PPO
pipelines therefore stopped cleanly before optimizer updates or shared before
panels. That original attempt produced no comparative learning or final trained checkpoint.

The user subsequently authorized continuation below the original AUROC gate.
The unchanged initializer scored **4/8 on the fixed held-out ID panel** and
**6/8 on the predeclared CloseFridge OOD panel**, before PPO; initial DEV was
7/8. These eight-case panels are descriptive and remain separate from TRAIN.
A common 64-episode residual-PPO batch now collects through eight allocated
GPU workers; the numerical checks preserve the serial draw stream and next
Adam update. The simulator-reward pilot finished after six updates under the
predeclared DEV futility rule, with a complete final recovery saved. Final
ID was7/8 versus4/8 before PPO; CloseFridge OOD was5/8 versus6/8. These
eight-case panels do not establish an improvement or a simulator-versus-Flora
result. [Pilot terminal audit](../outputs/robocasa-eight-gpu-speed-20261102-r1/terminal-20261102-0205/PARENT-REVIEW.json). The original failed calibration remains
unchanged. [Before panels and qualification](../outputs/robocasa-eight-gpu-speed-20261102-r1/RUNTIME-STATUS.json).

The persistent official scorer loaded once and completed all eight labels;
two affected core numerical/reuse/cleanup cases passed on allocated Linux.
This qualified scorer execution before the complete Flora training result below. A separate one-view throughput benchmark completed all
eight batch settings: batch8 processed64replay requests in75.35seconds per
GPU after about590seconds cold startup. Those repeated videos measure speed,
not additional independent calibration cases or three-view performance. The eight-case result is descriptive development evidence,
not a population estimate or a conclusion that Flora is generally unsuitable.
Original weights, complete native videos, raw probabilities and costs are
retained. No task or score was adjusted after observing this failed gate.
[Native result](../outputs/robocasa-native-residual-headroom-q4-20261101-r1/terminal/summary.json),
[scorer gate and terminal audit](../outputs/robocasa-residual-integrated-20261101-r3/TERMINAL-REVIEW.json).


The fresh three-view OpenDrawer preview used synchronized left, wrist and right
recordings with the unchanged official scorer and short prompt. Its three native
successes ranked above all five failures: **AUROC1.0 on eight cases**. This is
an exploratory preview, not the new50-case gate or a paired camera ablation.
Batch8 processed64replay requests in113.38seconds per GPU after about580–586seconds
cold startup, with observed peak77,625MiB on a97,887MiB card and no OOM. Replays
measure throughput and add no independent cases. [Three-view audit](../outputs/robocasa-flora-throughput-three-view-20261102-r2/PARENT-REVIEW.json).

A native rollout capacity test completed 64 full 750-step OpenDrawer episodes, using the same eight seeded cases on two GPU replicas for each lane count and one shared frozen policy per GPU. Mean collection throughput at 1/2/4/8 environments per GPU was 15.76/27.95/51.67/86.93 control steps/s, including resets, residual inference, physics and three-view video; cold loading and extra numerical comparisons are reported separately. Eight lanes were best among tested settings, 5.51× the one-lane rate, with sampled peak GPU memory of 18.77 GiB. No optimizer update ran. Batched BF16 execution produced initial-action differences up to 0.00474 versus singleton inference, so production comparisons must use the same batching. This does not qualify mixed-task throughput or native/scorer memory colocation. The successful allocation charged 2,732 GPU-s; the diagnosed launcher failure before model loading charged 640 GPU-s, retained once. [Capacity result](../outputs/robocasa-rollout-batch-capacity-20261102-r3/PARENT-REVIEW.json).

With one persistent official Flora TP1 engine (batch limit8, utilization.78), a four-lane worker completed ten mixed-task episodes and7,200 controls with all ten grades streamed before the final request and zero SGD. Collection from the first reset intent through the final complete video took147.65seconds (48.77controls/s/GPU); including native model loading it took183.02seconds (39.34controls/s). Sampled peak combined VRAM was88.01GiB. The first continuous-refill attempt exposed a forked reset child retaining another lane's FFmpeg pipe and blocking video EOF; spawned resets repaired it, with57/57 affected Linux core tests and a regression that fails on the old code. [Audited colocation measurement](../outputs/robocasa-joint-lane-qualification-20261102-r4/PARENT-REVIEW.json), [pipe evidence](../outputs/robocasa-joint-lane-qualification-20261102-r3/DEADLOCK-EVIDENCE.json).

This is one GPU's physical measurement, not an eight-GPU joint PPO iteration. After the real Flora warmup only20,664MiB was free: below the8-lane27,408MiB requirement and1,309MiB below the4-lane21,973MiB production requirement, which includes an8GiB operating reserve. At that pin the four-lane production reserve remained unmet. Sixteen native lanes have not been measured; eight was the largest and fastest standalone setting tested. That native-horizon measurement does not qualify the later600-control protocol.


The600-control migration now explicitly selects the declared simulator horizon instead of the registry's750 controls, while retaining native OOD horizons. With Flora utilization lowered from.78 to.75, batch capacity8 and the full8GiB operating reserve unchanged, ten fresh episodes completed6000controls and all ten streamed grades with one model load and zeroSGD. Warmup left23608MiB free, passing the21973MiB four-lane guard by1635MiB; eight lanes still fail. Collection took132.54seconds (45.27controls/s/GPU); including native loading,206.03seconds (29.12controls/s). Peak combined VRAM was85.14GiB, minimum free9.83GiB, and cold Flora plus its first eight warmup scores took473.50seconds. All video/frame-clock hashes,120records and6000resolved physical commands were audited;82/82 affected Linux core tests pass. This qualifies four-lane execution and the horizon change on one GPU. Fresh600-control calibration and joint before-panels remain required before joint SGD, and the prepared entries now enforce the bound lane count. [600-control colocation audit](../outputs/robocasa-joint-lane-qualification-20261102-r5/PARENT-REVIEW.json).

Four actual GPUs subsequently passed the same eight-video official Flora warmup, each leaving23608MiB free. That preparation job stopped before native policy loading because its qualification reference pointed to an absent output-directory copy; the exact bytes were present in the pinned Git checkout. It produced no fresh episodes, physical actions or optimizer updates, so fresh600-control gates and common-before panels remain unqualified. All four allocations closed; preparation accounts for3268GPU-seconds including prior startup attempts. Validate the full runtime input closure before model startup. [Four-GPU terminal audit](../outputs/robocasa-joint-h600-preparation-20261102-r3/RESULT.md).

The five-task initializer qualification and Flora scoring are complete. Task
selection was manual for diversity, and OpenDrawer carried over from the earlier
pilot. Fixed seeds control episodes and policy sampling; these tasks are a
development subset, not a random sample of RoboCasa365.

| PPO candidate | Native successes | Flora AUROC |
| --- | ---: | ---: |
| OpenDrawer |29/50 (58%)|0.966|
| PickPlaceCounterToCabinet |37/50 (74%)|0.900|
| PickPlaceDrawerToCounter |23/50 (46%)|0.977|
| TurnOffStove |7/50 (14%)|0.894|
| CoffeeSetupMug |17/50 (34%)|0.923|

All five AUROCs exceed0.7 on their paired full50-case native banks. Each episode
ran its full native750- or600-step horizon with synchronized left, wrist and
right recordings; all250 official raw scores/probabilities are retained.
Counter-to-Cabinet exceeds the declared70% native-success ceiling, so its gate
remains failed pending an explicit human keep-or-replace decision. The other four
native gates pass. This measures initializer success and scorer discrimination,
with no new joint-policy training or camera-ablation claim. The human selected
one joint policy per reward arm and five PPO-held-out OOD tasks.
[Paired native/Flora audit](../outputs/robocasa-flora-five-task-fifty-score-20261102-r1/PARENT-REVIEW.json),
[selection disclosure](../outputs/robocasa-five-task-flora-ppo-20261102-r1/SELECTION-DISCLOSURE.json).

A partial native first-success timing panel retained126 full episodes, with
52 ever-successes. Per-task empirical p99 first-success times were279
(OpenDrawer),538 (CounterToCabinet),538 (DrawerToCounter),317 (TurnOffStove)
and524 (CoffeeSetupMug). With fewer than100 successful observations per task,
these are sample maxima, not established population99% coverage. The human
selected a prospective600-control cutoff, and the timing allocation was
stopped after its second wave. All observed first successes occur by538, but
one DrawerToCounter episode is false at600 and true at750: terminal positives
are48 at600 versus49 at the original horizons. Stove's two successes were
transient, with no final750 success in this partial panel. A600-step comparison
requires matched Flora calibration. [Decision, episode values and closure](../outputs/robocasa-pi05-success-timing-20261102-r3/RESULT.md).

The matched six-update OpenDrawer residual-PPO arms now have full50-case
held-out after-training panels:

| Policy / PPO reward | OpenDrawer ID | CloseFridge OOD |
| --- | ---: | ---: |
| Untouched original pi0.5 |28/50 (56%)|28/50 (56%)|
| Simulator terminal success |34/50 (68%)|32/50 (64%)|
| Official three-view Flora expected score |31/50 (62%)|28/50 (56%)|

Flora did not exceed simulator PPO on these panels. Its OOD count ties the
untouched original, so there is no observed net OOD degradation on this cohort.
The original matching50+50 evaluation completed with all100 cases, actual noise
schedules and1650 unchanged zero-residual chunks verified. This is one training
task and one PPO source seed; the paired two-sided exact p-values for S versus
original are0.210 ID/0.481 OOD, and F versus S are0.508 ID/0.388 OOD. These
small-panel differences do not establish a reliable population advantage or cause.
Retained-data investigation confirms identical initial residual tensors/configs
and360 Adam steps per arm; true TRAIN success is197/384 for S and201/384 for F.
Flora TRAIN reward AUROC is0.925, but the model explicitly emits `Score: 5` on
58/183 simulator failures. Holding the F records/critic fixed and substituting
binary labels changes23.1% of normalized advantage signs in batch6 (actor-gradient
cosine0.308), without any optimizer update. This measures reward/metric mismatch,
not the cause of a reliable S/F performance gap. [Investigation and limits](../outputs/robocasa-ppo-reward-gap-investigation-20261102-r1/RESULT.md).
A frozen original-versus-qualitative-rubric replay on all 384 F TRAIN videos
and 250 earlier calibration videos does not support adopting the candidate. It
lowers 20 of the 58 original false full-success grades, introduces five new ones,
and removes full-success grades from 35 true successes. TRAIN reward AUROC falls
from 0.9269 to 0.9191; mean expected-grade reward on those 58 failures rises
slightly from 0.7145 to 0.7182. Stove false full-success grades rise from 6 to 17
on the 50-video calibration bank. This is retained-video development evidence,
not a new PPO or independent qualification result. The exact original prompt,
tested wording, complete comparison and an untested next draft are in
[the rubric document](../outputs/robocasa-flora-rubric-replay-20261102-r1/RUBRIC.md).
Do not substitute the separate50-case calibration bank for that baseline.
The Flora arm completed384 TRAIN episodes with all384 official labels audited;
its full5760-decision history exactly merges the preserved four-update prefix
and the two-update recovery. The transient disk-monitor failure and partial
fifth rollout remain charged:430,528 controls and46,384 GPU-s across the two
Flora attempts, including DEV and final panels. Full actor/critic/Adam/sampler/RNG
recoveries remain preserved. These histories support development of AD/DPT;
they do not by themselves establish in-context adaptation.
The actual case identities and per-chunk motor-noise seeds match across all100
final S/F episodes, as does initial public proprioception. Frozen visual features
and action chunks are not bitwise identical across runs, including a substantial
gripper-channel difference in one OOD case. Matching seeds therefore does not
establish perfect numerical pairing or the cause of a few-case success gap.
[Seed and numerical audit](../outputs/robocasa-original-pi05-eval50-20261102-r1/INITIAL-NUMERICS.json).
[Simulator panel audit](../outputs/robocasa-simulator-ppo-eval50-20261102-r1/TERMINAL-REVIEW.json),
[Flora terminal audit](../outputs/robocasa-opendrawer-flora-resume-20261102-r1/TERMINAL-REVIEW.json),
[original matching-panel audit](../outputs/robocasa-original-pi05-eval50-20261102-r1/TERMINAL-REVIEW.json).

All four PPO-history students reached their declared validation plateau:
AD/full and AD/static at1900 updates, DPT/full at23800 and DPT/static at24100.
The independent readouts now cover the same fresh native bank of50 OpenDrawer
ID and50 CloseFridge OOD cases, including both frozen PPO checkpoints.

| Policy | ID /50 | OOD /50 |
| --- | --- | --- |
| Initial PPO |33|28|
| Final PPO |32|30|
| AD/full, K0 |34|33|
| AD/full, separate K1 query |34|33|
| AD/static |29|33|
| DPT/full, K0 |29|32|
| DPT/full, separate K1 query |35|27|
| DPT/static |30|26|

Full students received one complete actual support episode before a separate
K1 query; support success was24/35 for AD and30/36 for DPT (ID/OOD).
AD K1 versus K0 has2/2 paired wins/losses on ID and6/6 on OOD (both exact p=1).
DPT has8/2 on ID (p=.1094) and3/8 on OOD (p=.2266).
Against initial/final PPO, AD K1 has5/4 and7/5 wins/losses on ID,
13/8 and12/9 on OOD. DPT K1 has5/3 and7/4 on ID,7/8 and5/8 on OOD.
None of these exploratory unadjusted paired tests establishes an improvement
over either PPO checkpoint. Full K1 versus its separately trained static
student is also reported in the paired evidence.

All1000 admitted native episode artifacts and100 shared canonical cases passed
the final CPU comparison: exact native scene/state/camera/public state,
original motor streams and checkpoint hashes, saved records and true simulator
feedback, with only qualified one-level uint8 render roundoff and original
policy pixels. A scheduler master-exit bug interrupted the last three AD
endpoints and one DPT query; only those missing episodes were recovered,
retaining597 AD and299 DPT records. The future launcher waits for successful
terminal receipts from every node in its own allocation. Source PPO streams
span different scenes across versions; these remain single-source-task and
single-fit-seed development results, and OOD means held out from student
training rather than foundation pretraining. Failed/partial/recovery costs
remain separate from admitted scientific episodes. The resource ledger covers
the inherited source and all closed fit/readout/diagnostic/recovery allocations;
the two subsequently authorized actual-attempt curve allocations are ongoing.
[Final combined result](../outputs/opendrawer-pairing-recovery-20261102-r1/COMBINED-RESULT.json),
[AD independent result](../outputs/opendrawer-ad-query-recovery-20261102-r1/RESULT.json),
[DPT full result](../outputs/opendrawer-dpt-full-query-recovery-20261102-r1/RESULT.json),
[DPT static result](../outputs/opendrawer-dpt-static-id-ood-20261102-r1/RESULT.json),
[resource ledger](../outputs/opendrawer-pairing-recovery-20261102-r1/RESOURCE-LEDGER.json).

## LIBERO

**Flora versus simulator PPO stopped before training (2026-11-01).** The
published few-shot PyTorch LIBERO40 SFT checkpoint succeeded on all64 baseline
episodes on task44, covering40 nominal TRAIN placements with24 second draws.
Each episode ran400 steps plus15 settling actions, with no training updates.
This exceeds the predeclared70% native-success limit, so neither PPO arm is
admitted. AUROC is undefined because the bank has no simulator failures. The
already-running Flora smoke-video qualification remains separate; no comparative
learning result exists. Historical C565 headroom does not establish headroom
for this different initializer.
[Result and limits](../outputs/flora-ppo-comparison-20261101-r1/RESULT.md).

**Frozen-motor DSRL.**

- *Original Pi0, task 57* (Baige `job-psgxs9n5b08w`): native 4/10 → learned
  10/10 at 500k updates. 10/10 is first reached at 40k updates (~50k training
  actions). This qualitatively reproduces the DSRL paper; its three-seed
  claims are not verified.
  [Evaluation](../outputs/dsrl-pi0-task57-20261101-r1/EVALUATION.md).
- *OpenPI40 pi0.5, task 57*: native and learned are both 10/10 (ceiling).
  Learned uses 14% fewer primitives; one regression panel at 390k. 28.7k
  GPU-s, 333k actions.
  [Result](../outputs/dsrl-pi05-task57-20261031-r1/RESULT.md).
- *OpenPI40 pi0.5, task 44 ("turn on the stove")*: screening 5/20 and
  confirmation 5/20. Fresh controls: native 6/20, untrained tied noise 13/20,
  initialized SAC 8/20. Learned panels: 1/20 at 20.6k updates, then 20/20 at
  every panel from 40k through 400k. Paused by the human at 406,780 updates,
  with full recovery, for the Flora comparison. The fixed 500k endpoint and the
  final native panel were never run.
  [Pause record](../outputs/owned-jobs-pause-20261101-r1/RESULT.md).

**Headroom screen** (384 episodes, 12 LIBERO90 candidates, states 1–8 × 2
draws): selected task 59 (native 3/16, tied 9/16), runner-up task 64. Task 44
was native 1/16, tied 10/16. Untrained tied noise often beats native sampling,
so any learned gain must be compared against it.
[Results](../outputs/libero90-pi05-headroom-screen-20261101-r1/RESULTS.md).

**Source40 → target90 meta-learning** (AD/DPT history controllers trained on
LIBERO40 source histories):

- Fixed100 phase: native 28/90; AD full/static 29/90 each; DPT full/static
  28/90 each. At K1, AD full, reset and native all score 6/9 on the same cases,
  with AD spending 4,014 actions vs native's 1,961. No memory, topline or
  sample-efficiency gain.
  [Result](../outputs/libero40-fixed100-target-plan-20261031-r1/RESULT.md).
- First full90 comparison: supervised and direct-PPO controllers fall below
  contemporaneous native. PPO source validation succeeds as execution but shows
  little adaptation headroom.
- All-40 existing-teacher assessment (319 valid source episodes): native
  78/80, final TRAIN SAC 78/80 in both mean and sample modes, tied warmup 70/79.
  The trained source actors add nothing over the native motor, so they cannot
  serve as teachers.
  [Run](../outputs/libero40-teacher-assessment-20261101-r1/).

## RoboTwin (Motus / VLRLA-style)

- FULL and trained-RESET learners each used ~1M source actions. Source success
  stays near 17% with no upward trend; source SAC reaches 277/2400 (11.5%).
  Two-goal targets: FULL 8/16 and RESET 8/16; cache-reset 7/16 and reward-blind
  9/16; saved native 6/16. No consistent memory or reward-input benefit.
- Steering-interface redesign: the physical teacher bank stays at 11/12, and
  fitted command/KV heads score 0/5 where native scores 0/5. No physical
  improvement; no teacher or student admitted.
  [Evaluation](robotwin/coefficient-steering/EVALUATION.md).

## ManiSkill / native pi0.5 bridge

Native pi0.5 CTM/XL PPO, the support/probe gate and the privileged tracker
completed and were audited on 2026-10-28. Cached CTM/XL/GTrXL screens and DPT
warm starts ran with one seed. None shows ICRL or a CTM advantage.
[Snapshot](../outputs/archive/coordination/coordination-20261031.md).

## Lessons carried forward

1. Establish headroom first: native success that is neither 0 nor ceiling.
   Compare against untrained tied noise as well as native sampling.
2. Get a positive control before testing a new interface or reward. A failure
   is uninterpretable without one; the Flora test uses simulator-reward PPO as
   arm S for this reason.
3. Source learners must actually improve before their histories can teach a
   meta-learner.
4. Report reset and reward-blind controls with every memory claim.
