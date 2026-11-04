# Findings to date

Consolidated 2026-11-01. Only the Meta-World ML10 comparison and the DSRL runs
are well-controlled enough to build on. The other threads are summarized here
so they are not repeated. Their run folders were deleted, except RoboTwin's
(restored under `outputs/archive/`); other links point to GitHub permalinks at
the pre-cleanup commit. Single-seed results are
descriptive, and none establishes a CTM, memory or in-context-RL advantage.

The frozen-steering history pipeline now exports and fits AD, DPT, DIT and
in-context IQL separately for velocity and all-layer K/V. Allocated-Linux CPU
qualification passed 98 core cases, all eight synthetic fit/export/five-attempt
routes and four exact interrupted value-stage recoveries at merged source
`414ab5d9`. These are
implementation and numerical checks, with zero real simulator episodes or
scientific student fits; IC-IQL omits S-ICQL's world-model prompt. See the
[qualification](../outputs/steering-four-method-pipeline-20261103-r1/RESULT.md).

The Nov4 five-task Flow-SDE PPO sources each completed 130 updates and
9,984,000 TRAIN robot controls. None passed either preregistered teacher gate:
TRAIN first-to-last-window gains were +4.61, +8.91, +1.56, +4.84 and −0.08
percentage points for OpenDrawer, Toaster, Drawer→Counter, Coffee and Stove;
their paired 64-case McNemar p-values were .167, .523, .115, .701 and .754.
All five remain reference sources. The full 650-cohort numerical corpus audit
passed, with zero initial likelihood gaps, finite losses and unchanged frozen
VLMs. These are source-learning results; no student or OOD adaptation result
is claimed. All 80 fixed-batch checkpoint likelihood reports are complete, with zero
initial-policy replay error on all five tasks. All five corpora and resumable final checkpoints passed archival/retention
closeout; all campaign GPU work is complete at 373.93 accounted GPU-hours. [Native results and evidence](../outputs/robocasa-flow-sde-10m-per-task-20261103-r1/RESULT.md).

A subsequent Nov4 code audit found a critic inconsistency in all five source
configurations: rollout averages the ten suffix value predictions, while PPO
replay uses only the selected denoising step. An unchanged-weight Linux CPU
reproduction has exact likelihood parity but unequal values; synthetic value
gaps can exceed the configured clipping threshold before SGD. The earlier
likelihood audits do not qualify critic parity. This is a confirmed
implementation defect, not an established cause of the failed teacher gates.
Repair and qualify the critic contract before further PPO.
[Bug audit and reproduction](../outputs/robocasa-flow-sde-bug-audit-20261104-r1/RESULT.md).

Matching-checkpoint GPU replay then confirmed the mismatch on256 saved Drawer
decisions at policies0,1,10,120, with exact likelihood and saved-mean parity.
One of64 initial-policy decisions crosses the.2 value-clip threshold and
suppresses an active critic gradient; none of192 later-policy decisions does.
Historical returns were held fixed, so corrected GAE and learning impact are
unmeasured. An independent CPU tensor/native-loss audit passes; the selected
suffix correction passes20 sampler cases without changing actions or RNG.
The short matched learning comparison is running, with no outcome claimed.
[Production critic evidence and limits](../outputs/robocasa-flow-sde-critic-parity-20261104-r1/RESULT.md).

A follow-up saved-record check rejects duplicated rollout noise for the
finished Drawer source: all eight ranks have distinct CUDA states/seeds and
actual initial-noise tensors in q1 and production. Same-index recovered SDE
innovations have maximum absolute pair correlation 0.00984 in the checked
decisions. PyTorch's implicit CUDA generators seed nondeterministically;
missing explicit HF-worker seeding does not imply cloned rank noise. This
check covers Drawer, not every task or draw.
[RNG evidence and scope](../outputs/robocasa-flow-sde-bug-audit-20261104-r1/RNG-RESULT.md).

The Nov4 dimension-drift replay completed all 80 checkpoint/cohort forwards;
independent Linux CPU reductions and original R3 per-sample likelihoods agree
exactly. Drawer's final conditional log ratio is −137.5202, with arm share
61.43% and gripper/base-mode share 24.44%; neither 70% concentration rule
fires. Different cohort denoising scales confound raw cross-task drift ratios.
This does not establish marginal action KL, behavioral change or a cause of
failed learning. [Verified decomposition and limits](../outputs/robocasa-flow-sde-dimension-drift-20261104-r1/RESULT.md).

A Nov4 native checkpoint-evaluation qualification preserves fixed eight-lane
batches and common sampler RNG, with zero initial-policy likelihood error.
Repeated initial-policy Drawer→Counter success was6/8 then8/8. Initial
physical/public states, all initial cameras and the first H50 actions match;
the first observed divergence is three one-level right-camera RGB differences
after50controls, before later action/trajectory divergence. This demonstrates
evaluation variability on those qualification cases, not a PPO bug or a
performance gain. No bitwise GPU-kernel requirement is introduced; the full
intermediate curve and fresh validation-selected TEST remain unrun.
[Qualification and limitations](../outputs/robocasa-flow-sde-checkpoint-selection-20261104-r1/RESULT.md).

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

The 2026-11-03 capacity extension completed the same32fresh five-task cases at600controls with8/16/32environments on one GPU: **83.04/138.93/202.08controls/s**, with sampled peak VRAM18.09/28.56/49.77GiB. Every setting completed32episodes and19,200physical controls with three-view video; total57,600controls, zeroSGD, exit0 and scheduler success. This allocation used64CPUs/256GiB RAM. The32-lane host-RAM peak was116.43GiB, and64CPUs/GPU cannot scale to eight GPUs on a180CPU Baige node. A separate22CPU/208GiB comparison of32and48lanes therefore measures a feasible per-node resource split; no eight-GPU scaling, steering-history overhead or Flora colocation follows from the standalone result. [Completed capacity evidence](../outputs/robocasa-rollout-capacity-1632-20261103-r1/RESULT-SUMMARY.json).

The node-compatible follow-up used22CPUs/208GiB RAM per GPU. **32and48lanes both completed**, at150.86and155.97controls/s respectively; the cohorts contain32and48episodes with the first32cases shared, so this is not a controlled speedup ratio. The48-lane peak was75.97GiB VRAM and175.37GiB host RAM, leaving19.63GiB VRAM and32.63GiB host RAM on the allocation. All80episodes have601-frame clocks and600physical controls, all48,000intents resolved, zeroSGD and scheduler success. Two eight-GPU jobs fit the advertised node resources at176CPUs/1664GiB each, but actual eight-GPU scaling and the velocity/KV history writer remain unqualified. Initial batch/single action differences reached0.11414; paired comparisons need matched actual production batching. Both capacity jobs together charge at most1217GPU-s, below1800. [48-lane audit](../outputs/robocasa-rollout-capacity-3248-20261103-r1/RESULT-SUMMARY.json).

A read-only audit of both 500-case velocity/KV zero-steering before panels found
**278 different first action chunks and 65 different first-success outcomes**.
All case descriptors, native noise tensors and zero steering vectors match;
170 first chunks differ despite identical images, public state and instruction.
The BF16 collectors have different asynchronous batch schedules on all eight
rank pairs, exposing the motor's previously measured batch sensitivity; the
within-batch zero/native check does not qualify cross-run reproducibility.
All 250 ID initial physical fingerprints match, but 19 OOD fingerprints differ,
including four different compiled scenes. Separate per-arm canonical roots
missed those cross-arm mismatches. Future paired comparisons need a fixed actual
batch plan and common physical case validation. The exact upstream causes of
the reset differences and the contribution of each mechanism to all outcome
flips remain unisolated. No new GPU/model/simulator work or active-run edits
were performed. [Baseline reproducibility audit](../outputs/robocasa-zero-steering-baseline-audit-20261103-r1/RESULT.md).

With one persistent official Flora TP1 engine (batch limit8, utilization.78), a four-lane worker completed ten mixed-task episodes and7,200 controls with all ten grades streamed before the final request and zero SGD. Collection from the first reset intent through the final complete video took147.65seconds (48.77controls/s/GPU); including native model loading it took183.02seconds (39.34controls/s). Sampled peak combined VRAM was88.01GiB. The first continuous-refill attempt exposed a forked reset child retaining another lane's FFmpeg pipe and blocking video EOF; spawned resets repaired it, with57/57 affected Linux core tests and a regression that fails on the old code. [Audited colocation measurement](../outputs/robocasa-joint-lane-qualification-20261102-r4/PARENT-REVIEW.json), [pipe evidence](../outputs/robocasa-joint-lane-qualification-20261102-r3/DEADLOCK-EVIDENCE.json).

This is one GPU's physical measurement, not an eight-GPU joint PPO iteration. After the real Flora warmup only20,664MiB was free: below the8-lane27,408MiB requirement and1,309MiB below the4-lane21,973MiB production requirement, which includes an8GiB operating reserve. At that pin the four-lane production reserve remained unmet. Sixteen native lanes have not been measured; eight was the largest and fastest standalone setting tested. That native-horizon measurement does not qualify the later600-control protocol.


The600-control migration now explicitly selects the declared simulator horizon instead of the registry's750 controls, while retaining native OOD horizons. With Flora utilization lowered from.78 to.75, batch capacity8 and the full8GiB operating reserve unchanged, ten fresh episodes completed6000controls and all ten streamed grades with one model load and zeroSGD. Warmup left23608MiB free, passing the21973MiB four-lane guard by1635MiB; eight lanes still fail. Collection took132.54seconds (45.27controls/s/GPU); including native loading,206.03seconds (29.12controls/s). Peak combined VRAM was85.14GiB, minimum free9.83GiB, and cold Flora plus its first eight warmup scores took473.50seconds. All video/frame-clock hashes,120records and6000resolved physical commands were audited;82/82 affected Linux core tests pass. This qualifies four-lane execution and the horizon change on one GPU. Fresh600-control calibration and joint before-panels remain required before joint SGD, and the prepared entries now enforce the bound lane count. [600-control colocation audit](../outputs/robocasa-joint-lane-qualification-20261102-r5/PARENT-REVIEW.json).

Four actual GPUs subsequently passed the same eight-video official Flora warmup, each leaving23608MiB free. That preparation job stopped before native policy loading because its qualification reference pointed to an absent output-directory copy; the exact bytes were present in the pinned Git checkout. It produced no fresh episodes, physical actions or optimizer updates, so fresh600-control gates and common-before panels remain unqualified. All four allocations closed; preparation accounts for3268GPU-seconds including prior startup attempts. Validate the full runtime input closure before model startup. [Four-GPU terminal audit](../outputs/robocasa-joint-h600-preparation-20261102-r3/RESULT.md).

The corrected four-GPU preparation subsequently completed all250 fresh600-control calibration cases and250 official grades, then40DEV/250ID/250OOD untouched-policy before episodes. All five gates pass without exceptions: native successes/AUROC were OpenDrawer31/50(.9338), CounterToCabinet28/50(.9058), DrawerToCounter14/50(.9821), TurnOffStove7/50(.9037) and CoffeeSetupMug15/50(.8476). Before-panel native successes were19/40DEV,90/250ID and126/250OOD, with full native OOD horizons and no OOD Flora scoring. The Linux CPU audit checked790 scoped case/seed plans,9930 records,496500 physical controls,all2370 three-view video hashes/frame counts,all grades/model receipts,complete initializer recovery and10720resolved intents; zero optimizer updates. All four allocations succeeded, charging16179GPU-seconds, or19447including the three preserved preparation attempts(5.402GPU-hours). Both inactive joint templates now bind this qualified source,fresh gates and common initializer; joint SGD stays NOT_ADMITTED. These are frozen-initializer measurements, not a learning result or an eight-/sixteen-lane qualification. [Audited preparation](../outputs/robocasa-joint-h600-preparation-20261102-r4/RESULT.md).

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
A subsequent failure-focused iteration uses only20 retained videos,14 failed
trajectories and6 success controls, with revision1 kept verbatim as its control.
Clearer target/opening/knob wording reduces false successes12→7, correcting6
and introducing1; it retains4/6 controls but loses1 and recovers1. All four
sampled Stove errors are corrected, while7/10 failed drawers remain accepted.
Direct0/1 labels under the same criteria accept11 failures; explicit numerical
predicates accept10 with graded or binary output. These ablations do not
improve on the clearer graded wording, and no variant qualifies production
reward. Cached revision1 hard grades reproduce20/20. This is an error-enriched
development sample, not population accuracy or first-success qualification;
no full-bank replay or new robot rollout was run. [Per-case readout and limits](../outputs/robocasa-flora-rubric-failure-sample-20261102-r1/RESULT.md).
A subsequent OpenDrawer-only check explicitly withholds Score5 until the
drawer visibly appears fully open. On the same14 selected videos, this raises
false successes7→8 out of10 failures and retains the same3/4 success controls:
zero old errors corrected, one new false success. The prior rubric reproduces
all14 hard grades. This stricter-sounding wording does not repair the sampled
errors and remains unqualified; it is not a population accuracy or first-success
result. [Exact wording and paired grades](../outputs/robocasa-flora-rubric-failure-sample-20261102-r2/RESULT.md).
A CPU diagnosis of retained OpenDrawer actions identifies two distinct errors.
Case112 opens the drawer below the uppermost target while the target stays
shut; case64 reaches native success at27.1s and then falls below the threshold
before termination. A success control first crosses at11.1s. All three replays
match every retained scene metadata field and all15 saved public16D robot-state
boundaries exactly; no full MuJoCo initial state was saved. A fourth example
is excluded before controls because semantic-tag ordering differs. The source
audit also corrects the physical interpretation: native0.95 is normalized by
0.55×drawer depth, so success begins at0.5225×depth, not95% of mechanical travel.
These examples justify testing uppermost-target tracking and endpoint wording,
and show why terminal failure counts cannot directly measure first-success
false positives. They do not establish the prevalence of these error modes.
[Source audit and retained-action diagnosis](../outputs/robocasa-opendrawer-rubric-diagnosis-20261102-r1/RESULT.md).
Explicit uppermost-target rules and a source-correct opening/endpoint rule also
fail on the same14-video panel: both give7/10 full-score terminal failures and
retain2/4 controls, versus7/10 and3/4 for verbatim1.1. Both fix88, introduce272,
and lose128; neither fixes diagnosed112 or64. All14 baseline full-success
decisions reproduce, though one lower grade changes2→1. Six causal prefixes on
two replay-qualified trajectories bracket both native first crossings, but a
prefix after the drawer is pushed back still gets5. This is diagnostic evidence,
not first-success qualification. [Paired grades, rationale limits and prefixes](../outputs/robocasa-flora-rubric-failure-sample-20261102-r3/RESULT.md).
Short direct task instructions also trade false-positive rejection for lost
terminal controls. On diagnosed112/64 plus four success controls,1.6 fixes112
but rejects every true control (1/2 full-score failures,0/4 controls, versus
2/2 and3/4 for verbatim1.1);1.7 gives2/2 and1/4, and1.8 gives2/2 and0/4.
All six baseline grades reproduce. None passes its frozen promotion rule.
[Exact task texts and paired results](../outputs/robocasa-flora-rubric-short-sample-20261103-r1/RESULT.md).
A causal-prefix check on six retained trajectories gives a narrower positive
result and a remaining tradeoff. On35 selected prefixes,1.6 removes all10
sampled pre-success/wrong-target full scores from1.1, including every wrong-
drawer112 probe. It recognises64/72/264 after native success, but misses128/144
that1.1 accepts; both have three episodes with a first sampled positive after
native success. A post-hoc Score>=4/two-probe screen recognises4/5 successful
episodes with no observed false trigger, still missing128. Only112 has the
full50-control probe grid; successful cases have four selected probes each.
These results neither qualify an online first-success detector nor justify
promoting the fitted threshold. The corrected retry completed70 labels,
including8 recovered baseline outputs, with673 worker GPU-seconds; the failed
allocation remains separately charged520. No new robot actions or SGD.
[Per-case times, raw scores and limits](../outputs/robocasa-flora-first-success-rubric-20261103-r2/RESULT.md).
A subsequent frozen comparison on the same35 prefixes does not repair these
losses. Adding clearest-view guidance gives1/20 false full scores and recognises
3/5 successful episodes at valid sampled times; adding native sufficient-opening
and first-success wording gives3/20 and1/5. Both still score successful128
prefixes1. Four repeated1.6 anchors reproduce exactly. All74 new outputs and
35 artifact hashes are verified, with0 new robot actions or SGD. No candidate
passes the prespecified development screen. These rewrites leave the system
and final grading instruction asking for terminal success; that motivated the
following system-rubric comparison.
[Exact changes, paired results and prompt limits](../outputs/robocasa-flora-first-success-rubric-20261103-r3/RESULT.md).
Preserving the native task and moving the verbatim1.6 rubric into system grading
also fails:5/20 false full scores and3/5 correctly timed successful episodes.
It recognises128 only at450, while reintroducing wrong-drawer112 errors.
A second composite change puts1.10 into system grading and aligns the generic
system/final question to first success:4/20 false full scores,1/5 correctly
timed successes, and128 still missed. All74 new grades,4 exact repeated anchors,
47 exported artifact hashes and74 actual renderer message hashes are verified.
CPU post-hoc two-probe confirmation removes observed false triggers but detects
only4/5 successful cases under either candidate, still missing128; these are
50-control-spaced prefix judgments, not consecutive frames or independent
validation. No rubric is promoted. Worker cost670 GPU-s,0 new controls/SGD;
the allocation succeeded and released. This motivated testing recent causal
chunks with the explicit criterion held fixed.
[System-message comparison, raw grades and limits](../outputs/robocasa-flora-system-rubric-20261103-r1/RESULT.md).
Recent chunks worsen recognition under fixed1.6. Last50/100-control windows
each give only1 full score on15 post-success endpoints, versus5 for full
prefixes, while all three inputs give0/20 false full scores. Valid raw first
detections fall from3/5 to0/5 and1/5. The predeclared two-Score5-probe rule
confirms0/5 successes for either window;128 remains missed. Post-hoc threshold4
confirmation gives0/5 and3/5, still missing128. All74 new grades,4 repeated
anchors,70 exported hashes and74 actual prompt/frame-read records verify;
8 CPU chunk decodes match direct source pixels. No production change, controls
or SGD;598 workerGPU-s, allocation succeeded/released. Removing early context
and changing sampled-frame count are coupled, so this does not isolate the
cause. Next wording work should separate target identification from opening
depth on full prefixes.
[Chunk protocol, per-case scores and limits](../outputs/robocasa-flora-drawer-chunk-rubric-20261103-r1/RESULT.md).
A full-prefix2×2 clause ablation isolates a useful wording tradeoff. With the
original target-description block, both with and without the depth sentence,
wrong-drawer112 is rejected but successful128 remains missed. Without that
block, both conditions recover128 at450 (native384) and falsely confirm112
at200. The predeclared two-score>=4 rule gives4/5 valid successes with0 false
accepted episodes for either target-block condition, versus5/5 and1 false
for either condition without it. Depth-only has no Score5 at all, illustrating
why zero false full scores alone is insufficient. All109 new outputs,4 exact
repeats,16 receipts,109 actual message audits and62 exported hashes verify.
No arm passes the development screen; no production change. Same reused sparse
six-video panel,0 controls/SGD,756 workerGPU-s. The next target-description
revision should test removing stack presuppositions while retaining an explicit
wrong-drawer rule; the cause inside Flora is not established.
[Clause texts, scores and limits](../outputs/robocasa-flora-drawer-clause-ablation-20261103-r1/RESULT.md).
Deleting only the stack-presupposition sentence from1.6 does not fix128:
all four grades remain1, two-probe>=4 detection stays4/5 with0 false episodes,
and valid raw detections fall3→2. A compact countertop description retaining
explicit wrong-drawer rejection also misses128 and falsely confirms112 at350
(4/5 valid successful cases,1 false accepted episode). Both fail the frozen
screen.74 new grades,4 exact repeats,11 receipts,74 actual message audits and47
artifact hashes verify;687 workerGPU-s,0 new controls/SGD. This rules out the
stack sentence alone as a sufficient explanation under the remaining1.6 text.
The positive location and negative exclusion clauses remain coupled in the
compact condition; neither is established as the cause. No promotion.
[Target wording and per-case results](../outputs/robocasa-flora-drawer-target-wording-20261103-r1/RESULT.md).
Separating location from exclusion also fails on the same35 prefixes. Adding
only the countertop-location sentence to the depth-only base lowers128's
350/400/450/750 grades2/4/4/4→1/2/3/3, while wrong-drawer112 is still confirmed
at200. Exclusion alone briefly gives128 a5 at400, then3 at450, and falsely
confirms112 at350. Both fresh conditions detect4/5 successful cases with1
false accepted episode under the frozen two-score>=4 rule. Neither passes.
All74 new grades,4 exact repeats,11 receipts,74 message audits and47 hashes
verify;175 paired labels include105 recovered grades.698 workerGPU-s,0 new
controls/SGD. This location sentence's effect does not require the negative
clause; the model's internal cause remains unknown. The next diagnosis should
add other retained never-success failures before further tuning to112 alone.
[Factorial texts, scores and limits](../outputs/robocasa-flora-drawer-location-exclusion-20261103-r1/RESULT.md).
Four additional retained failure replays qualify on allocated Linux CPU, with
all40 saved metadata fields and60 public16D state boundaries matching exactly.
Three never succeed:376 opens another drawer (target peak7.58% fixture depth),
256 opens the target50.22%, and336 opens it33.96%. Native requires52.25%, so
'roughly half' is too loose as a binary-success specification; this is not
proof of Flora's internal reason for its old grade.216 succeeds at control212
(10.6s), then closes to11.12% depth and ends as a terminal failure. It must be
an ever-success case for first-success analysis.3000 retained controls replayed,
0 new sampling/motor/optimizer/GPU calls,366.97s CPU wall. All7 exported hashes
verify. These selected examples establish additional failure modes, not their
population prevalence; the missing-full-MuJoCo-state caveat remains.
[Replay evidence and next small panel](../outputs/robocasa-flora-drawer-extra-failures-20261103-r1/RESULT.md).
An exact52.25%-depth wording comparison on10 retained trajectories/45 prefixes
also fails: both shorter candidates give0/27 false full scores but0/18 true
post-success full scores, with0/6 valid confirmed first detections. Four1.6
anchors reproduce. The expanded reference exposes further errors: near-boundary
256 and wrong-drawer376 both score4/4 at700/750, so accepting4 admits failures;
insufficient-opening336 scores5 at700. The numerical candidates suppress5
without repairing recognition, including128.104 new outputs,15 receipts,104
actual message audits and60 exported hashes verify;748 workerGPU-s,0 new
controls/SGD. Neither rubric nor a lower threshold is promoted. This reused,
error-enriched sparse panel supplies no independent or online qualification.
[Exact texts, per-case scores and limits](../outputs/robocasa-flora-drawer-exact-threshold-20261103-r1/RESULT.md).
A matched system-rubric comparison tests explicit meanings for grades1–5 on
the same10-video/45-prefix panel. Adding grade anchors raises true post-success
full scores11→15/18 but false full scores18→19/27; both variants falsely
confirm all4 never-success cases.128 gets5 only at750,366 controls after native
success384, so timely detection remains missed. Both also score72/264 as5
before native first success. No candidate passes the frozen confirmed-Score5
screen or enters PPO.94 new grades,4 exact repeat anchors,13 receipts,94 actual
message audits and54 artifact hashes verify;643 workerGPU-s,0 controls/SGD.
[Matched rubric texts and per-case grades](../outputs/robocasa-flora-drawer-grade-rubric-20261103-r1/RESULT.md).
An accompanying2.223s Linux CPU audit decodes14 raw three-view collages for the
4 added cases, verifying12 video hashes and5 exported artifacts. The selected
stills place several target regions near frame edges or behind occlusions;
the50.22%/52.25% distinction cannot be reliably measured from these stills.
This does not prove that all actual sampled frames lack evidence or establish
the cause of a Flora grade. The next input audit should inspect those actual
sampled sequences before further rubric search.0 model/simulator calls; these
annotated human-inspection sheets never enter the scoring comparison.
[Visual evidence and limits](../outputs/robocasa-flora-drawer-visual-audit-20261103-r1/RESULT.md).
A follow-up CPU reconstruction inspects all256 protocol-selected collages for
112@350,256@750,376@750 and128@450.128 includes10 post-native-first-success
frames, controls386–450, with opening/interior visible in the left view, despite
all3 preceding reference/candidate grades being1 at450. Omission of all source
opening frames is therefore insufficient to explain this selected miss; no
claim about saved post-processor tensors or model internals is made. Wrong-
drawer112/376 contain near-closed upper fronts alongside lower-drawer motion.
256 shows an opening but lacks a clear full-depth reference for a reliable
human52.25% boundary measurement.32 lossless pages/33 hashes verify;19.590CPU-s,
0 model/simulator calls. Three untested minimal texts now separate an exact1.6
boundary from initial-target tracking and a conditional shallow-drawer hint.
No reward promotion. [Sampled-frame evidence and next screen](../outputs/robocasa-flora-drawer-sampled-frames-20261103-r1/RESULT.md).
A subsequent26-prefix/10-video screen isolates exact native extent and two
minimal target hints. Exact1.6 boundary alone gives0/14 false full scores but
only3/12 post-success full scores and0/6 timely confirmed detections. Adding
initial-target tracking raises true full scores to6 and raw timely detections
to5/6, but falsely scores wrong-drawer376@700 as5; only1/6 confirms. The
shallow-target hint produces2 false full scores and1/6 confirmed successes.
128 remains1 at350/400/450 under all3 fresh texts. All82 actual raw RGB inputs
match CPU references, including the audited128 sequence; different decoded
frames therefore do not explain this run's miss. Post-processor information
and model internals remain unaudited. All82 new grades,104 paired labels,13
receipts,82 prompt/RGB audits,4 repeated anchors and67 exported hashes verify;
674 workerGPU-s,0 new controls/SGD. No candidate is promoted. Next cheap
step: inspect the official processor path before more wording changes.
[Exact texts and per-case results](../outputs/robocasa-flora-drawer-target-hints-20261103-r1/RESULT.md).
A pinned CPU processor reconstruction now preserves all 256 sampled frames
for 112@350, 256@750, 376@750 and 128@450 byte for byte, including after the
bfloat16 input conversion. The full HF path and direct video processor agree:
64 original 320×960 frames become a 32×20×60 grid with 38,400×1,536 feature
values; temporal pairs are packed, not averaged. The actual mean/std are
0.5 tuples, matching the saved config; a failed audit guard had compared them
with a list. Six artifact hashes, all source-frame hashes and four previous
rendered prompts verify after download. The passing identity is processor-
audit **r3**, source d565865a; r1/r2 remain failed attempts. 381.064 s audit
wall / 402.652 s process wall, exit 0, no weights, model inference or GPU work.
This reconstructed preprocessing stage does not erase the selected evidence;
previous GPU tensors and learned semantic retention remain unverified. The
128 opening lies near the lower image edge, motivating the bounded framing
comparison reported below; held-out qualification is still required.
[Passing receipt and limits](../outputs/robocasa-flora-drawer-processor-audit-20261103-r3/RESULT.md),
[consolidated rubric evidence](../outputs/robocasa-flora-drawer-processor-audit-20261103-r3/RUBRIC-REVIEW.md).
The successor frame-edge contrast has now passed CPU preparation on19 distinct
retained prefixes:1,216 raw RGB hashes,19 actual processor feature references,
all four previous RGB/float32/bfloat16 references, and40 prepared message
structures verify after download. Source43d9ca5d,426.152s CPU process,exit0,
zero weights/inference/GPU. This qualifies inputs for the small comparison,
not its framing hypothesis. The two fresh criteria share native fixture-depth
and first-success semantics and isolate one frame-edge clause.
[CPU evidence](../outputs/robocasa-flora-drawer-frame-edge-20261103-r1/CPU-RESULT.md).
The completed40-grade comparison rejects both: each gives6/10 negative
prefixes score5 and falsely confirms3/4 never-success cases. The clause raises
post-success score5 endpoints4→5/8 and timely two-probe detections1→2/3,
but128 remains missed and216's later-closed prefix still scores1. All4 anchor
repeats match. Actual vLLM processor outputs are bfloat16 and all19 feature
hashes match CPU references after that same cast; unrounded CPUfloat32 is
not a like-for-like comparison. All40 raw/message audits and54 exported file
hashes verify. Offline first-success latching preserves recognized successes
but neither one nor two score5 probes removes the same3 false episodes.
No reward promotion, cutoff retuning or renewed lexical fitting follows.
Baige job-pb53j1e4jom1 succeeded/released13:45:32 Shanghai,575workerGPU-s
(590scheduler seconds), no robot actions/SGD. The documented prefix/chunk/
clause/wording subtotal becomes7,925workerGPU-s, excluding earlier terminal
video work. [Exact texts, all scores, input audits and next diagnostic boundary](../outputs/robocasa-flora-drawer-frame-edge-20261103-r1/RESULT.md).
A current-state recognition diagnostic on18 reused endpoints rejects all four
variants:64-frame current-state grading accepts7/12 negatives and5/6 positives;
its condition checklist accepts10/12 negatives and6/6 positives. Reset/current
pairs reject all18 under either reporting contract. Grade AUROC is0.4167 for
prefixes and0.4444 for pairs, under current-state labels, not the earlier
ever-success or H600 terminal labels. All4 reference grades reproduce and all37
actual bfloat16 processor inputs match CPU references. These input checks do not
qualify learned recognition or the altered reporting protocol. Baige
job-6zq2cpuq1c7c succeeds/releases15:30:24 Shanghai,1,003wrapper GPU-s
(1,010scheduler seconds),0new controls/SGD. Raw receipts remain on Baige after
the organization blocked the archive download; aggregate evidence was inspected
there, not independently hash-verified locally. No candidate or cutoff promoted.
The next bounded audit concerns the official per-step rubric/thinking interface
and causal per-camera evidence, whose CLI defaults differ from terminal scoring.
[Results and evidence limits](../outputs/robocasa-flora-current-state-20261103-r2/RESULT.md).
A subsequent CPU geometry audit finds all six selected targets at z=0.785,
with no drawer above the target in its stack, and matching scene metadata.
Thus an incorrectly assumed uppermost target does not explain the128/144
misses on this panel; this does not establish equivalence of the native height
filter and uppermost geometry across all layouts. Six resets,0 policy controls
or motor calls. [Geometry evidence](../outputs/robocasa-flora-first-success-rubric-20261103-r2/TARGET-AUDIT.json).
An offline threshold screen on the same634 retained videos also leaves Flora
unqualified: on the original TRAIN scores, reward>=.90 gives89 true positives
and9 false positives (90.8% precision,44.3% recall); >=.95 gives15 and3
(83.3%,7.5%). No tested expected-grade/P5 threshold met the proposed precision,
FPR and recall screen on the fit cases. Terminal labels cannot establish
first-success precision or timing. This is development evidence, not an
independently qualified reward repair. [Readout and full sweeps](../outputs/gate-readiness-20261102-r1/RESULT.md).

The four gate0 native8 banks separate motor noise from selected case difficulty:
59/80 anchor-failure cases have mixed outcomes,21 fail all eight draws, and
none succeed all eight;28 have3–5 successes. OpenDrawer alone has19/20 mixed
cases and44.5% within-case pair disagreement. Its oracle best-of8 recovers19/20
anchor failures. These600-control, failure-selected banks show resampling
headroom, not an executable selector or a memory effect; eight failures do not
establish impossibility. [Case counts, intervals and variance decomposition](../outputs/gate-readiness-20261102-r1/OFFLINE.json).
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
the inherited source and all closed fit/readout/diagnostic/recovery allocations.
The subsequent actual-attempt curves and their terminal accounting are complete
below.
[Final combined result](../outputs/opendrawer-pairing-recovery-20261102-r1/COMBINED-RESULT.json),
[AD independent result](../outputs/opendrawer-ad-query-recovery-20261102-r1/RESULT.json),
[DPT full result](../outputs/opendrawer-dpt-full-query-recovery-20261102-r1/RESULT.json),
[DPT static result](../outputs/opendrawer-dpt-static-id-ood-20261102-r1/RESULT.json),
[resource ledger](../outputs/opendrawer-pairing-recovery-20261102-r1/RESOURCE-LEDGER.json).


The actual-attempt readout completed independently for AD and DPT, with50 ID
and50 OOD cases per model. Memory contains previous actual attempts, the same
case resets between attempts, and each new case starts with empty memory.
Weights stay frozen. All500 native endpoints per model passed final
physical/camera/truth/hash verification, all16 worker receipts pass, and both
allocations released successfully (DPT18:43:37, AD18:46:01 Shanghai).

| Actual attempt, successes /50 | 1 | 2 | 3 | 4 | 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| AD / ID |24|24|28|33|38|
| AD / OOD |35|34|34|35|28|
| DPT / ID |30|33|27|32|31|
| DPT / OOD |36|31|35|38|36|

AD's ID attempt5-versus1 comparison has16 paired wins,2 losses and32 ties
(unadjusted exact p=.0013123); its OOD comparison has9/16 wins/losses
(p=.2295). DPT's endpoint change is+1 ID and0 OOD. This is an exploratory
ID attempt association for AD, not an established learning or memory effect.
The motor stream changes with attempt number (matched across models). The
matched reset comparison below controls that source of variation. Separately scored PPO/static p0
controls cannot substitute for that repeated-attempt control. These curves
measure success on each actual attempt, not cumulative ever-success;75 ID/90
OOD tokens exceed the64-token training window while fitting128-token capacity.
The matched64-token intervention below tests this concern; new fits use a128-token
training window.
The single-source-task/fit-seed and renderer qualifications above still apply.

The curves reuse546 complete practices and collect454 new episodes, charging
351600 conservative controls and25106 measured process GPU-seconds for the
fresh continuations. The complete accounting retains every excluded failure,
partial call, obsolete query and recovery, with queue-inclusive bounds separate
from measured clocks. Source checkpoints and pinned worktrees remain preserved.
[Curves, confidence intervals and paired tests](../outputs/opendrawer-attempts50-20261102-r1/COMBINED-RESULT.json),
[figure](../outputs/opendrawer-attempts50-20261102-r1/figures/attempt-success.png),
[final resource accounting](../outputs/opendrawer-attempts50-20261102-r1/FINAL-ACCOUNTING.json),
[consolidated recent results and next-decision rationale](../outputs/opendrawer-results-rollup-20261102-r1/RESULTS.md).

The native-resampling arm now quantifies noise on80 selected anchor-failure
cases (four tasks, eight draws each,600 controls):21 cases fail all eight,
59 have mixed outcomes and none succeeds all eight. OpenDrawer redraws
succeed79/160 times and recover19/20 anchor failures at least once;12/20
cases have3–5 successes. These conditional TRAIN banks and horizons differ
from the AD/DPT held-out panel. Eight draws cannot establish p=0 or p=1;
oracle best-of8 recovery is a diagnostic, not an ICRL ceiling. This makes
matched empty-memory motor streams essential before claiming memory gains.
[Per-case estimates and uncertainty](../outputs/opendrawer-memory-controls-20261102-r1/NOISE.md).

**Matched memory/context controls completed (2026-11-03).** Frozen AD/DPT now
have matched reset and64-token raw-history controls on the same50 ID and50 OOD
cases and p1–p5 motor streams. All1100 collected endpoints (267r1+785r2+48r4)
and900 reused control endpoints are verified, with all1000 original full-history
episodes independently checked. Reset clears previous feedback and state before
each episode. Window64 rebuilds the last64 raw input tokens, including the query,
after eviction. Weights, native resets and historical750/900 horizons stay fixed.

| Attempt5 final successes /50 | Full memory | Reset each episode | Window64 |
| --- | ---: | ---: | ---: |
| AD / OpenDrawer ID |38|31|36|
| AD / CloseFridge OOD |28|35|37|
| DPT / OpenDrawer ID |31|33|33|
| DPT / CloseFridge OOD |36|32|35|

There is **no established memory gain** in these exploratory cohorts. Across
all five streams, full-minus-reset effects in percentage points (paired
case-bootstrap95% intervals) are AD ID+3.6[-1.2,8.4], AD OOD−2.0[-7.6,3.6],
DPT ID+0.8[-2.8,4.4], DPT OOD+0.8[-5.2,6.8]. Each interval includes zero.
AD's ID attempt5 advantage is7/50 (10 paired wins/3 losses; raw exact
p=.0923, Holm p=1). Its original attempt1→5 rise therefore does not by itself
establish learning from memory. The attempt5-minus1 difference in differences
is the same14percentage points because attempt1 is shared by construction.

The clearest context diagnostic is AD OOD attempt5: window64 scores37/50
versus28/50 with full history,10 paired wins/1 loss; window-minus-full is
18percentage points with case-bootstrap95% interval[6,30]. Raw exact p=.01172,
but Holm p=.46875 across the40 declared per-attempt comparisons. No comparison
survives that correction. This supports investigating long/older-history harm
for this checkpoint and panel; truncation also removes older feedback, so it
does not isolate positional extrapolation as the cause. DPT shows no clear
window64 benefit. Fifty case clusters, repeated streams and one source task/fit
seed remain the limits; a larger confirmatory cohort is needed for a claim.

The final one-GPU recovery succeeded and released its allocation at12:28:22
Shanghai, collecting only the missing48 episodes/36000 controls. All eight
logical workers completed with unchanged weights. The conservative full-campaign
GPU charge, including failed attempts, is65556/86400GPU-seconds (18.21/24GPU-h);
all original checkpoints, failed artifacts and immutable worktrees are retained.
[Full per-attempt curves and paired statistics](../outputs/opendrawer-memory-controls-20261103-r4/RESULT.json),
[readable curves](../outputs/opendrawer-memory-controls-20261103-r4/RESULT.md),
[figure](../outputs/opendrawer-memory-controls-20261103-r4/attempt-curves.png),
[independent audit and allocation reconciliation](../outputs/opendrawer-memory-controls-20261103-r4/FINAL-AUDIT.json).

**RoboCasa Stage 0 interface screen completed (2026-11-02).** On 20 TRAIN
cases/task selected by an independent native failure anchor, eight new draws
at 600 controls gave native successes 79/43/44/16 of 160 on
OpenDrawer/DrawerToCounter/Coffee/Stove. Action residual0.25 scored
81/42/31/12, residual0.5 42/38/16/7, bounded tied-noise32 1/0/0/0,
conditioning 89/44/47/20 and velocity 84/49/53/20. Positive per-task changes
do not reach p<.05 in exploratory exact tests that preserve eight-draw case
clusters. Native best-of-eight recovers19/15/16/9 of20 cases; it is simulator
oracle search, not a deployed selector or a bound on all chunkwise policies.
The realizable eight-candidate value-heuristic selector scores13/5/4/1 of20
versus matched native draw0 12/4/4/3 (paired p=1/1/1/.625); its scorer is not
a qualified Q-function or a history policy.

All six planned families were attempted, but the direct-head arm is a
**distillation diagnostic that bypasses the expert outputs**, not frozen-expert
steering. Only OpenDrawer qualified (8/8 versus native3/8 on eight fresh TRAIN
competence cases); its109/160 full-bank score uses target cases that also
contributed imitation training/validation and model selection, so possible
overfitting is unresolved and this score is excluded from steering rankings.
The other three direct families remain incomplete after fit/competence gates.
All35 banks and349 carried outcomes passed final Linux CPU truth/video,
checkpoint, control and physical/public-pairing audits. The213 carried native
rows lack actual chunk records and are not AD/DPT training histories.
The campaign charged2,582,096 controls and conservatively32,652 GPU-seconds
(9.07hours), including both failed attempts and the full four-GPU scheduler
lifetime;16 old unresolved calls remain charged. Zero PPO updates occurred.
Following this screen the human chose velocity residual and KV conditioning as
the two Stage 1 PPO arms (2026-11-02), replacing the action residual; the
screen itself does not establish that either is learnable by PPO.
Actual improving simulator- and Flora-reward PPO curves remain gates for new
AD/DPT histories. [Six-family results and limits](../outputs/robocasa-interface-screen-20261102-r3/RESULT.md),
[paired evidence](../outputs/robocasa-interface-screen-20261102-r3/FINAL-RESULT.json),
[terminal costs](../outputs/robocasa-interface-screen-20261102-r3/TERMINAL-ACCOUNTING.json).

**OpenDrawer simulator-reward action-expert Flow-SDE PPO completed
(2026-11-02).** Eighteen updates / 288 optimizer steps retain a frozen VLM,
but do not establish learning: TRAIN first-three versus last-three banks are
221→224/384 (+0.78 points), failing the declared ≥10-point gate. On the same
64 native cases, first success is 35→43 (18 recoveries / 10 regressions,
exact two-sided McNemar p=.18493); success at end is 34→43 (19/10,
p=.13605). Neither paired change is significant at .05. Physical/camera/public
reset identity matched; RGB stayed diagnostic-only. The positive held-out point
estimate is uncertain, not proof of zero effect.

Adam/LR continuation, finite updates, all 18 frozen-VLM checks and the full
final recovery checkpoint passed. Critic explained variance rises to .3437,
while 14,079/18,432 H50 decisions have valid PPO masks and 1,336 positive
rewards exactly match native TRAIN successes. Native per-update behavior-KL
proxies are .0219–.1287; they do not measure original-SFT cumulative drift.
This run does not identify a causal explanation for weak actor improvement.
The final eight-GPU job succeeded/released at 20:21:39 Shanghai. All attempts
charge 975,200/1M controls and conservatively 34,162 GPU-seconds (9.49 hours).
Raw histories remain candidate/reference data; compatible velocity/KV labels
and an improving matched frozen-motor teacher have not been established here.
[Result and curves](../outputs/robocasa-opendrawer-rlinf-sim-ppo-20261102-r1/RESULT.md),
[history compatibility and gaps](../outputs/robocasa-opendrawer-rlinf-sim-ppo-20261102-r1/HISTORY-AUDIT.md).

**Flow-SDE training-mode correction (2026-11-03).** The first 10M qualification
found an unchanged-policy likelihood mismatch before SGD: maximum chunk log
ratios were0.1243/0.2874 on the two actor ranks. GPU replay reproduces saved
likelihoods exactly in eval mode, including shuffled batches; training mode
changes the frozen SigLIP image encoder's attention path. Keeping only that
frozen encoder in eval mode restores exact likelihoods with expert gradients
enabled on both tested batches. This applies to the eighteen-update teacher's
same native training path, so its KL/clip diagnostics and source-learning
conclusion are not a qualified test of the intended PPO recipe. Preserve its
observed success counts and histories as reference data; downstream fits using
those histories retain that source limitation. The separate velocity/KV
steering workers explicitly keep the motor in eval mode. Corrected source
`0988b5f9` passes39Linux tests and all five native eight-GPU first updates:
zero initial likelihood error on both actor ranks,24optimizer steps per task,
finite losses, unchanged frozen VLM, and complete resumable boundaries.
Each task's1536actor samples join uniquely to saved environment episodes;
all three camera tensors and flow chains have byte-exact retained copies.
This qualifies execution and retention, not teacher improvement at10M.
[Diagnosis and scope](../outputs/robocasa-flow-sde-10m-per-task-20261103-r1/VISION-MODE-BUG.md),
[GPU replay](../outputs/robocasa-flow-sde-10m-per-task-20261103-r1/PARITY-REPLAY-R2.json).

### Native atomic task screen (2026-11-03)

A native frozen pi0.5 sweep completed all 240 episodes on 12 previously
unmeasured `atomic_seen` tasks (20 per task), 147,000 physical controls and
zero SGD. PickPlaceCounterToStove scored 10/20 first and end success, providing
one new pick-and-place candidate within the coarse 5–70% range; OpenCabinet
and PickPlaceToasterToCounter scored 15/20, and PickPlaceSinkToCounter 19/20.
SlideDishwasherRack scored 14 first / 13 end successes; TurnOnMicrowave 6 / 4,
so first and terminal success must remain separate. All 720 videos and 2,940
native chunk/noise schedules were independently audited. These single-bank
point estimates need 50-case qualification for newly selected tasks; they do
not establish learning or fix the final ID/OOD portfolio.
[All twelve rates and Wilson intervals](../outputs/robocasa-atomic-baseline-sweep-20261103-r1/RESULT.md),
[audit](../outputs/robocasa-atomic-baseline-sweep-20261103-r1/FINAL-AUDIT.json).

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

## Stove replacement native qualification (2026-11-04)

At 600 controls, unchanged released Human300 pi0.5 first succeeded on
18/50 TurnOnSinkFaucet cases (36%; Wilson 95% 24.1–49.9%) and 32/50
TurnOnElectricKettle cases (64%; 50.1–75.9%). End successes were 18/50 and
29/50; all 32 Kettle first successes already occurred by its native 450-control
horizon. The predeclared Faucet-first 3–35/50 rule selects Faucet. An independent
Linux CPU audit verified all 100 cases, 1,200 case/noise/action chunks, 300 fully
decoded videos (180,300 frames), 60,000 physical controls and zero unresolved
intents or SGD. Both GPUs are released; conservative total cost is 1,724 GPU-s
(0.479 GPU-h). This qualifies native headroom only; no new improving teacher,
Flora agreement or replacement PPO source is established. The subsequent
whole-portfolio / 30–60% preference review remains a separate decision.
[Qualified result](../outputs/robocasa-buttons-replacement-qualification-20261104-r1/RESULT.md),
[independent audit](../outputs/robocasa-buttons-replacement-qualification-20261104-r1/FINAL-AUDIT.json).

## Matched-denoising drift across the five PPO sources (2026-11-04)

All80 checkpoint loads /800forwards and independent float64 tensor reductions
passed. With16policy-zero samples at each of the same ten denoising indices,
Drawer→Counter's final variance-normalized mean signed log ratio is−0.3693,
close to OpenDrawer−0.3484; Coffee is−0.2572,Toaster−0.3380,and Stove−0.4746.
Drawer is not the most negative task at nine of ten matched indices, so the
predeclared uniform Drawer-specific excess is not supported. A last-step
difference remains (Drawer−122.565,OpenDrawer−82.541),with descriptive sample
SD49.917 and26.908. The original different-noise-scale comparison is not
evidence of a roughly100-fold Drawer-specific PPO problem. These saved
conditional transition likelihoods are not action marginal KL, behavioral
regression, or evidence that the critic bug caused the learning curve.
All50fixtures match original actor records,80staging/consumption joins and
private-copy cleanup pass,and all50initial likelihood replays are exact.
Baige job-pjgeuietm5of succeeded and released14:21:51; conservative cost4472GPU-s
(1.2422GPUh),inside4GPUh,with zero controls/sampling/SGD.
[Result and complete evidence](../outputs/robocasa-flow-sde-matched-denoise-drift-20261104-r1/RESULT.md).
