# Frozen-VLA steering: experiment summary

Objective: establish useful PPO and DSRL baselines, then find a meta-learning
method that improves robotic success and adaptation cost. The architecture is
open. More data is a hypothesis to test, not an explanation for every failure.

Current scientific readout,2026-10-31:

- LIBERO's completed first full90 comparison remains below native; no reliable
  memory benefit. Expanded collection/readout are ready, with compute pending.
- RoboTwin FULL and trainedRESET each finished approximately1M source actions.
  Source training stays near17% success in both, without an upward trend.
- Their two-goal target totals are both8/16. Same-FULL-weight cache reset7/16
  and rewardblind9/16 establish no consistent memory or reward-input benefit.
- A saved native16 panel is now verified on the same cases and underlying
  motor contract:6/16. Reuse adds no rollouts. The small observed steering
  gains do not establish a robust performance or memory improvement.
- Source SAC also finished100 rounds:277/2400 successes (11.54%) after
  1,003,790 source actions. Its quarter totals78/57/76/66 per600 show no
  sustained improvement. Weak source learning occurs in all three learners;
  a common cause is unisolated. Final source validation is2/24.
- Success by attempt1–4 also lacks a strong monotonic gain: FULL
  103/106/97/110 per600, RESET98/110/106/107 andSAC72/70/64/71.
  Later-attempt gains are not specific to FULL. These are source trials
  with different scheduled scenes, not a causal memory or held-out query test.
- SAC's final two-goal panel is complete: frozen6/8 bottle +2/8 stapler,
  total8/16; source-initialized online SAC5/8 +0/8, total5/16. Both use the
  same actual target cases as FULL/RESET. Online learning changes parameters
  inside episodes, so episode indices are not separate support/query budgets.
- No high-performance or sample-efficiency claim is established. Prioritize
  effective source learning and matched robot controls over more compatibility
  work or simply repeating a larger PPO collection with the same recipe.

Current priority: **DSRL first; native PPO afterward**, explicitly chosen by
the human. Train the noise controller while the entire official pi0.5 stays
frozen. The new single-seed task57 profile follows the authors' PixelSAC recipe
with pi0.5 H10 adaptation, varied training states0..39, matched query states40..49
and a500k optimizer-update stop. It has a fresh <=10GPUh/1M-action/64GiB ceiling;
implementation and new-profile qualification are complete:16/16 allocated CPU checks passed,80 worker-seconds/0GPU. Baige job-iudscqfntb4e was submitted at14:06:36Shanghai; actual GPU/motorC565/same-noise checks passed. At15:37:00 the run has480/480 training successes,6,769 replay chunks,
125,520 SAC updates and87,903 all-in actions. Completed learned panels through
actual120,220 updates preserve native10/10. The newest uses1,239 query
primitives versus native1,451; this continues the small-bank completion-cost
trend, without a success-rate, memory or overall sample-efficiency claim.
No failure/recovery histories have emerged. Recovery, milestone and sampled
raw/RPC hashes passed; no new journal traversal was performed. Continue the
fixed curve under its caps. Current operational status is in
[coordination](../outputs/coordination.md). [Saved check](../outputs/dsrl-pi05-task57-20261031-r1/CHECK-1536.md),
[interpretation](../outputs/dsrl-pi05-task57-20261031-r1/INTERPRETATION.md).
See [plan](../outputs/dsrl-pi05-task57-20261031-r1/PLAN.md).

The compatible collection records actual steering noise under the same frozen
motor: AD can predict that noise from ordered learning histories; DPT can use
a later competent final SAC noise teacher to label saved query observations.
Native PPO changes the action expert, so its noise seeds cannot automatically
label the original frozen motor. Physical-action learning remains possible.
See [data contract](../outputs/dsrl-pi05-task57-20261031-r1/DATA-CONTRACT.md).

Native PPO GPU dispatch is held. Its allocated CPU qualification completed:
274 conservative worker-seconds total, no GPU/actions/SGD; the latest three
changed data regressions passed. The explicit human instruction authorizes25
TOTAL iterations after the pilot passes, matched to the same128 starting-model
cases; this is queued behind DSRL, not an extra25 after the pilot's2. The
existing6GPUh pilot cap remains unchanged. AD100/500 fits are also held;
existing RoboTwin export continues under its separate owner.

The fixed100 LIBERO phase is now audited terminal: native28/90; AD full/static
29/90each; DPT full/static28/90each. ADK1full/reset/nativeK0all6/9with the same
success set; DPTK1full5/9versus reset6/9. AD spends4014actions for the same
6/9query successes native attains with1961. No positive memory, high topline
or sample-efficiency result. Full/static aggregate ties conceal per-case swaps.
Earlier100/500data/optimizer/native-wave changes prevent an early-stopping or
scaling conclusion. Recorded competing explanations and a proposed fixed-bank
contrast remain in the analysis; the new human PPO priority supersedes that
proposal. That result alone does not justify repeating the same source collection.

The completed486groups/522episodes cost172045actions,7126conservativeGPU-s
(6846scheduler-s) and8.61GB. All522trace/contextarchives,486group recoveries
and3541referenced files passed the saved-byte/array audit; archives were
rehashed, not restored, and the full RPC intent forest was not replayed.
See outputs/libero40-fixed100-target-plan-20261031-r1/{RESULT.md,cost-receipt.json}.
The old LIBERO monitor is stopped; no terminal job is polled as live. Closed
unused9074GPU-s are not borrowed for the native PPO pilot. Original32h1180s,
the pending25hsource expansion and deferredRoboTwin AD grant remain separate.

For an unexpected result, record the observation, what is known, competing
explanations, the next discriminating experiment, and the decision before
expanding that experiment. Verified numerical results stay in the method's
EVALUATION.md; live jobs and costs stay in outputs/coordination.md.

Execution correction: after a bounded runtime/data-integrity smoke, run source
collection in parallel across benchmarks and fit/evaluate initial students
from completed immutable batches while collection continues. Do not wait for
the entire corpus or serialize every optional diagnostic before the first
training attempt. Fresh PPO training still requires rollouts from its current
policy; saved SAC histories can train the offline students but cannot replace
that on-policy collection. The initial LIBERO batch has now been trained and
tested. Its negative transfer result and sparse recovery histories justify a
targeted recipe diagnosis before repeating the same collection at larger scale.

## RoboTwin source100 export is ready; GPU fitting stays deferred

Owner closeout154fdb970de397480db94f59760cd76149864ac8 records CPU R4
job-vth2qv6r4lxu asSucceeded/exit0, with admitted AD training arrays for
32,425 decisions /2,400 episodes /600 K4 histories from the existing1,003,790
source actions. Separate BC validation has328 decisions /24 episodes /6K4,
10,180 selected historical actions. Saved independent index/NPZ hashes,
42ZIP CRCs and606-case identity/disjointness/ancestry metadata passed. Root
read the committed closure and its exact bindings; this receipt is not a new
array-restoration or runtime audit by this chat.

Total CPU phase2,588 worker-seconds;4,612 unused seconds sealed. New GPU,
physical interaction and SGD are all zero; conservative storage96.728GB is
within128GiB. The owner deleted the completed CPU monitor. AD GPU fitting and
readout remain deferred with the separate10,800GPU-second/60k-action grant
unspent. This is dataset readiness, not evidence of a good teacher, a trained
student, native PPO replication or robot-performance improvement. The later
human DSRL-first priority supersedes the closure's older native-PPO wording.
[Immutable owner closure](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/154fdb970de397480db94f59760cd76149864ac8/outputs/robotwin-final100-ad-export-20261031-r4/CLOSURE.json).

## LIBERO: first transfer result is below native

- Source: frozen official OpenPI pi0.5 trained on LIBERO40; steering source
  collection uses all40. No LIBERO90/Plus/Pro data enter source pretraining.
- Completed source SAC data: 326,277 training actions, 527 K4 cases,
  2,108 episodes; separate validation 85,301 actions / 584 episodes.
- Four supervised controllers: AD/DPT, full history / separately trained
  static; 500 updates each, one training seed.
- All90 LIBERO90 goals, one initialization each: native **31/90**;
  AD full/static **26/90**, DPT full **26/90**, DPT static **27/90**.
- Fixed nine-case K1 panel: AD/DPT full **6/9**, same-weight episode reset
  **5/9**. Native and trained-static already reach **6/9** at K0.
  The sole memory discordance follows successful support, not failed-attempt
  recovery. Tiny Plus/Pro panels are ceiling results, not full benchmark scores.
- Evidence: [completed target results](../outputs/libero40-target-firstwave-plan-20261030-r1/RESULTS.md).

Known: all target groups finished; motor weights stayed fixed; paired actual
initial observations and query seeds matched. This makes a runtime interruption
or changed backbone an inadequate explanation of the deficit. It does not
establish that the learned controllers or their source teachers are good.

Unresolved explanations and tests:

| Explanation | Discriminating experiment / decision |
| --- | --- |
| Tying or bounding the noise changes native behavior before useful learning | Contemporaneous all90 native / zero / tied unbounded Gaussian / tied clipped Gaussian. Separate interface loss from learned-controller loss. |
| Source teachers provide weak or near-ceiling histories | Evaluate each corresponding task's actual trained PixelSAC mean and sample against native on disjoint source states; count corpus failures and actual failure-to-success trials. Saved optimizer steps alone do not prove a good teacher. |
| Insufficient or unstable fitting | Separate whole-case data amount from optimizer budget; compare fixed data/checkpoints before collecting more. Do not select source checkpoints using target outcomes. |
| Direct meta-PPO has too small a gradient batch | Current LIBERO uses one complete K4 trial/update. Prepare a multiple-trial update with explicit valid-token weighting and complete recovery, then qualify it before scale-up. |
| Steering cannot improve the difficult target cases at the proposed budget | Ordinary target-online authors' PixelSAC on six predeclared cases, with frozen query milestones and actual interaction costs. The short run tests learning onset, not DSRL convergence. |

The ordinary PixelSAC actor consumes one64x64 image plus8 proprioceptive
values. The history policies consume4104D frozen-prefix features. Recording
those features beside SAC data does not make the SAC actor representation
matched. Keep this difference explicit in architecture claims.

Corpus diagnosis from all40 canonical training receipt sets:

- **2,014 successes / 94 failures** among2,108 episodes:4.46% failures.
- **454/527 K4 cases are SSSS**:86.15% never show a failure.
- Warmup:1,332 episodes,76 failures; learned SAC behavior:776 episodes,
  18 failures. These are behavior stages, not expert-quality labels.
- 60 cases have any failure followed by later success:44 warmup-only,
  eight learned-only, eight spanning the stage transition. Only10 contain
  a learned-behavior failure followed by learned-behavior success;27 start
  failed and succeed later.
- Nominal state hashes exist for527 cases. All2,108 episode receipts lack
  measured post-settle simulator-state hashes, so the number of distinct
  realized physical states is **unknown**; seeds do not supply that count.

Hypothesis: this corpus teaches mostly already-successful behavior and offers
little evidence for learning recovery from feedback. This is a concrete reason
to reconsider the collection distribution, not proof of the cause of OOD
failure. Do not blindly multiply this recipe to1M actions. Finish the teacher,
interface and PPO/DSRL baseline tests; then choose and predeclare the next
source-history distribution while retaining all40 goals and the target split.

## PPO source validation: successful execution, little adaptation headroom

- Final full/reset source policies: 238/237 PPO updates, 476/474 optimizer
  steps, 171,166/171,707 charged source actions.
- Frozen validation: eight fixed source goals, states40/41, K4; **32 trials /
  128 episodes**, 22,753 actions, 1,203 conservatively charged GPU-seconds.
- Full policy **63/64** episodes; separately trained episode-reset **61/64**.
  Each is16 fixed cases with four correlated attempts, not64 independent tasks.
- By attempt: full **16,16,15,16 /16**; reset **16,16,14,15 /16**.
  Both start at16/16. This panel cannot establish useful adaptation or failed
  attempt recovery. Target transfer remains a separate required test.
- Exit0, unchanged weights, zero deployment updates; all32 trial trace hashes
  and final recovery checksum independently verified.
- Immutable Baige run: `libero40-ppo-source-validation-20261030-r1`,
  source `9002a2dd352e6176c43943c1e46bf2b85eddc397`.

Decision: finish the source-teacher diagnosis and frozen all90 PPO comparison
inside the approved compute. Do not interpret high source success as target
generalization or automatically expand the corpus because the first students
underperformed.

Source-teacher diagnostic R1 failed before any episode: its Python3.10 GPU
probe inherited the model's Python3.11 Torch overlay and could not load the
compiled extension. This is an interpreter/import-path mismatch, not a policy
result. Preserve R1; charge78 seconds and zero simulator actions. Retry with
the probe's own qualified import path and only the remaining original hour,
binding the failed setup receipt so its cost cannot disappear.

## RoboTwin: systems failure is separate from a method result

- Owner reports roughly1.13M committed source actions across three learners
  before native simulator reset/GIL crashes around rounds38–39.
- All six recovery files, replay/optimizers/RNG and unfinished episode prefixes
  passed the owner's integrity audit. All three collectors have resumed and
  passed a complete-round process restart; the owner reports first student
  fitting and comparison underway. The crash is not evidence against PPO or
  meta-learning. These are owner-reported, separate benchmark costs.
- LIBERO32D tied H10 steering and RoboTwin full1024D H32 steering are different
  control problems. Compare tied versus full noise before treating cross-bench
  differences as a dataset-size effect.

The later completed source100 baselines supersede the early crash/progress
counts above. FULL/RESET source success is416/2400 and421/2400; SAC is
277/2400. All three aggregate source curves lack sustained improvement.
SAC has638,500 joint updates over32,425 distinct source decisions, so its
optimizer count is not a new-sample count. Unlike PPO's shared LR1e-6/norm1,
SAC usesLR3e-4 without a norm cap; a PPO-only optimizer explanation is
insufficient. Shared interfaces and separate algorithmic problems remain
possible. Some SAC goals improve while others decline; overall flatness is
not proof that every goal behaves identically. The sampled diagnostics are
finite but include an early soft-value transient. Smaller later critic error
does not establish better robot success; alpha is an entropy coefficient,
not action std. No causal recipe remedy is established. The original fixed100
validation/targets are now complete, as recorded below. No extra collection
budget follows from these observations alone.
[Final source100 comparison](../outputs/robotwin-sac-source-diagnosis-20261031-r3/ANALYSIS.md).

A separate attempt-position analysis checks whether the aggregate curves
hide within-trial improvement. It reconciles600completeK4groups/2400episodes
for each learner. FULL attempt4minus1 is+1.17percentage points, versus
trainedRESET+1.50 andSAC−0.17; neither all100 nor first/last25 rounds shows
a strong monotonic FULL-specific gain. All four scheduled scene seeds differ
within each group; PPO weights stay fixed through the rollout batch, whereas
SAC also updates online. Thus these marginals cannot isolate memory use,
same-case recovery or support/query adaptation. They provide no evidence
that a strong inner learning pattern was hidden by the outer averages.
Saved learning histories and DPT labels should still not be called expert
demonstrations merely because the exporter passes its integrity checks.
[Attempt-position evidence](../outputs/robotwin-k4-attempt-analysis-20261031-r1/ANALYSIS.md).

Final SAC evaluation exits0 with sourceVAL2/24. The frozen reactive policy
scores8/16 on the same cases as FULL8/16 and trainedRESET8/16. The online
source-initialized version scores5/16 and performs3320 extra target SAC
cycles over5120actions in these final16episodes. It uses fresh target replay
from its first row, without a second warmup; this is a declared fine-tuning
control, not the published task-from-scratch DSRL schedule. Its lower observed
score is descriptive, not a statistically established harm from online RL.
All224 evaluation episodes across10/25/50/100 are complete; the source actor
stays fixed for frozen queries. Canonical all-phase SAC cost is1,088,029
actions, with pending/uncertain0. Its continuation allocation is32,230GPU-s,
counted once and separately from earlier source prefixes. Neither source
success nor these small target panels establish the intended high performance
or sample-efficiency benefit. The existing next phase is the source100
AD/DPT fitting/readout proposal; it remains unfunded and its source100 bank
bindings/teacher-label conditions must stay explicit.
[Independent SAC endpoint](../outputs/robotwin-sac100-independent-20261031-r1/RESULT.md),
[scheduler receipt](../outputs/robotwin-sac100-scheduler-20261031-r1/receipt.json).

The historical native6/16 reference also matches the SAC cases through the
verified native→FULL/RESET→SAC case chain and the motor/capture contract.
FrozenSAC has3wins/1loss against it; online has0wins/1loss. Distinct
executors, execution dates and stochastic draws remain explicit. These are
descriptive gains on16 development scenes, not a robust benchmark result.
Also, the frozen target total rises3→4→3→8 across source10/25/50/100:
flat source averages do not mean every target result stays flat. The late
reactive steering result warrants retaining a strong non-memory control;
it does not establish in-context adaptation.
[Historical native/SAC reconciliation](../outputs/robotwin-sac-native16-reconciliation-20261031-r1/RESULT.md).

Next execution correction: **AD does not need an expert mean teacher.** Its
labels are actual executed noise; AD's validation uses behavior cloning on
held-out source episodes. Static review finds that the existing offline plan
globally gates all exports/fits on mean-teacher qualification and hardcodes
paired AD+DPT work; the exporter also computes teacher labels for AD-only.
The native trainer already supports AD full/static. Request the existing
implementation owner to make that route objective-aware, leaving DPT's label
review conditional on requesting DPT. Keep source/behavior/checkpoint hashes,
case separation and resource funding checks. This removes optional teacher
work from the first AD fit; it neither proves improving histories nor grants
new compute. Final source100 materialization and training funding remain.

The native student evaluator already contains the real robot loop, action
accounting, raw capture, fixed-parameter checks and a reset lesion that clears
cache plus previous noise/reward. Its old orchestration still pins SAC5/
candidate100, all four models and K8, and labels four support/four query
episodes. The source100 AD500 profile and allocated entrypoint are required
before claiming the first new fit-to-robot comparison is executable. Reuse
those existing components; prepare proper one-support-plus-separate-query
cases and independent paired query randomness alongside the AD fit path.
This initial development readout does not replace the larger0/1/2/4/8 or
full-suite protocol. No new controller or benchmark adapter is required.

For the source100 AD500 phase, predeclare sampled deployment of the learned
Gaussian (then the existing bound1.5 clip). The native model trains Gaussian
NLL and learns log_std, whereas the legacy robot caller selected its mean.
Keep legacy mean-mode results separate; add no new inference-mode condition
inside the fixed phase. This choice precedes new target outcomes and uses
the existing private sampling generator with independent paired query seeds.

Allocated software checks now pass AD108 and57 selected source/teacher
compatibility cases, plus actual source100 actor/RNG/counter comparison.
Both wrapper jobs still exit1: an absolute pytest deselection mismatch,
then a run-ID/helper-name collision during final JSON publication after
successful phases and cleanup. The fixes are isolated; neither failure is
a robot or learned-policy result. Keep the failed statuses and component
passes separate. Cost139CPU-worker-s/1112core-s/0GPU; unused461s is sealed.
The combined AD-first request remains pending. Full bank materialization,
two fits and sampled readout are not yet
executed. [Fault analysis and closure](../outputs/robotwin-source100-ad-first-20261031-r1/SOFTWARE-OUTCOME.md).

The distinct sampled-readout implementation f7594a0 subsequently passes all48
allocated CPU regressions, with valid terminal summary and clean teardown;
its job is Success/exit0. This covers sampling/private RNG, support trace and
clock isolation, artifact selection and costs/resume. It adds31CPU-s, so
aggregate software cost is170CPU-worker-s/1360core/0GPU, counted once inside
the pending7200CPU-s request.7030 remains only if approved. Fixtures are not
actual500 model payloads or robot outcomes; production bank, fits and physical
readout still await the phase decision. No further routine test retry is
needed. [Readout qualification](../outputs/robotwin-ad500-software-qual-20261031-r1/RESULTS.md).

## Literature guides targeted choices

- [VLRLA's simulation history teacher](https://arxiv.org/html/2610.19517v1)
  updates the action expert; it is not our frozen-motor noise policy. Its
  parallel rollout and optimization batches are much larger. The exact realized
  simulation corpus and student update budget are not fully specified, so
  nominal slot counts are not an observed dataset size.
- [Released DSRL LIBERO recipe](https://github.com/nakamotoo/dsrl_pi0/blob/7f48937d4553e95244cd81c79236a3256df80597/examples/scripts/run_libero.sh):
  batch256, UTD20, warmup500, noise bound1, execution interval20. Our
  pi0.5 H10/execute10 interface is a declared departure. Query decisions,
  primitive actions, episodes and optimizer updates must remain separate.
- Around1M source actions per learner is the next requested scale, not proof
  of sufficiency. Report actual task coverage, failures, learning-history
  length and updates; never replace missing baseline results with scale claims.

## Completed diagnostic readout and next decision

- Noise controls: native29/90,zero25/90,tiedGaussian28/90,clipped-tied29/90.
  All360 raw/query/noise/physical/RNG/recovery checks passed. Native beats
  zero on four cases and loses none in this run. Equal aggregate clipped
  versus native does not establish equality on individual cases or explain
  policy-specific effects. Cost122993actions/4411GPU-seconds.
- Keep native scores separate: first student wave31/90; noise diagnostic29/90;
  PPO wave30/90. Across the first two, all90 nominal/reset/observation/seed
  metadata match. Four outcome flips have bitwise equal initial inputs and
  first Gaussian noise, but first decoded actions differ by.00195–.00354.
  This is cross-process decoding sensitivity; a specific kernel cause is not
  identified. Use each wave's contemporaneous control; do not pool them.
- Frozen PPO target wave: native30/90,full24/90,trained-episode-reset26/90.
  Fixed9 K1 queries: full5/9,same-weight episode reset5/9,trained reset5/9.
  All297 group hashes verified;324episodes;108894actions;4526GPU-seconds.
  Both motor hashes unchanged and strict paired-initialization flags true.
  No reliable memory or performance improvement is established.
- Source SAC competence: native/mean/sample each16/16 on the predeclared
  source panel. All48 trace hashes and final recovery verified.7540actions;
 532successful GPU-seconds plus78failed-setup seconds. This panel cannot
  establish a useful source-teacher advantage; it is not an all40 score.
- Six-case ordinary PixelSAC:4/6 at EVERY measured budget0/1/4/8/13/16.
  Tasks0/40 remain unsuccessful despite3200updates each; task30 has800;
 10/20/50 never clear warmup.35558physicalactions/1851GPU-seconds.
- Saved task0/40 trajectories:0/16 TRAIN successes each,constant failure
  rewards,13warmup episodes and only3 episodes using updated SAC behavior.
  Observed failures involve wrong objects/sequences, not just timing:
  bowl placement instead of drawer closing, dish manipulation instead of
  moving the pan. Images do not identify the causal representation defect.
- AD/DPT already use Gaussian NLL and sampled deployment. The earlier
  deterministic-MSE explanation is ruled out. At500updates global sigma is
  about.585 for AD and.421 for DPT (initial.5); label sigma is.879 for AD
  and.144 for DPT teacher means. Fitting/dispersion calibration is a testable
  concern, not proof of the transfer deficit's cause.

Decision before further scale: continue the two failed DSRL development cases
losslessly toward40/80/160/240training episodes, with100k NEW actions and
6000 cumulative GPU-seconds per worker. Parent35558actions and1851seconds
are counted once, copied prefixes are not new data. Longer runs may still
end before240; none is declared converged in advance. This tests discovery
of successful behavior before committing to more of the same meta corpus.
Prepare a bounded same-weight mean/dispersion diagnostic; source batch14 code
is prepared but remains unqualified/unlaunched. Redesign the source-history
distribution only after these diagnostics; retain all40 source goals and the
full90/Plus/Pro target protocol.

At23:54:45 Shanghai the two continuations passed13 allocated tests each, real
CUDA probes and exact full learner/replay/optimizer/RNG restoration, then
advanced beyond the inherited16 episodes/3,200 updates. Task0 reached41
training episodes/23,180 updates and its first training success at episode41;
the separate frozen query40 still failed. Task40 reached45/26,400 with zero
training successes and failed query40. This is evidence of eventual discovery
on one development case, not a successful independent query or reliable gain.
Both remain active inside their original6,000-second continuation budgets.

Detailed evidence: [trajectory/dispersion analysis](../outputs/libero40-failure-analysis-20261030-r1/ANALYSIS.md),
[native cross-run receipt](../outputs/libero40-native-crossrun-20261030-r1/README.md),
[scale references](../outputs/steering-scale-reference-20261030-r1/COMPARISON.md),
[long DSRL protocol](../outputs/libero40-target-online-dsrl-long-plan-20261030-r1/README.md).

## 2026-10-31: first longer-DSRL frozen query success

Task0 succeeds on its separate frozen query after80 training episodes:
249 primitives+10 settling, after32,651 training actions and54,140 optimizer
updates. The trace and milestone recovery hashes were independently checked;
weights and replay remained unchanged during the query, with zero query SGD.
This is one outcome-selected development case/seed, not one-shot adaptation,
convergence, repeatability or broad target performance. First training success
was episode41; query40 failed. At00:15 the run had93 training episodes,
7 training successes and63,300 updates. Task40 had100 episodes/70,400 updates,
zero training successes and failed query80. Both remain active.

Interpretation: bounded noise steering can produce successful behavior on at
least one tested failure after substantial online interaction. The short
16-episode negative onset was insufficient to reject that possibility. Task40's
finite, small critic loss with exclusively failed data does not prove useful
learning or an adequate exploration budget. Continue the declared160/240
queries before deciding whether its limitation is exploration, representation,
steering capacity or training duration. Do not enlarge its budget mid-run.

The fixed-nine mean/std diagnostic is submitted asjob-45fnd9k9p9s5 at source
1a65226e0d3b6a68cf59808028245b3da5be3d07, initially queued.63K0 queries,
27k actions and3,600 cumulative seconds are funded from prior terminal
surplus;2,540 seconds remain unassigned inside the original32h. Allocated
regressions and actual loaded-policy action/cache/RNG parity must pass before
any comparative result. This is a distribution intervention, not a new trained
meta-learning method.

PPO follow-up hypothesis: the first direct PPO used one K4 trial per gradient
batch. A14-complete-trial batch tests whether this update noise contributes to
negative transfer. The implementation and a bounded one-batch20k-action/40min
qualification launcher are in preparation; no qualification reservation or
full-scale launch yet. Its purpose is to measure replay correctness, finite
updates, memory and real collection/update costs before a larger proposal.

The diagnostic received its allocation00:16:41.61 Linux checks and the actual
CUDA probe passed. Both loaded source AD/DPT policies reproduced original
sampled actions, recurrent caches and generator state bitwise for four steps;
same-noise motor delta was0.0, with no extra simulator resets. At00:20 nine
queries/3,108 actions were committed; final outcomes and parameter/recovery
verification remain pending.

Source teacher length analysis on the16 successful paired cases: native mean
147.75 primitives, actor mean146.375, sampled actor147.125. The latter shorten
means by only0.93%/0.42%, with per-goal reversals. Source PPO shows no monotonic
execution shortening across attempts. These are descriptive execution lengths,
not training or adaptation sample-efficiency gains; native wall time includes
qualification/JIT and cannot support policy-speed claims. See
[source length receipt](../outputs/libero40-source-lengths-20261031-r1/README.md).

## 2026-10-31 terminal readout and revised next test

Both ordinary DSRL continuations completed240 training episodes, exit0. Task0
had72training successes and153,480updates; task40 hadzero/182,400. Both final
240 queries failed. Task0's only successful measured query was at80, followed
by failures at160 and240 on the same recorded query seed/initial hashes.
This does not estimate a success-rate decline: each changed checkpoint has
only one fixed stochastic query, and training successes mix many policies.
The next diagnostic therefore freezes80/160/240 plus native and uses eight
new paired stochastic seeds on the same physical case. That comparison is
outcome-selected development; it neither selects a best source checkpoint nor
supports broad generalization. No more online interaction is added to these
closed240-episode runs. See
[terminal DSRL receipt](../outputs/libero40-target-online-dsrl-long-audit-20261031-r1/README.md).

The mean/std diagnostic completed63queries/14,916actions/728GPU-s: allseven
conditions score6/9 with identical case outcomes (fail0/40/80). The actual
interventions changed executed noise std fromAD.548/DPT.427 toabout.719;
unit-std clipped about32% of coordinates. Thus this is not a no-op, but this
run shows no success sensitivity on the selectednine. The panel omitted20of
23 previously observed sensitive IDs; it does not rule out dispersion effects
on the broader suite. Preserve separate contemporaneous native controls.
[Full report](../outputs/libero40-distribution-report-20261031-r1/README.md),
[sensitivity map](../outputs/libero40-sensitive-cases-20261031-r1/README.md).

Batch14 PPO qualification completed one actual update:14K4trials/56episodes,
795decisions,8,313all-in actions andtwoAdamsteps. Exact individual/merged
likelihood andvalue replay errors areall0. Policy/candidate match, full
optimizer/sampler/global+motorRNG recovery and unchanged frozen-motor hashes
passed. All56episodes succeeded BEFOREthisupdate, so it supplies no recovery
signal or target-performance evidence.795real decisions is still much smaller
thanVLRLA's nominal8,192-decision iteration; trials are not interchangeable
withquery decisions.

Measured852GPU-s:651.24collection+4.09replay+13.46update seconds, with
10,804MiB sampled devicepeak and1,345,175,040-byte Torchupdatepeak. A single
simulator collected trialsserially;14 is thegradientbatch, not14parallel lanes.
The initial global update had finite metrics butKL.0645/clipfraction.348;
oneupdate cannot establish optimization stability. Steadycost extrapolates
about22.4GPUh/Mactions, or28.5including this coldstart, before workloadvariation.
[Qualification](../outputs/libero40-batch14-qualification-audit-20261031-r1/README.md).

Decision: preserve allclosedruns; use the pairedfrozen-SAC repeats to measure
checkpoint variability. Prepare explicit, lossless sourceSAC andR3→batch14PPO
continuation/admission paths before requesting materially larger compute.
The proposed1Msource/learner scale adds postwarmup exposure (oldSAConly272.9
learned querychunks/goal onaverage), but does not guarantee useful recovery
histories or targetgains. New source validation/qualification costs stay inside
thatcap. An optional cached-prefix B1 optimization remains unqualified and
must not delay the original-runtime collection proposal. Nofullbank top-up
has been launched or newlyapproved.

Original32h accounting after terminal UI/clock closure:109,311seconds held
including conservative shutdown margins,5,889seconds unassigned. This is
remaining oldapproval, not authorization for the proposed24hSAC or122hfull
envelope. The per-learner scale proposal remains a staged resource request,
with frozen rollout/update counts, actual coverage and later full target
matrix still required.

## 2026-10-31 collection cadence correction

The user asked why we did not collect both benchmarks in parallel and then
train. Initial LIBERO collection and training did occur, but the subsequent
loop became too serial: multiple diagnostic branches delayed source scale-up.
Prioritize the source-SAC continuation and PPO migration needed for the next
collection phase; defer optional repeated-query and cache diagnostics until
they do not delay collection. RoboTwin stays under its independent owner's
running collection/training loop. Keep the original32h ceiling: its5,889s
remaining can fund bounded continuation checks, while the proposed24h source
SAC top-up still needs explicit approval. After those checks, overlap collection,
offline fitting from complete snapshots and target evaluation rather than
waiting for an entire corpus. Fresh PPO collection remains on-policy; SAC
histories are not interchangeable PPO rollouts.

## 2026-10-31 continuation evidence and preflight repairs

Warm PPO completed a real native batch14 update after lossless R3 migration:
14 K4/56 episodes/894 decisions,55 successes and one SFSS pattern, two new Adam
steps. Exact replay is zero; all old policy/optimizer/global+motor RNG leaves
were preserved, actual Adam step478, candidate/full recovery equal, motor
parameters unchanged. Cost9,272 actions and910 charged GPU-seconds. One SFSS
under a stochastic fixed behavior, without a matched reset control, is not
evidence of memory or adaptation. This qualification is runtime evidence.

SAC continuation preflight exposed two implementation faults: missing GPU
visibility fallback, then a missing selected.slot field in a test fixture.
The first attempt used76s; the second passed GPU qualification and used70s.
Both made zero simulator actions and never created a source experiment. The
original data and full recoveries are unaffected. The new source cf49bfa fixes
the launcher/fixture and permits a smaller worker share under the unchanged
shared1,000s guard. A fresh r2 output receives854s, with the old146s retained;
the invocation slice700+60 leaves94s headroom. No scientific collector, case,
SAC optimizer or behavior contract was changed. These preflight failures say
nothing about method efficacy. See the preserved
[launch/repair record](../outputs/libero40-sac-topup-preparation-20261031-r1/LAUNCH.md).

Decision: finish actual restored-SAC optimizer/export qualification, then use
the executable source collection→snapshot→student-training path. The legacy
assembler hardcodes original caps/pins/quota exhaustion; an explicit top-up
complete-prefix admission is being implemented so fitting need not wait for
the full bank. Existing source and warm PPO remain near ceiling, so more
source actions are an experiment rather than a guaranteed recovery signal.
Optional repeated-query/cache diagnostics remain deferred. Full source scale,
matched target matrix and the robotic sample-efficiency claim are unfinished.


## 2026-10-31 source continuation and overlapping fitting

Both bounded SAC bridge collectors completed with full recovery and paired
AD/DPT exports. The qualification and bridges added224 source episodes;
all224 succeeded. This extends learned-policy histories without supplying new
failure-recovery evidence. It does not reverse the earlier negative target
result or establish a teacher improvement.

Current valid training data: all40 streams,583 complete K4 cases,2,332 episodes,
33,729 steering decisions and327,547 executed primitives. Including settling
and stream charges, training costs351,297 actions. The disjoint fixed8
validation streams contain146 cases/584 episodes/8,202 decisions/79,381
primitives, costing85,301 all-in actions. Historical overhead21,253 is counted
once: total SAC source resource charge457,851. Resource charges are not the
number of training examples. [Canonical counts](../outputs/libero40-sac-source-bridge-audit-20261031-r1/canonical-counts.json).

The CPU capture/assembly job completed all48 immutable complete prefixes,
34 metadata tests and25 saved-result checks; zero GPU time or simulator
actions,710 conservatively charged CPU-worker seconds. A mistaken3.10 runtime
guard failed its first CPU attempt; actual qualified base is3.12.15 and the
fresh corrected CPU job passed. The OpenPI student runtime remains3.11 with
its qualified cu128 overlay. [CPU receipt](../outputs/libero40-source-pipeline-cpu-audit-20261031-r1/README.md).

Decision: use the assembled snapshot immediately for all four AD/DPT full/static
fits at25/100 updates, source2bc15da, Baige job-wbvs39zbt0pw. This uses1,800
seconds of the original32h pool, with1second unassigned; no further actions
are collected by fitting. The complete1M-source and500-update phase is still
unfinished. A concrete25additionalGPU-hour/130GB source-topup plus final-fit
phase is being prepared for approval; it is not running. Collection, frozen
snapshot fitting and robot evaluation should overlap after bounded checks.
Optional repeated-query/cache diagnostics remain deferred.


## Interim100 fit readout, 2026-10-31 03:51

The immutable new snapshot fitted all four AD/DPT full/static controllers
through100 updates, with exact25/100 policy and full optimizer/sampler/RNG
archives. Script exit0; no additional simulator actions. This is an interim
fit point, not completion of1M/500 updates or a robot-performance result.

AD full/static validation NLL at100 is51.8066769/51.8066893. Static review
found distinct feedback/memory paths: full retains prior executed noise,
reward/done and cached state; static zeros feedback and resets every query
(vla_history.py124–140,186–225; gtrxl.py140–150). Both use the same initial
parameters, training windows and fixed validation RNG734001. AD predicts
actual stochastic behavior noise, including fresh Gaussian exploration;
its innovation cannot be predicted from history. Current observations may
explain most predictable mean. Aggregate near-equality does not prove equal
policy outputs or establish a memory benefit/absence. The Gaussian head has
one global32D standard-deviation vector, so history cannot express
teacher-stage-dependent uncertainty; this is an architectural limitation,
not an observed routing bug.

DPT-full logged minibatch NLL was8.6601 at25 and6.9910 at100, but these
training batches differ, so they do not establish a training-loss trend. Validation
NLL worsened13.5081→16.0686 on identical fixed4×8 validation windows. This
is a real worsening on those windows. Overfitting, source-task mismatch and
variance calibration are possible explanations; one seed and these windows
do not identify the cause. The predetermined100 checkpoint remains fixed;
no new source selection or target-dependent tuning is introduced.

Decision: robot evaluation of the fixed candidates is the discriminating
performance test. Keep the separate25h source-scale request explicit and
unapproved until the human replies; more source data remains an experiment,
not an established repair. Optional diagnostics remain deferred.


Final interim audit:26 saved-result checks passed, including independently
rehashing all16 policy/recovery files. The worker charged571GPU-seconds;
terminal scheduler duration531s and50s additional shutdown margin close621s
of the1800s hold, releasing1179s. Original32h now has1180s unassigned.
All controller exports retained source resource charge457,851 exactly, with
no extra actions. [Terminal audit](../outputs/libero40-source-topup-interim100-20261031-r1-audit/AUDIT.md).


## Ready source assignments and target profile, 2026-10-31

The production source plan is now prepared on a real0GPU/4CPU/16GiB job,
exit0: all40 goals admitted,38 unique assignments across8 workers, two exact
quota-filled bridge leaves retained and fixed8 validation streams imported.
Inherited source resource457,851 and cumulative cap998,554 are verified.
The source ledger is byte-for-byte unchanged and the bound macro budget
remains UNFUNDED. CPU107s; no tests, model, simulator or GPU work. The pending
25h approval request is unchanged; source GPU collection has not resumed.
[Plan receipt](../outputs/libero40-source-production-plan-cpu-audit-20261031-r1/RESULT.md).

The additive fixed100/LIBERO90 readout passed48 actual allocated CPU tests,
with zero skips, including the real100 matrix, CPUmanifest/assembly and all
8 policy/full-recovery hashes. No target model/simulator ran. This makes the
522-episode/220k-action/4.5GPUh readout executable once separately funded;
it does not grant that budget or complete full Plus/Pro or2/4/8-shot tests.
[CPU receipt](../outputs/libero40-interim100-target-cpu-audit-20261031-r1/README.md).

A proposed explanation for the old robot deficit was deterministic-mean
inference after Gaussian-likelihood fitting. Static inspection falsified it:
the historical and current AD/DPT deployment chain already uses stochastic
mean+std×randn by default. Deterministic=True is used to obtain likelihood
parameters inside training; rollout and recording wrappers retain the
stochastic default. Thus no new mean-versus-sample diagnostic was launched,
and1,180 old GPU-seconds remain unassigned. This does not establish good
variance calibration or explain DPT's matched-window validation regression.
[Source-line evidence](../outputs/libero40-stochastic-deployment-static-20261031-r1/README.md).


## PPO continuation check and validation interpretation

The new source-PPO continuation glue passed7 allocated CPU tests with zero
skips and19 terminal checks. Its actual saved FULL checkpoint migration
preserves every non-contract leaf, cursor14, Adam478, pending state, typed
motor RNG and the9,272-action copied journal prefix; original checkpoint
bytes are unchanged. Reset reuses the existing generic episode-mode R3
warmstart. A first CPU attempt failed because its mock supplied object()
instead of the required RL2Config dataclass; the corrected fresh r2 passed.
R1/r2 used82/80 conservative CPU-worker seconds and zero GPU time/actions.
This is new glue/recovery evidence, not a new PPO update or robot result.
Each44h production phase remains proposed/unfunded, with30k actions and1h
held for final source validation inside its1M/44h envelope.
[Result](../outputs/libero40-ppo-production-cpu-20261031-r2-audit/RESULT.md).

DPT validation is cross-teacher plus heldout-state imitation. All8 matching
TRAIN/VAL goals use independently trained SAC teachers with different seeds,
published learner checkpoints and optimization clocks. The same fixed
validation windows/labels are reused at25/100; their change does not explain
the NLL rise. But teacher identity/stage mismatch is an additional possible
source of error, so the rise cannot be attributed solely to overfitting or
worse robot control. The stored hashes identify full learner checkpoints,
not pure actor parameters; actual actor-function disagreement on common
observations was not measured. Labels, fixed milestones and selection stay
unchanged; robot readout remains the performance test.
[Eight-pair provenance](../outputs/libero40-dpt-validation-teacher-provenance-20261031-r1/ANALYSIS.md).

Full shifted-target preparation is also explicit: Plus has10,030 registered
variants, one upstream trial each; a five-model K0 pass is50,150 episodes.
The complete pinned public Pro file set is now staged and hash-verified on
Baige (676files/10.27MB), including160 core BDDL/init pairs. Forty registered
environment tasks still require official generation/staging; no full-suite
readiness or score is claimed. Neither staging nor inventory enters source
pretraining or gates source collection. Full-suite GPU costs remain separate.
[Inventory](../outputs/libero-shifted-suite-inventory-20261031-r1/README.md),
[Pro stage](../outputs/libero-pro-full-stage-20261031-r1/README.md).


Next-generation validation decision: prefer the same source-only, preselected
TRAIN actor per goal for TRAIN and heldout-state VAL DPT labels on the eight
matched goals. Preserve actual behavior histories, noises/actions/rewards and
all legacy archives. The proposed derived-label schema separates behavior
and label-teacher provenance, pure actor digest and full checkpoint, with
source costs counted once. Until verified, explicitly retain teacher_transfer
as the metric; no silent relabel or fixed500 checkpoint-selection change.
Collection remains independent. Up to600GPU-s comes from existing source
contingency within25h (89,480 planned/520 unassigned), not a larger budget.
[Design](../outputs/libero40-dpt-teacher-binding-design-20261031-r1/DESIGN.md).

The ready robot readout has been added to the pending resource request:
29.5additionalGPUh/150GB, comprising25h source+4.5h fixed100 LIBERO90 readout.
This supersedes the unanswered source-only question; approving source only
still grants only25h/130GB. No new budget has yet been granted. PPO44h/arm,
full shifted suites and2/4/8-shot completion remain separate/unfunded.
[Combined request](../outputs/libero40-source-and-readout-budget-20261031-r1/PROPOSAL.md).

## Measured teacher disagreement and execution priority

The optional CPU writer has now produced same-TRAIN-actor targets for all8
validation goals/8,202 recorded decisions. Mean absolute differences from
the original VAL-actor targets range0.1403–0.6046; goalwise RMS differences
range0.1891–0.6966. Each actor's saved parameter digest matched its loaded
TRAIN checkpoint and remained unchanged. One TRAIN-label replay per goal
checked the CPU backend: the largest observed absolute difference was
3.28e-7 (tolerance1e-5). This sampled check does not prove exhaustive
cross-backend equality.

This measures actual label disagreement on common observations, rather than
inferring it from different full learner checkpoint hashes. It confirms
teacher mismatch is part of the old validation objective. It does not
identify the cause of the25-to100 NLL change, rank the teachers by robotic
quality, or establish a benefit from new labels. Preserve every legacy loss
and fixed checkpoint choice. The new heldout-state metric requires the
separate derived-generation admission, which passed its complete assembly
round trip with all AD/TRAIN arrays unchanged and source charge457851.
All49 focused tests passed on the actual0GPU CPU allocation; exit0,
348 conservative CPU-worker seconds, zero new simulator actions or SGD.
[Evidence](../outputs/libero40-dpt-same-train-teacher-cpu-20261031-r1-audit/RESULT.md).

The user's renewed execution priority is collection and training in parallel
on both benchmarks. Source collection and the ready target profile do not
wait for this optional analysis. The residual-noise proposal was held at a
static feasibility note with no launcher, allocation or GPU reservation.
LIBERO expansion remains a concrete pending compute decision, not another
qualification requirement. RoboTwin's owner reports89PPO/91reset/83SAC
completed source rounds and2.570M closed actions as of05:26Shanghai; its
100-stage evaluations remain unfinished and its budget is separate.

The CPU-only qualification returns the proposed600GPU-s label-preparation
reserve to contingency: source planned88,880/90,000s,1,120s contingency.
The pending29.5GPUh/150GB request and original32h ledger are unchanged.

## RoboTwin reset100: weak source learning, uneven target success

The separately trained episode-reset recurrent PPO controller completed100
balanced source rounds/600K4 trials/2,400 episodes,963,759 source actions and
500 Adam steps. The owned terminal audit passes and the motor remains frozen.
This is a completed RESET endpoint; persistent PPO and SAC final comparisons
are still pending.

| Source round | Bottle target /8 | Stapler target /8 | Source validation /24 |
| --- | ---: | ---: | ---: |
|10|6|1|4|
|25|7|0|4|
|50|7|0|2|
|100|8|0|3|

The100-round training history itself is weak and flat:421/2,400 successes
(17.54%); consecutive25-round blocks112,106,102,101 /600. Per-source-goal
successes/400 are handover76, bottles35, a2b66, object-stand138, phone86,
scan20. Final source validation is0,1,1,1,0,0 /4 respectively. Small fixed
validation panels and stochastic actions limit generalization conclusions;
there is no demonstrated strong source policy followed by a validation-only
collapse. Target novelty alone cannot explain weak source training.

The100 updates contain31,172 actual valid chunk decisions, mean311.72 per
global update; all500 optimizer steps are accounted for. Losses are finite
and gradients nonzero. Logged approximate JOINT KL averages0.000232, maximum
0.001665; median clipping fraction0. Log-probability sums all1,024 action
coordinates, so tiny KL is not per-coordinate averaging. Pre-clipping
gradient norms54.94–76.35 exceed the norm1 cap. These show conservative
updates, not that raising LR or reducing clipping would improve robots.
Actual source horizons600/500/400 match the task configs; GAE crosses earlier
episode boundaries and terminates at the final K4 boundary. This review finds
no supported reward/GAE defect or old uniform400-horizon explanation.

The already-recorded VLRLA reference has8,192 nominal chunk slots/update,
about26.3 times this learner's mean actual batch. That compares capacity with
realized data, and VLRLA trains a different component. Five epochs and more
iterations do not reproduce its optimization scale. It does not prescribe
collecting10M actions. Small batches, conservative optimization, task
interference and the bounded1024D noise parameterization remain competing
hypotheses, not established causes.

Bottle8/8 is not memory or adaptation evidence: this control resets cache
and previous noise/reward at every episode, while retaining within-episode
recurrence. Its target seeds differ from the available prior native bank,
so there is no paired contemporaneous native improvement claim. The one
extra bottle success from50 to100 does not establish a scaling benefit;
stapler remains0/8. This is eight scenes per goal in a two-goal development
panel, not eight independent heldout goals or a full benchmark. Keep all
predeclared checkpoints and finish the already
running full/SAC comparisons. Do not simply multiply this PPO recipe because
the source data count reached approximately1M. Any later optimization or
native-source diagnostic must have its own explicit execution budget; it
does not gate LIBERO collection/readout.
[Measured training diagnosis](../outputs/robotwin-reset100-diagnosis-20261031-r1/ANALYSIS.md).

## RoboTwin full/reset100 comparison

Both PPO arms have now completed their fixed100 source rounds and final
evaluation; the shared pair exits0. FULL uses964,875 source actions/31,211
valid chunk decisions; RESET963,759/31,172. Each has600K4/2,400episodes/
500Adam steps. FULL training success416/2,400 versus RESET421/2,400;
quarter totals116→104→93→103 /600 versus112→106→102→101. Both source
policies are weak and flat. FULL final source validation2/24 versus RESET3/24
is too small a panel to establish a ranking.

| Fixed100 condition | Bottle /8 | Stapler /8 | Total /16 |
| --- | ---: | ---: | ---: |
| Saved native Gaussian, matched cases |6|0|6|
| FULL normal |7|1|8|
| Same FULL weights, cache reset |6|1|7|
| Same FULL weights, reward blind |8|1|9|
| Separately trained episode reset |8|0|8|

There is no consistent demonstrated memory or reward-use advantage. The
normal/cache-reset stapler successes occur on different episodes; equal
aggregate counts do not mean identical trajectories. Cache reset removes
the recurrent cache but preserves the FULL policy's direct previous-noise/
reward input across a boundary. The trainedRESET clears both. These are
distinct controls, and neither is a fully memoryless MLP.

RESET target requested seeds and summary initial-feature hashes match across
source checkpoints; the summary field is named initial_observation_sha256
but refers to features in the inspected records. At stage100, additional
raw RGB/wrist/proprio/features/language descriptors match across all four
learned conditions. Their first FULL-policy episodes have identical executed
histories before interventions can affect feedback. Five source-validation
goal banks have matching summary hashes; handover uses the same seeds with
different hashes, so the entire validation series is not an identical-input
paired experiment. Controller RNG may
also differ across conditions/checkpoints. Prior native probes use a separate
seed bank. A separate earlier native16 panel has now been verified against
all64 learned target records, including resolved seed fallbacks and raw
input/feature descriptors. It shares motor, normalization, backend capture
qualification and action-transform contracts, while executor versions/dates
and noise samples differ. This historical case-matched reference scored6/16;
FULL has3 recorded wins/1loss against it and trainedRESET2wins/0losses.
These descriptive gains are not independent replications, statistical proof
or a broad benchmark result. Reuse adds zero actions/GPU time; the original
native panel's5051actions remain charged once to that panel.

Both learners have finite nonzero optimization but very conservative logged
joint KL (FULL mean0.000226, RESET0.000232). Source training is already weak,
so target novelty alone is not an explanation. The data do not identify the
cause: weak exploration/steering support, small effective batches, conservative
optimization and task interference remain hypotheses. Finish the existing
sourceSAC comparison before choosing the next recipe; none of this gates the
ready LIBERO source/readout phases.
[FULL diagnosis](../outputs/robotwin-ppo100-diagnosis-20261031-r1/ANALYSIS.md),
[RESET endpoint verification](../outputs/robotwin-reset100-independent-20261031-r1/RESULT.md),
[matched FULL endpoint verification](../outputs/robotwin-ppo100-independent-20261031-r1/RESULT.md).
[Historical native16 reconciliation](../outputs/robotwin-trained-continuation-20261031-r1/NATIVE16-REUSE.md)
comes from the owning chat's committed46856dc audit; the contemporaneous
single-seed probes remain distinct and unpaired.
