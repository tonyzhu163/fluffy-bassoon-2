# Frozen-VLA steering: experiment summary

Objective: establish useful PPO and DSRL baselines, then find a meta-learning
method that improves robotic success and adaptation cost. The architecture is
open. More data is a hypothesis to test, not an explanation for every failure.

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
