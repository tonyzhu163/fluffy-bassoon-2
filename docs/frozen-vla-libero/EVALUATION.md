# Frozen VLA steering evaluation

Verified robot evidence only. Method: [LIBERO interface](../frozen-vla-libero.md).
Paper: [draft](../paper/DRAFT.md), [decisions](../paper/DECISIONS.md).

## All40 students: first broad target screen complete

All six workers exited0:558 groups,626 episodes,189,051 actual simulator
steps including settling and7,957 conservatively charged GPU seconds. All
committed result hashes, unchanged motor hashes and actual initial-observation
matches passed. Source-only fixed500 checkpoints; no deployment SGD.

| K0 query | Native | AD full | AD static | DPT full | DPT static |
| --- | ---: | ---: | ---: | ---: | ---: |
| LIBERO90, one initialization per goal |31/90|26/90|26/90|26/90|27/90|
| Plus,4 development conditions |4/4|4/4|4/4|4/4|4/4|
| Pro,4 seeded development conditions |4/4|4/4|4/4|4/4|4/4|

On the predeclared nine-goal LIBERO90 K1 panel, AD and DPT each score6/9
retaining support history versus5/9 erasing it. Both had5/9 atK0; native and
AD-static already score6/9 atK0 on those same cases. The only memory discordance
is task70 for both objectives, after successful support. No failure recovery
or superiority over native is demonstrated. Plus/Pro remain ceiling panels
at280-step development cutoffs, not exhaustive scores. One training seed and
one initialization per L90 goal limit generalization. Ordinary DSRL/PPO were
not part of this wave. [Full results/costs/integrity](../../outputs/libero40-target-firstwave-plan-20261030-r1/RESULTS.md).

## All40 source collection and student fitting complete

The canonical40 training/8 validation SAC streams are terminal, exit0 and
collection-qualified:527 complete K4 training cases/2,108 episodes/326,277
simulator actions,146 validation cases/584 episodes/85,301 actions. The
411,578 selected actions exclude12,944 separately spent source-qualification
actions. State/seed repetition is explicit in the receipts. Target data did
not enter this source corpus. All four AD/DPT full/static controllers then
completed500 fitting updates; all24 milestone model/recovery checksums passed.
Corpus quantity and successful SAC updates establish neither useful
meta-learning nor strong robot performance.
[Receipts and costs](../../outputs/libero40-bulk-approved-20261030-r1/README.md).

Both shared PPO modes initially stopped at40 iterations on a repeated-scene
observation guard. Preserved fresh-process probes found exact simulator state
and proprioception with at most1uint8 camera-level variation. After a qualified,
source40-specific camera tolerance, lossless continuations completed exit0:
full238 updates/476 optimizer steps/171,166 actions; separately trained
episode-reset237/474/171,707. Both cover all40 source goals and retain complete
optimizer/sampler/RNG recoveries. Final frozen validation and target results
are recorded below; the repaired interruptions are not method results.

## Shared-motor source-scale comparison completed

Baige `job-7mznlehmgocq`, source`f5552d6`, exited0 after all28groups/76episodes:
2,976steering chunks,29,745primitives+760settling=30,505actions within32k.
52Linuxchecks passed. Both source banks, native and frozen final target-SAC
shared one actual motor process with unchanged fullNNXparameter SHA256.
An independent saved-data audit checked76traces/76completecontexts, every
recovery/journal,8studentfiles,2finalSACrecoveries, actual paired initialarrays/
queryRNG,8full-reset pairs and bitwise static K0/K1 queries.

| Fixed500 candidate | K0 query successes | K1 query successes |
| --- | ---: | ---: |
| Native; frozen final target-SAC (each) |0/2|not rerun|
| One-source AD full |0/2|1/2|
| One-source AD static/episode-reset (each) |0/2|0/2|
| Two-source AD full/episode-reset (each) |0/2|0/2|
| Two-source AD static |1/2|1/2|
| Either source bank, DPT full/static/reset (each) |0/2|0/2|

On task0, one-source AD full succeeds339primitives after failed400support,
while same-weight reset fails. Two-source AD static succeeds103primitives
with or without support; both query trajectories match exactly. Task1 fails
for every method. All24supports fail. Thus there is a source-trained steering
benefit on one case without history, and no monotonic source-scale memory
gain. No reliable/high-performance/broad adaptation claim follows. This
remeasures the original discovery case; it is not another independent case
for the six-state replication. SAC's original15episode/2,400update target
training costs5,976/6,150actions remain separate from these frozen queries.
Prefix-feature offline students versus fresh pixel-SAC do not isolate an
architecture advantage. [Results and audit](../../outputs/libero40-common-runtime-target-audit-20261030-r1/README.md).

The source-quality audit matched all3,215selected train/validation chunks and
the exact4,000sampled windows in each of8controller recoveries. Source2train
is65.2% Gaussian warmup; its two training failures and subsequent same-case
recoveries occurred before any optimizer update. All37post-update training
episodes succeeded, without a matched learned-improvement demonstration.
Finite windows expose much recent feedback but not the entire replay-driven
learning process. Final DPT actors were saved after the last update and have
no subsequent source rollout in these histories. These are candidate teachers,
not demonstrated experts. Selected source-generation cost33,419actions plus
6,406unselected qualification actions is39,825counted once. No corruption or
new admission gate was found. [Source-quality evidence](../../outputs/libero-source-bank-quality-20261030-r1/REPORT.md).

## Targeted subset training and first LIBERO90 robot comparison

The fresh target-online DSRL onset comparison is now terminal:
`job-hweficar0ilk`/source `0c89764`,59 Linux checks. Both original target cases
completed15 training episodes and2,400 SAC updates; all eight frozen queries
after0/1/13/15 training episodes failed. Total15,426 actions includes all30
training/eight query episodes and qualification settling. Source teachers
were not restored; queries never entered replay. All38 trace hashes, eight
query recoveries and both final learner/replay/RNG recoveries were checked.
This is a short learning-onset test, not converged DSRL. Identical initial
observations and noise nevertheless produce first-action differences up to
0.00415 versus earlier native controls; the exact cross-run pairing remains
unresolved. The completed zero-action replay below rules out key type and
the tested within-process initialization stages; it does not identify the
cross-process cause.
[DSRL results and limitations](../../outputs/libero40-target-online-dsrl-audit-20261030-r1/README.md).

Baige `job-xla7geuytgau` completed four AD/DPT full/static controllers with
500 optimizer updates each, preserving fixed25/100/500 exports and full
recoveries. Source `1a41503`;71 Linux tests passed. The training reused742
chunks from one source goal (Spatial00,80 episodes/20 nominal states) and617
independent same-goal validation chunks. It generated no simulator actions.
The2,000 unique optimizer updates took1863.26s; every milestone policy and
recovery hash and Adam step was independently checked. AD's own validation
loss worsened while its training loss fell. DPT's own loss improved; their
raw losses have different target entropies and cannot rank robot performance.
[Training receipts](../../outputs/libero40-subset-scale-audit-20261030-r1/README.md).

The robot comparison `job-mjadtzoxnmy6`, source `cbd7064`, completed28 primary
method/case groups,84 episodes,34,369 physical actions within35,046. It used
LIBERO90 tasks0/1, nominal state0, scene seed123, declared before outcomes.
Every group had an empty-context query and an independent trial with one
actual support followed by a separate query; deployment weights stayed fixed.
250 Linux tests passed/two probe-environment JAX skips. All84 raw trace hashes
and their primitive costs were independently checked.

| Method | Zero-support query successes | One-support query successes |
| --- | ---: | ---: |
| Native, tied Gaussian, zero noise, frozen source pixel-SAC (each) |0/2|0/2|
| AD full,25 or100 updates (each) |0/2|0/2|
| AD full,500 updates |0/2|1/2|
| AD trained static,25/100/500 updates (each) |0/2|0/2|
| AD500 same weights, episode-reset |0/2|0/2|
| DPT500 full/static/episode-reset (each) |0/2|0/2|

The single successful query closed task0's top drawer in329 primitives after
a failed400-primitive support. Same-weight episode-reset failed. Their actual
support trajectories, empty-history queries, query RNG and nominal-state
hashes matched exactly; all initial RGB/wrist/proprio arrays also matched.
The first query noise changed by max0.402480781 with retained context. This
is a controlled one-case history effect, not reliable adaptation, scalar-reward
causality, high topline performance or full-suite transfer. The second case
failed throughout. Most of this84-episode pilot failed; more source diversity
and independent replications are needed. The frozen source pixel-SAC is not
ordinary target-online DSRL, and it uses a different observation representation
from prefix-feature students. That baseline comparison remains unfinished.
[Robot results and paired-context audit](../../outputs/libero40-subset-target-audit-20261030-r1/README.md).

The subsequent offline input-sensitivity probe `job-7hpkjw790zxm`/`3ab93f9`
exited0 (six Linux checks), using zero new simulator actions. Genuine actual
support/first-query proposals replayed with error0. With identical recorded
observations/actions and paired Gaussian draws, erasing support changed the
first-query mean by L2=1.28944; masking rewards to0 changed it by0.06592;
flipping terminal failure reward−1→0 changed it by0.06569. The controller is
sensitive to these feedback inputs. No counterfactual physical success was
measured, so this does not show a causal reward benefit.
[Probe receipts](../../outputs/libero40-context-probe-audit-20261030-r1/README.md).

Independent physical replication `job-8wtz8wd2dhbz`/source `3ab93f9` completed
all six predeclared task0 states1–6: retained context1/6, reset context0/6.
Every shared support failed. State6's retained-context query succeeded in111
primitives; its reset query failed at400. All18 raw traces/context hashes,
matching query RNG and initial public arrays, six distinct nominal-state
hashes, full recovery and7,091 actual actions were independently checked.
45 Linux tests passed. This is one repeat on the same goal, with low overall
success, not a reliable or broad adaptation result. The original discovery
case stays outside the replication denominator.
[Replication receipts and per-method costs](../../outputs/libero40-memory-replication-audit-20261030-r1/README.md).

A second source slot, LIBERO10 task0 `job-5ghfldksvvgn`/source `3944ff7`, also
completed:28 episodes/27 successes,7 nominal states,809 chunks,6,620 SAC
updates,7,985 primitives+290 settling=8,275 actions. Full recovery and paired
exports were independently hashed. This is mixed warmup/online collection
behavior, not evidence that its final actor beats native or is an expert.
[Source receipts](../../outputs/libero40-source-l00-audit-20261030-r1/README.md).

Its independent validation stream `job-aehuoen0j6a5`/`261bd43` completed
36 episodes/32 successes,9 nominal states40–48,1,047 chunks and11,400 SAC
updates.10,344 primitives+370 settling=10,714 actions within12k.263 Linux
checks passed/two JAX probe skips; raw/prefix/checkpoint/recovery hashes were
independently checked. Together with the existing Spatial00 data, the next
two-goal subset contains1,551 training and1,664 validation chunks on disjoint
physical states. This supplies the second source-data scale, not full40.
[Validation receipts](../../outputs/libero40-source-l00-validation-audit-20261030-r1/README.md).

Two-source-goal training `job-ctqxbzx4bhhv`/`0c89764` is terminal exit0:
four AD/DPT full/static controllers each completed500 updates, with all12
25/100/500 policy/full-recovery pairs independently checked.79 Linux checks
passed;2,000 unique optimizer steps took1858.58s, with zero new simulator
actions. The1,551/1,664 train/validation chunks have shared source costs
16,167/17,252 actions, counted once across controllers. This is two of40
source goals. Different validation banks prevent a cross-run NLL claim about
data scaling. [Training audit](../../outputs/libero40-two-goal-training-audit-20261030-r1/README.md).

Motor replay `job-ptt7fgi29ebx`/`b517a91` is terminal exit0, five Linux checks:
64 predictions,121.03s, no simulator actions/SGD. Typed/legacy keys, repeated
identical inputs, saved input origins and plain/prefix-attached/qualified/SAC
initialization stages all gave delta0 within one process. Full motor
parameters stayed unchanged. The new process still differs from both saved
reference chunks by up to0.00342, so the cross-process cause remains unproven.
HLO fingerprint extraction was unavailable. The completed comparison above
uses one motor process for both source banks, native and frozen final target-SAC.
[Numerical replay audit](../../outputs/libero40-motor-replay-audit-20261030-r1/README.md),
[common-runtime contract](COMMON-RUNTIME-SOURCE-COMPARISON.md).

Full40 source coverage, converged ordinary DSRL comparison, reliable memory
gains, full90/Plus/Pro evaluation and the RoboTwin learning comparison remain open.

## Capped source collection and fresh PPO qualification

The first physical-budget SAC stream and both fresh PPO qualification jobs
completed on Baige with the same frozen official OpenPI40 motor. All physical
costs below include qualification/reset settling and count inside the shared
1M-action source campaign, once.

| Run / job | Actual exposure | Episodes / successes | Query chunks | Updates | Total physical actions |
| --- | --- | ---: | ---: | ---: | ---: |
| Capped SAC `job-p5we8pk45d8i` | Spatial00,20 distinct nominal states | 80 /79 | 742 | 4,920 SAC | 7,892 |
| Single-task PPO `job-rg68oj4uy6mv` | Spatial00 | 32 /32 | 289 | 8 PPO /32 optimizer | 3,107 |
| Shared PPO `job-b7u5ogwy6udu` | Spatial00/01/02, one shared learner | 12 /12 | 125 | 3 PPO /6 optimizer | 1,303 |

SAC respected its9,500-action cap and completed every started K4 case, with
paired final AD/DPT exports and full recovery. The shared PPO job retained
one policy/optimizer across actual task switches and replayed every behavior
likelihood with error0. It recorded task/instruction, raw public current/next
observations, actual noise/actions/rewards/boundaries and PPO-specific samples,
values/cache/policy versions, plus full learner/motor RNG recovery.

The single-task PPO's3e-4/four-epoch recipe produced first-update KL13.76.
A separately pinned shared pilot used1e-4/two epochs: KL.132/.026/.037.
This changes the explicit pilot recipe; it does not demonstrate stable or
sample-efficient learning. All source successes are near ceiling. No target
benefit or improvement over ordinary DSRL follows from these runtime checks.
The later robot comparison is above; full source coverage remains unfinished.

Compact receipts: [capped SAC](../../outputs/libero40-1m-sac-qualification-audit-20261029-r1/README.md),
[single-task PPO](../../outputs/libero40-1m-ppo-qualification-audit-20261029-r1/README.md),
[shared PPO](../../outputs/libero40-1m-shared-ppo-qualification-audit-20261029-r1/README.md).

A separately pinned width128/two-layer/context300 shared PPO check also
completed: `job-y48o6pc3928k`, source `3944ff7`,221 Linux tests passed/two
JAX probe-environment skips. Eight episodes/two tasks,80 query chunks,
two PPO iterations/four optimizer steps,834 total physical actions.
Behavior replay error0; KL.190/.0346. Full raw/behavior/recovery hashes
independently inspected. This qualifies the larger architecture's execution,
not improved performance. [Receipts](../../outputs/libero40-1m-shared-ppo-width128-audit-20261029-r1/README.md).

## Completed official OpenPI40 source runtime qualification

The new source40 collector completed independent train and validation streams
on `libero_spatial:00`, with the official frozen OpenPI `pi05_libero` motor.

| Source partition / Baige job | Episodes / successes | Query chunks | SAC updates | Primitive + settling actions |
| --- | ---: | ---: | ---: | ---: |
| Train `job-f0gv7b32ers5` | 60 / 59 | 606 | 2,220 | 5,796 + 610 |
| Validation `job-1g94ufvin6wa` | 68 / 68 | 617 | 2,400 | 5,848 + 690 |

- Both exited0 and saved paired AD/DPT datasets, source checkpoints and full
  recovery states; final file hashes were independently verified. Train uses
  fifteen nominal states0–14; validation uses ten states40–49 across seventeen
  seeded cases, with four episodes/case. Repeated seeds do not create new
  nominal placements. Learners/replays are independent across partitions.
- Published-recipe warmup exceeds500 query chunks, batch256/UTD20, with
  updates between episodes. Actual SAC-behavior chunks:103 train/112 validation.
  Finite source losses and advancing updates qualify execution, not an
  improving policy or a competent deterministic expert.
- Train source `b48b3da`; validation source `1650ae2`. Focused Linux tests:
  133/138 passed respectively, with two JAX skips in the probe environment;
  actual OpenPI typed/legacy PRNG round-trips passed. Actual hardware was
  NVIDIA RPBZZZ6, not a verified RTX6000; worker times642.16/666.66s.
- These bounded chunk-target runs do not cover all40 tasks or exhaust their
  schedules. The later subset controller comparison is reported above.
  The 127/128 source successes leave little failure
  contrast on this one task and establish no adaptation or memory advantage.
- [Independent receipts, exact indices and interpretation](../../outputs/libero40-source-qualification-audit-20261029-r1/README.md).

## Completed source40 controller and recovery integration

Baige `job-jfsu3lm82kud`, run `libero40-pi05-history-integration-20261029-r3`,
source `7c7a54f`, exited0. AD/DPT full/static each completed25 optimizer
updates with fixed final exports and complete recoveries. The same606 train
chunks and617 validation chunks were reused; this job collected no new
simulator actions. Focused Linux tests153passed/2skipped.

Saved SAC model/optimizer state and RNG restored exactly. Independent GPU
next updates differed by at most3.65e-6 and passed the declared numerical
rtol1e-4/atol1e-5 check; bitwise continuation is not claimed. Source input
recovery hash stayed unchanged. Earlier byte-equality/tighter-tolerance
failures remain preserved. [Exact receipts and limits](../../outputs/libero40-history-integration-audit-20261029-r1/README.md).
No robot or adaptation benefit follows from this25-update technical check.

## Completed RLinf130 history-controller development smoke

Baige Plus `job-immogbrrev71` and Pro `job-4cemq3xgqeiq`, immutable source
`041678bd896669be87f34b8716d32d705a85dbb6`, both exited0 after72 episodes.
Each allocation passed121 focused Linux tests, including the actual
controller/feedback integration check, and qualified motor/features on GPU.
Four clean-source models completed500 optimizer updates each; their primary
exports are final-update500, not validation-selected intermediate weights.
Source-replay SAC completed5000 updates on the same training interactions.

| Controller | Plus | Pro |
| --- | --- | --- |
| Native frozen motor | 8/8 | 8/8 |
| AD full / episode-reset / trained static | 8/8 each | 8/8 each |
| DPT full / episode-reset / trained static | 8/8 each | 8/8 each |
| Source-replay SAC, frozen | 8/8 | 8/8 |
| Source-replay SAC, target online updates | 8/8 | 7/8 |

- Two development variants per suite, one nominal scene each, four attempts.
  One goal family; these are not a blind benchmark or new-goal evaluation.
- Meta-controller optimizer updates during deployment: zero. Context lesions
  reuse full-history weights; trained static models use separate matched training.
- Plus evaluation:8,976 primitive+740 settling actions, worker712.52s.
  Pro:10,976 primitive+370 settling, worker757.42s. Source costs are separate.
- No success advantage or memory benefit established; native and static
  baselines are already at ceiling. This does not test the proposed method's
  capability on an informative population or with a substantive corpus.
- The offline source-replay SAC phase is an additional data-matched control,
  not the published online DSRL source recipe. Frozen SAC uses its mean and
  online SAC uses stochastic actions; their difference does not isolate SGD.
- Augmented-arm models also completed500 updates each, with identical clean
  validation hashes, but robot evaluations were deferred after the user's
  instruction to prioritize genuine failures and the new source40 protocol.
- [Audited compact results and source receipts](../../outputs/libero-rlinf130-candidates-audit-20261029-r1/RESULTS.json).

## RLinf130 clean corpus and Plus/Pro development probes

Frozen `RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT`, revision
`6222623f635769bfc73c9472e29fab9b7fd8e027`; native horizon ten. Strict loading
matched all 667 tensors with no missing/unexpected keys. This is a different
motor/feature contract from the older OpenPI pilot below.

The completed clean corpus contains **six K4 trials, 24/24 successful source
episodes**, all in one cream-cheese-to-basket goal family, original training
task 90:47. Four source scenes supply 16 training episodes; two other scenes
supply 8 local-validation episodes. These are local scene splits of one
training goal, not the planned held-out semantic-task validation bank.

| Clean source partition | Episodes | Executed primitive actions | SAC updates |
| --- | --- | --- | --- |
| Training scenes0–3 | 16 | 2,243 | 4,600 |
| Local validation scenes36–37 | 8 | 1,128 | 2,320 |
| Shared generation total | 24 | 3,371 | 6,920 |

- Actual settling/qualification reset actions: 250; total physical simulator
  actions: 3,621. Worker wall time 441.59s; allocation/setup time is additional.
- Fresh common SAC initialization restored independently per trial; batch 32,
  UTD 20, warmup 0. Source optimizer updates advance, but source episode success
  is already at its ceiling; a useful improving algorithm is not established.
- Paired AD/DPT exports have bitwise-identical context arrays. AD labels equal
  executed noise. DPT-style labels come from the final **candidate** actor;
  its deterministic competence was not independently measured. Source success
  does not make that candidate an expert.
- One nominal-state-vector hash and one initial public-observation hash per
  complete trial in these six saved trials. Shared 4104D RLinf features remain fixed.
- Source generation is counted once across both objectives. It is not a new
  3,371-action cost for every student trained on the same corpus.

The disjoint Plus training partition also completed six K4 trials:
**21/24 successful source episodes**, 3,692 primitive +260 settling actions
=3,952 simulator actions, 7,560 SAC updates, worker 603.58s. Light case 2387 succeeded 9/12;
sensor-noise case 1438 succeeded 12/12. All six first attempts succeeded;
aggregate successes by attempt were 6/6, 4/6, 5/6, 6/6. These are not monotonic
teacher improvements, and the final deterministic actors remain unqualified
as experts. Cases 667/290 and all Pro cases are absent from these source data.

The augmented training bank therefore has 2,243+3,692=5,935 primitive source
actions; common clean validation adds 1,128. Counting shared generation once,
clean-only costs 3,371 primitive +250 settling, while clean+Plus costs 7,063
primitive +510 settling. Student optimization and later deployment costs
must be reported separately. Paired context equality is verified for both
source collections.

The separate Plus/Pro probes used two declared development variants each,
one nominal scene per variant and K4 attempts per method. All share this
same goal family. They evaluate baseline execution and ordinary online SAC;
**no frozen meta-controller was evaluated**.

| Development method | Plus successes / episodes | Plus primitive actions | Pro successes / episodes | Pro primitive actions |
| --- | --- | --- | --- | --- |
| Native independent Gaussian | 8/8 | 1,034 | 8/8 | 1,159 |
| Zero noise | 8/8 | 1,000 | 8/8 | 1,068 |
| Tied Gaussian | 4/8 | 1,865 | 4/8 | 1,817 |
| Zero-first simple search | 8/8 | 1,000 | 8/8 | 1,068 |
| Fresh ordinary DSRL online | 6/8 | 1,474 | 4/8 | 1,828 |

- Search starts at zero and retains a successful arm; it never needed to
  leave zero in these probes. No search advantage over zero is established.
- Online DSRL uses batch 32/UTD 20/warmup 32 and no source-pretrained prior.
  Its fourth attempts succeeded in both cases of each suite, while the native
  policy already succeeded throughout. This does not establish an improvement
  over the frozen motor or a meta-learning benefit.
- Plus: 40 episodes across all five methods, 6,373 primitive +420 settling
  actions =6,793 simulator actions; worker 604.51s. Pro: 40 episodes,
  6,940 +210 =7,150 simulator actions; worker 627.68s. Settling includes motor
  qualification resets. These are development costs, not student pretraining
  costs or final benchmark scores.
- Only two correlated trials per method/suite. Native8/8 leaves weak success
  headroom on these selected cases; the clean 24/24 supplies little
  failure/recovery contrast. Duration/reward variation still exists, but
  neither outcome establishes useful history-dependent adaptation.
- The one-goal smoke test cannot establish new-goal or broad meta-task
  generalization, even if a student later matches these successes.

Clean source pin: `d6a981037b424eb27a38c06ce5c6a240818954e5`;
Plus source: `42fd5176f484aa6b884e52881a2a3021ff18e258`;
both development probes: `b3d1df4a9b77c66650c011e1e70c17f8cdb6a8f1`.
The [independent audit and compact raw receipts](../../outputs/libero-rlinf130-evidence-audit-20261029-r1/AUDIT.md)
verify 168 expected dataset/trace/checkpoint hashes, source costs and per-trial
paired-context equality without model computation. Large checkpoints and raw
image traces remain on Baige.

Adding the declared Plus training partition changes the distribution claim:
clean-only → Plus/Pro tests concerns shifts unseen by the attachment;
clean+Plus → disjoint Plus cases is within-Plus generalization, with any
stronger perturbation-family holdout stated explicitly. Pro remains absent
from pretraining in both arms. See [meta-task contract](META-TASK.md).

## Released π₀.₅ technical pilot

Frozen official OpenPI `pi05_libero`; native horizon10, execution chunk10,
ten flow steps. LIBERO-90 task57: cream cheese → tray. This task belongs to
the source-training bank. Four auxiliary validation initial states40–43,
two attempts each; official held-out LIBERO-10 tasks remain untouched.

| Method | Successes | Interpretation |
| --- | --- | --- |
| Native π₀.₅, independent full Gaussian initial noise | 8/8 | Working motor on these four scenes |
| π₀.₅, 32D tied Gaussian noise | 8/8 | Working tied-noise interface on the same scenes |
| Frozen final DSRL steering actor | 8/8 | No success improvement on these pilot scenes |

- Four independent scenes; repeated attempts are correlated. This small
  sample does not establish benchmark-wide performance or a memory benefit.
- Observed baseline success is already at its ceiling on these pilot cases.
  Useful steering/adaptation headroom remains to be measured elsewhere on
  predeclared training/development cases before large corpus collection.
- Canonical shared prefix features4104D: maximum reference delta0.0.
  Replaying identical native initial noise: maximum action delta0.0.
- Actual GPU motor/renderer binding verified;79 focused Linux tests passed.
- Exact source: `6ec4fff88390041fea6bb23a8fd6fb7e70a7125c`.
- Checkpoint receipt SHA256:
  `ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170`.
- [Raw episodes, source contract and parity receipts](../../outputs/libero-pi05-corpus-20261029-r1/checks/20261029-1714/).

The fresh DSRL learner completed84 training episodes, all84 successful,
12,022 primitive interactions and24,540 SAC updates. Thus its source histories
also have no episode-success contrast on this task. On the same eight
validation episodes, native/tied/frozen-DSRL executed1,128/1,229/1,509 primitive
steps respectively; this pilot does not show a steering efficiency gain.

The deterministic final teacher separately passed its declared competence
gate with7/8 successes on calibration states24–31, costing1,428 primitive
steps. Those are different scenes, so this7/8 is not paired with the8/8 table.
Exact saved-state and next-action recovery passed; the next GPU optimizer
update passed the fixed numerical tolerance rather than bitwise equality.
The complete source checkpoint remains on Baige.

[Final raw episode/progress/qualification receipts](../../outputs/libero-pi05-corpus-20261029-r1/teacher-t57-final/)
include the completed teacher stage. Student adaptation, held-out-task
transfer, Plus/Pro performance and an advantage over ordinary DSRL or trained
static/reset controls remain unestablished.

## Native prefix feature implementation

The original fused model-plus-pooling path differed from separately
materialized native hidden states: maximum absolute error0.0059466 on a π₀
diagnostic. Pooling the same materialized hidden tensor with NumPy versus JAX
differed by at most9.54e-7; repeated fused calls were deterministic.

Canonical feature-v2 materializes native hidden states, then pools in float32.
Its reference error was0.0 in the corrected π₀ diagnostic and in the π₀.₅
pilot above. The qualification tolerance remains2e-4. This verifies the
chosen numerical feature contract; it does not establish a learning result.

## Student CUDA runtime

The original OpenPI Torch2.7.1+cu126 wheel failed on the allocated RTX6000
architecture. An isolated read-only overlay of existing Torch2.7.1+cu128
packages completed two real history-policy forward/backward/optimizer updates.
JAX GPU convolution outputs before and after those updates matched exactly;
actual same-process GPU binding and a complete student checkpoint were saved.
All86 focused Linux tests passed. Shared environments were unchanged.

[Runtime receipts](../../outputs/libero-pi05-corpus-20261029-r1/student-cuda-r1/).
These verify execution compatibility, not useful robot learning.

## 2026-10-30 frozen PPO source validation

Both final source policies completed independent fixed-weight validation:32K4
trials/128episodes on eight source goals and states40/41. Full63/64 successes;
separately trained episode-reset61/64. By attempt (16cases each):full
16/16/15/16,reset16/16/14/15. Both already solve all16first attempts; this
near-ceiling panel does not demonstrate adaptation or failed-attempt recovery.
Episodes within a trial are correlated. Target transfer is a separate test.

Exit0;zero deploymentSGD;unchanged weights;all32trialtracehashes and final
recovery independently verified. Physicalcost22753actions;conservative GPU
charge1203seconds, scheduler terminal1150seconds. Original56k/4h envelope
is retained for explicit remaining-validation aliases; no duplicate allowance.
[Compact terminal receipt](../../outputs/libero40-ppo-source-validation-audit-20261030-r1/summary.json).

The completed source40 corpus has only94failures among2108episodes and
454/527SSSS cases. This motivates testing history/teacher quality before
blindly scaling the same distribution; it does not causally explain the
student's target deficit. [Failure-content analysis](../../outputs/libero40-train-corpus-statistics-20261030-r1/README.md),
[experiment summary](../frozen-vla-steering-summary.md).

## 2026-10-30 broad controls and baseline readout

These are separate, development-stage waves with one initialization/seed per
LIBERO90 task. Use each wave's contemporaneous native control; do not pool the
native31/90,29/90 and30/90 results.

| Completed comparison | Query successes | Actual actions | Charged GPU seconds |
| --- | --- | ---: | ---: |
| All90 noise controls,360 queries | native29/90; zero25/90; tied Gaussian28/90; clipped tied29/90 |122,993|4,411|
| Frozen source PPO target wave,324 episodes | native30/90; full24/90; trained episode-reset26/90 |108,894|4,526|
| Source SAC competence,48 queries | native16/16; trained mean16/16; trained sample16/16 |7,540|532 +78 failed setup|
| Ordinary target-online PixelSAC onset,132 episodes |4/6 at every measured training budget0/1/4/8/13/16 episodes |35,558|1,851|

The PPO fixed-nine-case K1 query scores are5/9 for retained context,
same-weight reset and separately trained reset. No memory or performance
advantage is established. All297 target group hashes passed, both motor
parameter hashes were unchanged and paired-initialization flags passed.
The noise report independently verified all360 query traces, reconstructed
noise, physical/reset parity and recovery records. Equal aggregate clipped
and native scores do not establish casewise equality.

The source-SAC diagnostic is an eight-goal/two-state ceiling panel, not an
all40 result. R1 failed before physics because its Python3.10 probe inherited
a Python3.11 Torch overlay. R2 used the qualified probe path; all48 traces and
the final recovery passed, with zero deployment updates and fixed actor,
critic, optimizer and replay signatures.

The ordinary target-online DSRL onset uses the released authors' PixelSAC
interface: one64x64 image plus8 proprioceptive values, batch256, UTD20 and
warmup exceeding500 query chunks. The frozen pi0.5 motor has horizon10 and
execution10; actual-duration discounting and that horizon are declared
departures from the original pi0 recipe. Tasks0/40 each reach640 chunks and
3,200 updates with no train/query success; task30 reaches800 updates;10/20/50
never clear warmup. This is an onset result, not converged DSRL. Longer
lossless continuations on the two observed failures are outcome-selected
development diagnostics, separate from confirmatory target scores.

Across the student and noise waves all90 initial observation/scene/seed hashes
match, yet four native outcomes flip. On those four, initial images/proprio
and first Gaussian draws match bitwise, but first decoded actions differ by
up to.00195–.00354. This establishes cross-process decoding sensitivity without
identifying a particular kernel cause.

Source history analysis found94 failures/2,108 episodes;454/527 K4 cases have
four successes. Only10 cases show a learned-behavior failure followed by a
learned-behavior success. Thus more of this recipe is not yet supported as a
route to improved adaptation. AD/DPT already use sampled Gaussian likelihood
policies; a deterministic-MSE explanation is ruled out. Dispersion calibration
and teacher quality remain hypotheses to test.

Evidence: [noise report](../../outputs/libero40-noise-controls-report-20261030-r1/report.json),
[failure analysis](../../outputs/libero40-failure-analysis-20261030-r1/ANALYSIS.md),
[native cross-run receipt](../../outputs/libero40-native-crossrun-20261030-r1/README.md),
[experiment summary and next tests](../frozen-vla-steering-summary.md).

## 2026-10-31 ordinary DSRL continuation: first positive development query

On task0/state0/scene seed123, the frozen80-training-episode query succeeded
in249 primitive actions (+10settling). Training cost32,651 actions;54,140
optimizer updates. Query seed238430264, no SGD, unchanged policy/replay and
no query transitions admitted into replay. The independent query40 failed;
first training success was episode41. Task40's query80 still failed, with
zero training successes by episode100. Both continuations remain active.

Task0 query trace SHA256:
`14cba02f88a7649611c9f9754bd58ccf1cb922d2345a7c85add12d7181cacf1d`;
complete milestone recovery:
`02e3e578eb8f6cb57e4c97e1617719dd22abf2c7029fe82f69bc5d03d4aea2bc`.
Both actual bytes were rehashed from saved artifacts. This is one successful
query on an outcome-selected development case, not reliable or few-shot
performance. Exact later160/240 outcomes and full costs are still pending.

## 2026-10-31 final bounded DSRL and batch14 runtime results

The two outcome-selected DSRL cases completed240 training episodes each.
Task0:72training successes,8,154chunks,153,480updates. Task40:0successes,
9,600chunks,182,400updates. Both final240queries fail; task0's80query is the
only successful query among its0/1/4/8/13/16/40/80/160/240 checks. Onequery
percheckpoint cannot estimate stable performance. All250training/querytrace
hashes percase andfinalquery/fullrecoveries were verified. Unique six-case
onset+continuation cost207,594actions/12,322GPU-s, inheritedprefixes countedonce.
[Terminal receipt](../../outputs/libero40-target-online-dsrl-long-audit-20261031-r1/README.md).

The frozen nine-case distribution test is complete:63queries, allseven
methods6/9 withidenticaloutcomes. Motor/controllerbytes and actual original
sampler/cache equivalence passed; eachrawtrace/recovery was read/rehashed.
Cost14,916actions/728GPU-s. This panel/run gives no successsensitivity to
those interventions; it omits20of23 previously observed sensitive cases.
[Report](../../outputs/libero40-distribution-report-20261031-r1/README.md).

Native-horizon batch14PPO runtime qualification:one14K4 batch/56successful
sourceepisodes/795validdecisions,8,313all-in actions/twooptimizersteps, exit0.
Individual/merged behaviorlogp+value replay errors0. Fullcandidate matches the
finite66-tensor checkpoint;63Adamstate entries areatstep2; sampler/global
andtypedmotorRNG/cursorarepresent. Frozen3.3Bmotor parameter hash unchanged.
852GPU-s (UI811);651.24scollection/4.09sreplay/13.46supdate,10,804MiB
sampledwhole-devicepeak. This is one update on14goals, noall40training,
parallel-lane speedup, target-performance or adaptation claim.
[Actual receipt and independent CPU checkpoint read](../../outputs/libero40-batch14-qualification-audit-20261031-r1/README.md).

## 2026-10-31 R3 to native batch14 PPO continuation

The explicit warm-start qualification finished exit0:14 K4 trials,56 source
episodes,894 decisions,55 successes and one SFSS trial before the new update.
The first global update made two Adam steps. All four replay errors were zero;
50 focused tests,2 motor-RNG tests and27 independent saved-state checks passed.
Inherited238 updates/476 steps and all old numerical/RNG leaves were retained;
the operational Adam clock is478, new cursor14 and no pending batch. Candidate
weights equal the full recovery, and motor parameter hash c565dd3f... is unchanged.

Cost:9,272 new actions (9,262 training-episode actions plus10 qualification
settling),712.040s collection/replay/update,849.842s cold worker and910 charged
GPU-seconds. The cumulative prior+new resource cost is205,236, which includes
validation/qualification overhead and is not dataset size. The SFSS pattern
has no matched control and establishes no adaptation, memory or target gain.
[Audit and exact hashes](../../outputs/libero40-batch14-warm-qualification-audit-20261031-r1/RESULT.md).


## Source continuation snapshot, 2026-10-31

Canonical source leaves now contain583 K4 training cases/2,332 episodes/
33,729 decisions across all40 goals and146 cases/584 episodes/8,202 decisions
in fixed disjoint source validation. Training all-in cost351,297 actions;
validation85,301; historical source overhead21,253 once, total457,851.
These are source collection counts, not target performance. All224 episodes
added by the continuation qualification and two bridges succeeded, so they
provide no additional failure-recovery evidence. See the
[canonical counts](../../outputs/libero40-sac-source-bridge-audit-20261031-r1/canonical-counts.json).

Real zero-GPU capture/assembly passed34 tests and25 saved-result checks and
produced immutable paired inputs for fitting. No new robot result is yet
available from this snapshot. [CPU receipt](../../outputs/libero40-source-pipeline-cpu-audit-20261031-r1/README.md).


All four controllers completed the new interim100-update fit at fixed25/100
milestones, with16 independently rehashed policy/recovery files and26 audit
checks passing. Zero new simulator actions;571 conservatively charged GPU-s.
AD full/static fixed-window validation NLL51.806677/51.806689. DPT full/static
validation NLL increased13.5081/13.4585 at25 to16.0686/15.8921 at100 on the
same sampled windows. Logged training minibatches differ, so their losses do
not establish a trend. Possible overfit/distribution/calibration explanations
are recorded in the summary. These are fitting metrics, not a robot or memory
result. [Interim audit](../../outputs/libero40-source-topup-interim100-20261031-r1-audit/AUDIT.md).


## Pipeline validation after interim fitting

No new robot-performance result is claimed by the following checks. Fixed100
LIBERO90 evaluation passed48 allocated CPU tests, including actual saved
matrix/manifest/assembly and all eight model/recovery files with no skips.
PPO production glue passed7 CPU tests and actual full-checkpoint migration,
preserving its complete operational state without any optimizer step. The
source collection plan is ready with38 assignments plus two retained leaves;
its budget is still unfunded. Source and target GPU expansion require their
separate existing resource decisions.

DPT's fixed-window validation targets come from independently trained VAL
teachers, rather than the matching TRAIN actors on heldout states. NLL thus
includes teacher/stage generalization. Checkpoint identities alone do not
measure actor-function disagreement; the observed25→100 regression has no
established causal explanation. See the
[provenance evidence](../../outputs/libero40-dpt-validation-teacher-provenance-20261031-r1/ANALYSIS.md).
Historical AD/DPT deployment already samples stochastically, ruling out a
mean-only evaluation explanation. No new sampling diagnostic was launched.
