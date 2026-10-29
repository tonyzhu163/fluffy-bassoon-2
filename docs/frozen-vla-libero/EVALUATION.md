# Frozen VLA steering evaluation

Verified robot evidence only. Method: [LIBERO interface](../frozen-vla-libero.md).
Paper: [draft](../paper/DRAFT.md), [decisions](../paper/DECISIONS.md).

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
HLO fingerprint extraction was unavailable. The next robot comparison uses
one motor process for both source banks, native and frozen final target-SAC.
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
