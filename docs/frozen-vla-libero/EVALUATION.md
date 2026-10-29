# Frozen VLA steering evaluation

Verified robot evidence only. Method: [LIBERO interface](../frozen-vla-libero.md).
Paper: [draft](../paper/DRAFT.md), [decisions](../paper/DECISIONS.md).

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
  schedules. No controller robot result under the new source40/target90
  contract exists yet. The 127/128 source successes leave little failure
  contrast on this one task and establish no adaptation or memory advantage.
- [Independent receipts, exact indices and interpretation](../../outputs/libero40-source-qualification-audit-20261029-r1/README.md).

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
