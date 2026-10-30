# Frozen-VLA ICRL implementation inventory and exporter handoff

Dated snapshot: 2026-10-30, Asia/Shanghai. Shared plan:
`162d6942e4af267e231d2059d1cf2a77a01a1144`,
`outputs/icrl-master-plan-20261030-r1/PLAN.md`. This is a dated inventory,
not another live job ledger. Owner reports and immutable receipts below are
evidence; this document does not make a fresh scheduler-status claim.

LIBERO baseline rows were amended on2026-10-31 from committed completion
receipt `1a65226e0d3b6a68cf59808028245b3da5be3d07`; see
[bounded results and provenance](AMENDMENT-20261031-LIBERO-BASELINES.md).
RoboTwin native AD/DPT rows were amended on2026-10-31 from the completed
first frozen panel; see [bounded results and provenance](AMENDMENT-20261031-ROBOTWIN-FIRST-PANEL.md).
This reconciles the2026-10-30 native implementation/fitting amendments.
Untouched rows retain their original snapshot dates. Neither completion
receipt qualifies the future parent-death guard or changes a live source run.

The human's coordination authorization was verified in coordinator chat
`01a18be4-a6ab-79a2-af3d-13f76308e253`: “can you coordinate all the other
agents to follow this? keep me posted on status”. Export instructions were
sent to both execution owners under that authorization.

## Status and common contracts

Implemented means an executable path exists. Qualified specifies which tests
or runtime passed. Fit complete means an offline optimizer reached its declared
endpoint. Running refers to the owner's source job, not a completed adaptation
result. No cell below has established a robust held-out ICRL or architecture win.

| Contract | LIBERO noise cells | RoboTwin noise cells |
| --- | --- | --- |
| Source / targets | Source40; full90/Plus/Pro targets, source cases separate | Six source / six target tasks; adjust_bottle and move_stapler_pad currently development |
| Frozen motor | Official OpenPI40 receipt `ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170`; OpenPI `215abfb217dbac7d5f1273282331b9b1866c0479` | Motus revision `effc6e1a99bd4d0edf9447e8ba313def67649cfe`; RLinf `c605b9d55ed93a180119b099e6f836abff293b08`; RoboTwin `dca9ec682688821c42944eacd7f5dd9bfb15396f` |
| Features | Prefix-v2: visual2048 + instruction2048 + state8 =4104 | Masked prefix2048 + normalized proprio14 =2062 |
| Feature provenance | Exact `feature_contract` in source/export manifests; tested canonical JSON SHA256 `90a994b51120ea3590a863a5b2e552bdd3a7d82a800be8a53d4d3d30adc697c0` | Runtime contract records pooling, normalization, model/runtime pins; norm stats SHA256 `1dce5ceb9a98eaa6ae587795e0e8977dc3ce046a7ba0a5ec07858ea896ed1adb`; actual motor tensor hashes remain in owner qualification receipts |
| Noise | Physical32D, repeated H10, bound1 | Physical H32x32 =1024D; steering bound1.5; native Gaussian is unclipped standard Gaussian |
| Policy inputs | 4104 + previous32 noise + reward/done/start =4139 | 2062 + previous1024 physical noise + reward/boundary/episode-clock/primitive-clock =3090 |
| Current Transformer | GTrXL width128/layers2/heads4/context300; offline source windows128 | GTrXL width128/layers2/heads4/memory300; full-trial replay; no factory for other cores |
| Context and resets | Source histories grouped by task; source trial boundaries and teacher resets explicit; static/full fitted separately | Same-task K4 trials; original task-specific horizons and reset distribution; full/reset PPO trained separately |
| Deployment reporting | 0/1/2/4/8 supports followed by separate query | Existing K8 episode indices1/2/4/8; index8 is seven prior episodes |
| Budget / seed authority | Existing bank/fit/resume ledgers remain binding; future ~1M per learner schedule/resource estimate owned by LIBERO owner. Seed/config in exact fit/recovery contract | New source seed20261030; 100 whole rounds x six K4 trials per learner, maximum1.08M source actions before success; actual parent/child reservations in wholetrial protocol |

These interfaces are not dimension-matched. Hashes/settings apply to the cited
receipt, not automatically to a later collector optimization or migration.
Original LIBERO400-step source banks and earlier RoboTwin400-step data remain
separate from future native-horizon contracts.

## Concrete noise-interface matrix

All rows inherit the benchmark contract above. Architecture cells require
their own compatible actor/critic, exporter/trainer and deployment qualification;
the existence of a compact benchmark core does not implement that VLA cell.

| Benchmark / objective | Transformer / GTrXL | GRU | CTM | Next dependency |
| --- | --- | --- | --- | --- |
| LIBERO AD | **Implemented; all40 full/static fits complete**, source33827ab, four-arm fitter. PPO-source paired exporter additionally qualified at9e2bb54 | Planned; generic GRUCore exists, VLA history factory/checkpoint/deployment path not wired | Planned; PersistentCTMVectorCore exists, VLA history factory/checkpoint/deployment path not wired | Bind immutable source assembly and candidate hashes to source validation / separate-query robot evaluation; supervised trained episode-reset still missing |
| LIBERO DPT-style | **Implemented; all40 full/static fits complete**, source33827ab. Existing source bank is SAC; PPO-source DPT is a distinct data arm. Bounded source competence control is now complete for EIGHT task-specific SAC actors: native/mean/sample each16/16 | Planned; same-data architecture comparison after useful reference | Planned; same-data architecture comparison after useful reference | The ceiling-limited eight-actor control does not establish joint all40 teacher expertise or a teacher advantage. Keep original fit/model pins and separately bind target outcomes; no AD/DPT target result is newly inferred from this receipt |
| LIBERO direct RL2/PPO | **Implemented; frozen source validation and bounded target panel COMPLETE**. Source validation full63/64, separately trained reset61/64; contemporaneous LIBERO-90 native30/90, frozen full24/90, trained reset26/90. Fixed-nine K1 full/same-weight reset/trained reset all5/9 | Planned; current RL2ActorCritic hardcodes GTrXL | Planned; current RL2ActorCritic hardcodes GTrXL | Preserve the source-only frozen checkpoints and one-seed/fixed-bank limits. No reliable memory benefit or robust ICRL claim; source-training quota completion is not inferred from frozen evaluation completion |
| RoboTwin AD | **Implemented and qualified; full/static100-update fits and first frozen development panel COMPLETE**. Bottle full5/8, static5/8, same-weight reset6/8; stapler all0/8 | Planned; no noise-interface supervised GRU integration | Planned; no noise-interface supervised CTM integration | One fit seed and early SAC5 bank only; no early history advantage. Final source endpoints, independently trained supervised reset and architecture comparisons remain separate |
| RoboTwin DPT-style | **Implemented and qualified; full/static100-update fits and first frozen development panel COMPLETE**. Bottle full5/8, static6/8, same-weight reset5/8; stapler full/static0/8, same-weight reset2/8 | Planned | Planned | Fixed source SAC5 deterministic-mean labels; teacher competence unqualified. Reset E4/E6 wins do not establish ICRL; final source endpoint and broader target performance remain unmeasured here |
| RoboTwin direct RL2/PPO | **Implemented and runtime qualified**, originally sourcedb01840; independently trained reset PPO is another worker. Owner reports continued100-round source collection separately from the completed frozen student panel | Planned; RecurrentNoisePPO currently hardcodes GTrXL | Planned; RecurrentNoisePPO currently hardcodes GTrXL | Complete source updates/validation at rounds10/25/50/100 and bind actual source/target outcomes. Student panel completion does not complete these source streams |

### Offline controls versus online controls

| Control | LIBERO | RoboTwin |
| --- | --- | --- |
| Independently trained static AD and DPT | Complete500-update fits in current all40 bank; same data/initialization/optimization recipe | Complete100-update fits on same early source bank; both static models included in completed two-task K8 panel |
| Independently trained supervised episode-reset AD/DPT | **Absent.** Generic `vla_history_train --mode episode` exists, but current source40 fitter uses `MODES=(full,static)` | **Absent** |
| Same-weight inference episode reset | Implemented; a lesion, not another trained student. Completed fixed-nine PPO K1 panel scores5/9 for full, same-weight reset and trained reset alike | AD/DPT lesions implemented and completed in first frozen K8 panel; all four full/reset E1 traces match exactly. PPO reset path remains separate |
| Independently trained episode-reset PPO | Frozen-source and LIBERO-90 query panels complete:61/64 source episodes and26/90 target queries; fixed-nine K1 is5/9 | Implemented/qualified; source worker status and eventual endpoint results owned separately |
| Native, reactive SAC frozen / target-updating, reward/history controls | Bounded noise screen, frozen PPO targets, eight-actor source-SAC competence and six-case target-online onset now COMPLETE; exact panel-specific results below. Full all40 teacher advantage and target-online convergence remain unestablished | First frozen panel COMPLETE: bottle native6/8, zero6/8, SAC5 mean5/8; stapler all0/8. Target-updating SAC and reward-blind outcomes are not supplied by this panel |

Static policy success is generalization evidence. An inference lesion and a
separately trained reset model answer different questions. Reward blindness
does not necessarily remove information in visual/dynamics observations.

## Exact executable pins and evidence

| Item | Pin / artifact / owner-reported job | Qualification / remaining result |
| --- | --- | --- |
| LIBERO source40 SAC assembler/fitter implementation | `f21c6592cf497c519461504183946462efbec627`; `vla_source40_offline_pipeline`, `vla_source40_subset_training`; source data admission `760a5dafb110e7864f0314613540f3ce028dd6a0` | Strict all40/fixed8 assembly; current bank527 trainK4 cases/2108 episodes and146 validationK4 cases/584 episodes;326277+85301=411578 selected source actions |
| LIBERO all40 fit runtime | `33827ab9a7ad2bc1b767f2cffd0a48f220097530`; owner abbreviated job7pdp | ADfull/static and DPTfull/static each500 updates with25/100/500 exports and full recoveries. Full job/artifact/checkpoint hashes remain owner binding; no new robot result inferred |
| LIBERO PPO source / frozen evaluation lineage | Source learner runtime `c4cadc8a9a8cebbe2cc977834e19ad8ba36bb523`; frozen source validation `9002a2dd352e6176c43943c1e46bf2b85eddc397`; target runtime `7780fb38b61fb37a3e2c0aec67ffea8e8e9c0b94` | Original325-test/resume evidence remains historical. Committed completion receipt now binds unchanged full/reset frozen model hashes,128 source episodes, all90 K0 queries and fixed-nine K1 controls; completion is scoped to those panels |
| LIBERO bounded baseline completion receipt | `1a65226e0d3b6a68cf59808028245b3da5be3d07`; `outputs/libero40-baseline-completion-20261030-r1/{README.md,receipt.json}` in execution owner's checkout | Eight terminal runs,297,738 actions,12,601 charged GPU-seconds; source pretraining excluded. Noise native29/zero25/tied28/clipped29 per90; PPO native30/full24/trained-reset26 per90; source SAC native/mean/sample16/16; six-case onset4/6 at all declared0/1/4/8/13/16 budgets. See amendment for exact pins, costs and scope |
| LIBERO PPO exporter implementation | `9e2bb54a6bdc6caa6bee1ce757153189eac721d0` |66 Linux CPU tests and real-data export289 paired decisions +125 DPT decisions; deterministic repeat byte-identical;83/33 pinned inputs unchanged |
| LIBERO exporter handoff / evidence | `105d4dea0023112ec72b733d4b5b7f6727c18068`, branch `codex/ppo-dpt-export`; `outputs/ppo-dpt-export-check-20261030-r1/` | Includes usage, test log, smoke script/receipt; zero new simulator/VLA calls. Not source-teacher or robot-performance qualification |
| RoboTwin source runtime | `db0184089fa88a5f96c153f9b80029bb91b77def`; `outputs/robotwin-wholetrial-20261030-r1/{PROTOCOL,LAUNCH}.md` in owner's checkout |112 Linux tests, independent GPU probes and native parity0.0 on all three workers |
| RoboTwin original source launch snapshot | PPO `job-6phpm6rnvuf0` / `robotwin-wholetrial-ppo-20261030-r2`; SAC `job-yd05kvzuqlh9` / `robotwin-wholetrial-sac-20261030-r2`; reset `job-d8moi1k3ufye` / `robotwin-wholetrial-ppo_reset-20261030-r2` | Historical launch receipt16:19 and confirmation16:25 Shanghai. ~29–32 collection actions/s/GPU; first8 actual cases match. Later continuations remain source-owner evidence; these IDs are not a current scheduler claim |
| RoboTwin corpus schema implementation | `32271f093efa92f46771536bea05c4e185c29544`; current source above includes it | `robotwin_public_decisions_v1`: JSON reset/episode/chunk manifests, content-addressed public-frame NPZ, policy versions, intent/native_complete/complete distinction |
| RoboTwin PPO input/horizon contract | `74282448befc4a5f43d73a6ef3902bfe65533b17`; source runtime above includes it | Task-specific horizons, causal clamped clocks, actual physical prior noise. Native offline adapter is now separately qualified below |
| RoboTwin native exporter/trainer | `1aeea4f9fcc307bb147def92374bb81a3e2cb50e`; CPU `job-tmocet3s5ur4` |46 Linux CPU tests and real native export/teacher/toy-fit checks; source train1539 decisions/120 episodes/30 K4/47602 actions. Explicit validation340 decisions/24 episodes remains separate |
| RoboTwin first source-only fits | Model code `1aeea4f9fcc307bb147def92374bb81a3e2cb50e`; `job-ielg6papl3hn` | Four full/static AD/DPT endpoints at100 updates, one seed;39621 sampled positions per model from1539 unique decisions. Recovery/model identities bound in panel provenance |
| RoboTwin native integration and first frozen panel | Integration `82c4d8c006db064096549538e9cbab0c2f05facd` CPU `job-tmtbmt14kl6w`; live `2fd2096906e52aabeaeb984f333fe1aaa6c4e196`, Baige `job-ekg1ivu6j8pg` |151 integration tests plus5 actual CPU admissions;22 live-wrapper tests. Panel COMPLETE18 K8/144 episodes,47137 actions,2951 scheduler GPU-seconds, zero SGD/pending/uncertain. Owner independent audit22135 checks passed; native parity0.0 in all18 units |

Owner updates are authoritative for later status. No source teacher is called
expert solely from optimizer counts, a final filename or a finite export.

## LIBERO exporter handoff

Fetch `codex/ppo-dpt-export`. Implementation commits in order:
`22555a05216bafc55c3739b281e5a75a33303ce9`,
`c0abf06a5c68ea09b5e19574ac30d9f711032f27`,
`9e2bb54a6bdc6caa6bee1ce757153189eac721d0`;105d4de adds evidence.
Review/cherry-pick into a future clean owner pin. Existing active worktrees,
recoveries and source budgets are not modified by this handoff.

```sh
python -m fluffy_bassoon.meta_rl.vla_source40_ppo_corpus \
  --source-run <immutable-experiment-snapshot> \
  --recovery-sha256 <learner-latest-SHA256> \
  --teacher <source-trained-candidate.pt> --teacher-sha256 <teacher-SHA256> \
  --objectives ad dpt --output <fresh-output> --device cpu --threads 4 \
  --source-commit <actual-clean-exporter-commit> \
  --expected-hostname <allocated-Linux-host>
```

Frozen source validation uses the SHA of `validation-recovery.pt` and explicit
`--objectives dpt` or `behavior_cloning`. It cannot be fabricated into AD.
The standard NPZ/manifests feed `vla_history_train`; SAC-specific all40 assembly
is unchanged. A PPO bank is a separate data arm. The source-only converter
rebuilds chosen-teacher memory from actual history, keeps targets out of
context, checks motor/features/noise and rejects uncommitted trailing trials.
Labeling is deterministic tanh(mean), whose competence still needs measurement.

## RoboTwin native adapter contract and qualified path

Adapter ownership: this exporter chat, coordinated with RoboTwin owner
`01a18762-9db6-74c2-940a-96524803e454`. Dedicated native exporter, trainer and
frozen deployment paths are qualified at the exact pins above. Earlier handoff
requirements below are now implemented for this bounded native path.

Do not call the LIBERO exporter with reshaped RoboTwin records. Both schema
and controller semantics differ. Native previous noise is already scaled by1.5;
teacher labels must be `1.5*tanh(mean)` in H32x32 order. The native scalar inputs
are reward, episode boundary, episode clock and task-horizon primitive clock,
not LIBERO reward/done/start. Retain2062 features and all4 native scalars.

The existing `BehaviorPolicyArchive` records immutable actor/value weights,
`policy_version`, parameter SHA and file SHA. Full source recoveries retain
PPOConfig/config dimensions, learner/schedule/sampler/optimizer/RNG, pending
rollouts and iteration/evaluation cursors. The adapter must select only source
data covered by a declared committed boundary; unresolved intents/native-only
records and target/validation adaptation cannot become source examples.
AD groups same-task trials across real behavior updates. DPT uses a separately
pinned source-trained teacher, reconstructed causal memory and original reset
rules. Labels never rewrite historical noise, reward or boundaries.

The native path retains3090-token semantics and1024D/bound1.5 labels;
generic `vla_history` dimensions alone do not make its +3-scalar/bound1 LIBERO
consumer compatible. Actual first-fit DPT labels use the feature-conditioned
SAC5 teacher, deterministic `1.5*tanh(mean)`, without relabeling historical
noise/rewards. Source teacher expertise remains unqualified.

The first frozen panel completed on the two development tasks with one paired
K8 per condition/task and one fitting seed. It shows no early history advantage.
See the [panel amendment](AMENDMENT-20261031-ROBOTWIN-FIRST-PANEL.md) for the
full nine-condition table, hashes, accounting and audit limits. Supervised
trained episode-reset, GRU/CTM and final100-round source endpoint comparisons
remain open. Future parent-death guard `c166171b847ada9a20a9629a07d5a0100b517097`
is source-prepared/CPU-staged only, pending its separate allocated Linux gate.

## Conditional later cells

AD->PPO and DPT->PPO need an explicit supervised-to-PPO policy/critic/optimizer
initialization contract; compact/native bridge warm-start code is not this
noise-interface integration. GRU and CTM need model factories/checkpoint schema,
matched capacity/context/gradient settings and allocation qualification per
benchmark. Other steering interfaces (KV/features/residual) and LoRA remain
conditional specifications; LoRA is parameter adaptation. CMM is gated on its
failed calibration being resolved; compact/ML10 or historical ManiSkill tests
do not qualify any current LIBERO/RoboTwin matrix cell.
