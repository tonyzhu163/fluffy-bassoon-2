# Measured qualification and existing final100 handoff

Dated 2026-10-31, Asia/Shanghai. **Static handoff; no new resource admission or
launch.** This updates the existing final-bank proposal using R6 measurements
and exact SAC100 source identities. No implementation, wrapper, tests,
collector, teacher panel, fit or target evaluation is added here.

## Closed SAC65 compatibility result

Baige `job-ug06puib7vek`, exact code
`80ffd1bba1c27ce40cc7f7d36682250e6fe97a07`, completed Success/exit0 on
0GPU/8CPU/64GiB. All150 selected fixtures passed with zero failures/errors/skips.
Projection, paired AD/DPT export and audit passed with clean phase/final owned
teardown. Torch2.11.0+cu128 resolved to the approved shared package, with no
`.pth` processing, forbidden imports, CUDA initialization or optimizer steps.

| Actual phase | Seconds |
| --- | ---: |
| Tests | 35.619 |
| Full projection | 495.045 |
| Paired training export | 683.673 |
| Paired audit | 9.888 |
| Parent including setup/cleanup | 1,254.209 |
| Scheduler Running07:30:23–Success07:51:33 | 1,270 |

Projection covered143,506 unique files /56,795,082,979 bytes over26 roots. The
paired training arrays contain21,048 decisions /390 K4 /1,560 episodes /
651,638 actual source actions, including the protected selected-trial prefix.
Historical AD/DPT fields match. AD targets are executed noise; DPT changes
only target labels. The allocated audit checked paired arrays, native masks
and duration sums. Owner inspection rehashed the index/NPZs and passed all21
ZIP members' CRCs in each NPZ.

| Artifact | SHA256 |
| --- | --- |
| Projection metadata | `8f03b480cbb8bd0e57a94d9004f8fade2b845c4c5dea476f9c75e58ee5f9051c` |
| Training index | `c697f339fa25eacc25b5972efb4f67db047ff050bb7ed116f5fc206645fade31` |
| AD NPZ | `7c25166ad347bf1b8fc308f1d4cc1be82c487bccef9c5be0be9e90c1b06ef34d` |
| DPT NPZ | `34bb016cd805ac2cbbfa959fac2e917735a7505ea8c2ad7dc4f471bc59e37add` |

[Independent review](SAC65-R6-QUALIFICATION-20261031.json) retains the small
raw phase/import/JUnit/projection-summary/paired/terminal/cost receipts. This
review checked their bytes and consistency and independently rehashed the full
saved58MB projection metadata, reconciling file count/byte sums. NPZ hashes/CRC
and tensor equality are attributed to the owner/allocated audit; they were not
repeated on macOS. This qualifies the fixed **SAC65 reference**, not a final100
train/validation bank, teacher competence, student fit or robot adaptation.
The execution owner's complete evidence is committed at
`5d3255cf8a8b5d20eb42da442be088fa181e24c4`.

The original CPU envelope remains CLOSED at620 actual +1,180 sealed. The
separate qualification grant closes at71 R5 +1,270 R6 actual +459 sealed =1,800
seconds. Combined actual CPU-gate cost is1,961 scheduler seconds /15,688 CPU
core-seconds /0 GPU-seconds. Pod1,260 and parent1,254.209 seconds are separate
clocks. Unused capacity stays sealed and funds no proposal below.

## Exact source100 inputs and pending validation ancestry

Closed SAC100 source collection contains600 K4 /2,400 episodes /1,003,790 actual
actions,638,500 updates per optimizer /1,915,500 total optimizer steps. The saved
audit has zero pending/uncertain source actions. DB scientific lineage and e825
execution remain unchanged; PPO/reset runtime570 is not substituted.

| Input | Exact identity |
| --- | --- |
| Current61-root source ancestry | [SAC100-SOURCE-REFERENCE-20261031.json](SAC100-SOURCE-REFERENCE-20261031.json), SHA `403e35ba951a7ebe5b595ad73fbee17a84f0cc856e09758fa2fb5fddf90d9bd7` |
| Source tip | `slices/0060/experiment/continuation-manifest.json`, SHA `b2ae61a72c03bffb5c6e174d80077fb068089dcb0660a19732f4c83bf427999a` |
| Source coordinator | `slices/0060/experiment/coordinator-recovery.pt`, SHA `20c512080f3012452343a0c7d8aff67ee03966add4497d61256854e52d5d611f` |
| Preselected per-method source100 snapshot | `slices/0061/experiment/stages/iteration-100/sac-recovery.pt`, SHA `59588f07be18d4a1339af0c7536c62ef735a713f3c933d7e372fb3f4f4fa7e90` |

Remote paths share prefix
`/mowenjie/fluffy-bassoon-2/outputs/robotwin-native-continuation-sac-20261030-r1/`.
The source coordinator is1,043,148,177 bytes. Source and stage checkpoints each
passed210-member ZIP CRC inspection in owner audits. This amendment rehashes
retained JSON, not checkpoint tensors. Full paths, counters and evidence hashes
are in [the machine-readable handoff](MEASURED-HANDOFF-20261031.json).

At the retained07:55/07:57 snapshot, the closed source chain ends at0060 while
stage100 validation is being recorded in0061. Those61 roots do not include the
final validation corpus. Before final train/validation projection, the owner
must supply actual completed stage100/closed ancestry containing its real
validation root. Do not manually append an unbound root or infer validation
completion from a checkpoint filename. The existing converter checks distinct
stage serialization against closed-tip actor weights, source counters and
source iteration. The fixed59588 source snapshot stays the teacher candidate;
a later final coordinator may have different serialization after evaluation.

Final training retains all600 K4 /2,400 episodes /1,003,790 source actions,
including warmup/failed histories in chronological order. Export training as
AD+DPT; export separate source stage1006 K4 /24 episodes as actual-action BC+DPT.
Require episode/K4 case disjointness, accepted seeds, masks, input origins and
matching source/teacher bindings. Target-adapted SAC clones/replay are excluded.
R6 did not run this final100 validation export or the existing deferred final100
index-admission fixtures; these remain final-bank qualification within the
existing proposal, with no new diagnostic pilot.

## Measured basis for unchanged proposed resources

A simple600/390 K4 scale gives about762s projection +1,052s paired train export
=1,813s, plus15s paired audit; the analogous unique-byte estimate is87.38GB.
These are planning estimates. Roots grow from26 to at least61; metadata costs
need not scale linearly. Validation export, setup, storage behavior and remaining
admission checks add work. Final-bank inventory/throughput remains unmeasured.

| Existing phase proposal | Resource | Scheduler cap | Internal clock | New robot actions |
| --- | --- | ---: | --- | ---: |
| Final100 projection + train/validation export/admission | 0GPU/8CPU/64GiB | 7,200s | One6,900s parent deadline starts before projection; export gets only remaining time | 0 |
| All-new native/mean/sample source teacher panel | 1GPU | 3,600s | Six source goals ×K4 ×3 conditions, including startup/reset/probe/failure/cleanup | At most32,400 |
| Conditional native+mean, with verified paid sample reuse | 1GPU | 2,700s |48 new episodes;24 sampled episodes reused only after exact audit | At most21,600 |
| Four sequential AD/DPT full/static fits | 1GPU/8CPU/32GiB/shm8 | 3,600s |3,300s parent; each process750s external/600s fitting clock | 0 |

Teacher rows are alternatives: default GPU proposal7,200s total; conditional
sample reuse6,300s. The remaining CPU proposal is7,200 scheduler seconds,
separate from paid R1–R6. No row is a reservation or launch permission. Exact
code/runtime/host/storage/resource receipts remain execution-owner admission.
Reuse the existing qualified ownership/runtime path; add no duplicate wrapper.
R6 itself remains fixed to65-round inputs and its1,580s clock; it is not a
final100/6,900s launch profile. The owner must bind/review the actual final
phase contract without editing R6 or assuming the larger proposal is executable.

Labels stay deterministic physical `1.5*tanh(mean)`. Stochastic stage100 SAC
validation does not measure mean-action competence. Default teacher measurement
remains72 episodes /32,400 maximum new actions. Reusing24 paid sampled source
episodes requires exact59588 actor/source state, stochastic convention, private
interaction RNG, accepted cases/initial bytes, no updates, and original cost
counted once. Reuse is unadmitted. Weak mean stays weak; it cannot silently
switch labels/teachers or select favorable target endpoints.

Original source parent SHA
`1f80c08310dfa4543ec7a748ca6e2bf9aad589cd55978a402210abb0d83bdbc0`
caps SAC total actions at1,194,400 and source at1,100,000. Stage100 source
validation10,800 and development targets12,800 are existing obligations, not
extra teacher capacity. A fresh stable canonical ledger/reservation snapshot
and charged GPU cost must establish separate new-action/time authority after
outstanding obligations. Subtracting source actions alone does not establish
usable residual. Borrow no sealed CPU/student-panel/trained-N6 balance and
repurpose no original source/evaluation cap.

## Four fits and source-only selection stay fixed

The existing runner requires `teacher_source_qualification_reviewed` before
executing export as well as fit. The owner's sequence/admission must satisfy
that existing requirement; R6 compatibility cannot set that flag or authorize
waiting for an unadmitted teacher panel inside a CPU allocation.

Keep AD full, AD static, DPT full and DPT static, each seed0/500 updates with
the existing architecture/optimizer/window recipe: four trajectories,2,000
updates, fixed25/100/500 exports. Predeclare `candidate-000500.pt` as primary.
Source validation may diagnose curves; targets select no weights/teacher/bank.

Unchanged `run_offline.py` and false-admission bindings remain executable
preparation. After qualification, bind source/teacher/projection/train/validation
hashes to a fresh owner-hashed copy with actual remaining CPU time or separately
admitted GPU fit time. Keep full optimizer/sampler/RNG recoveries and fixed
milestones. R6 completed no final-bank fit, teacher panel or target evaluation.
Early SAC5 four100-update student hashes/results/costs remain separate. Direct
PPO source100 completion adds no PPO/reset offline fit arm here.

[Prepared bindings](bindings.source100.prepared-20261031.json) and
[prepared teacher plan](teacher-qualification.source100.prepared-20261031.json)
fill only the exact fixed source/teacher identities in copies of the existing
templates. Every admission remains false, resources/time remain unreserved and
the final projection/index/host bindings remain unresolved. They are concrete
review inputs, not execution manifests or a migration of the active R6 code.
[Printed command preview](COMMANDS.source100.preview-20261031.txt) confirms the
existing runner can describe the two exports/four fits without `--execute`;
no exporter, trainer or model was launched to generate it.
