# Objective-aware AD/BC-first execution handoff

Dated2026-10-31, Asia/Shanghai. **Source-prepared; allocated CPU qualification,
actual final-bank payload admission and resources remain pending.** This is
one change to the existing exporter/runner, with no new wrapper, collector,
student architecture, fit arm, physical action or allocation. Mean-teacher
competence is a DPT requirement; it is not an AD/BC prerequisite.

## Exact existing path and behavior

`robotwin_history_corpus.export_corpus` now loads and infers a teacher only if
`dpt` is requested. AD/BC use the actual executed physical noise, preserving
all history fields. Their unused teacher metadata is null; optional stale
teacher arguments create no teacher provenance claim. The same source recovery,
acting archives/tensor hashes, projection aliases, native masks/durations,
chronology, case selection, feature/runtime and physical-count checks remain.

The existing `run_offline.py --objectives ad` prepares train AD plus source
validation BC and selects only the existing AD full/static fits. The hashed
execution bindings must explicitly select `objectives: ["ad"]`. Every source,
projection, case-split, funding, actual-host, code/input SHA and GPU-fit admission
remains mandatory. `teacher_source_qualification_reviewed` stays false and is
not required for this branch. Any DPT request still requires that review before
Git/model/subprocess work; DPT source/teacher binding and inference remain.

The default remains paired AD+DPT exports and four500-update fits in the
declared order. The trainer, normalization, architecture, batch/window sampling,
optimizer, source-only selection and fixed `candidate-000500.pt` primary do not
change. This branch starts the two AD trajectories already in that four-fit
proposal, not another fit arm. No target outcome selects a bank or checkpoint.

## Final100 data is closed; actual full-bank admission remains

Use the owner-issued FINAL62-root source+validation ancestry, not the historical
61-root evaluating snapshot:

| Input | SHA256 |
| --- | --- |
| [Final62-root ancestry](SAC100-FINAL-REFERENCE-20261031.json) | `0af5082349ec1a032eff6b752e3b91fd473b8c922a394c496b4f75bceff68c7e` |
| [Terminal provenance tip](SAC100-FINAL-TIP-20261031.json) | `5b978a88c423ccdc7e1921f309ec047d0fc6cfe034d07e1f9e5523e6cc2f25d4` |
| Final coordinator in0061 | `f551e3f28d14702556392e67dafdf62f094c824f9069511219f8dc5af4db9fd8` |
| Fixed source100 stage/acting snapshot | `59588f07be18d4a1339af0c7536c62ef735a713f3c933d7e372fb3f4f4fa7e90` |

Training must contain600 K4 /2400 episodes /1003790 actual source actions;
source validation contains6 K4 /24 episodes /10180 actions. Source sample
validation scored2/24; mean competence is unmeasured. This affects DPT teacher
interpretation and does not substitute target data or filter AD histories.

Projection must retain all raw bytes, including non-training phases, and bind
actual stage59588 source state/actor/counters to the final closed tip. Existing
export selection alone admits `source` into train and iteration100
`source_validation` into BC validation. Targets/native/preflight/teacher-panel
data cannot become training rows. Verify episode/K4 case disjointness and all
source/action/mask/provenance/normalization inputs. Header/CRC/JSON closure is
not full payload qualification.

The [prepared AD bindings](bindings.ad-first.prepared-20261031.json) contain the
exact59588 source snapshot and final62-root references, no teacher reference,
all admissions false, no resource receipt and no time allowance. The execution
owner must fill/hash a fresh actual code/host/projection/index/resource copy.

## Necessary allocated CPU software check

Meaningful regressions in `tests/test_robotwin_ad_first.py` compare every AD/BC
scientific array to the qualified paired path, then forbid teacher loading/
inference and remove the unused teacher file. They cover PPO/reset/SAC AD,
source-validation BC, missing teacher arguments, unchanged behavior/frame/mask
guards, final100 coverage/provenance/disjointness, default four fits, AD-only
two fits, early unreviewed-DPT rejection and unchanged AD funding/split guards.
The AD subset also rejects the full four-fit time allowance.

Recommended exporter-family checks, all inside the root's existing zero-GPU
isolated CPU qualification context:

```text
python -m pytest -q tests/test_robotwin_ad_first.py \
  tests/test_robotwin_history_corpus.py tests/test_robotwin_offline_compat_gate.py \
  tests/test_robotwin_finalbank_index_bindings.py --junitxml=<fresh-output>/ad-first.junit.xml
```

Root reviews/schedules one necessary<=600s0GPU/8CPU/64GiB software allocation,
not a new diagnostic pilot or reuse of closed R6 funds. If viable within its
single bound, separate owned steps may also qualify teacher candidate
`26dd63d1b11271e29e502f6367a8230bbd999c67`'s60 prepared fake/legacy cases and
the actual59588/20c CPU actor/private-RNG admission. Those DPT results stay
separate and cannot invalidate completed, clean AD-only qualification evidence
or become an AD teacher-review prerequisite. No software test, model call or
production payload scan ran on macOS or development SSH for this preparation.

## Exact command interface and resource subset

Existing command interface after exact code/inputs/admissions are bound:

```text
python <exact-code>/outputs/robotwin-finalbank-offline-plan-20261031-r1/run_offline.py \
  --bindings <owner-hashed-AD-copy> --bindings-sha256 <SHA256> \
  --objectives ad --phase export --execute
python <exact-code>/outputs/robotwin-finalbank-offline-plan-20261031-r1/run_offline.py \
  --bindings <owner-hashed-AD-fit-copy> --bindings-sha256 <SHA256> \
  --objectives ad --phase fit --execute
```

Without `--execute`, the same existing runner prints commands only. It describes
two exports (AD/BC, no `--teacher-*`) and two fixed500 fits (AD full/static).
Omitting `--objectives` preserves the paired/four-fit default. DPT-only is the
existing deferred DPT subset and still requires full source-only teacher review.

Keep the existing proposed final projection/export CPU ceiling7200 scheduler
seconds /6900s one outer deadline from before projection. Pass only remaining
time into the exporter; do not reset clocks between projection and partitions.
The root's final100 outer profile/admission remains its responsibility; R6's
fixed65/1580 profile is not repurposed as this6900s job. This change duplicates
no owner runtime wrapper.

For the AD two-fit subset, propose at most1800 scheduler GPU-seconds, parent
allowance<=1650, on the existing1GPU/8CPU/32GiB/shm8 recipe. Each process keeps
750s external/600s fitting limits. The runner enforces1650 for one objective;
the paired default stays3300 within the original3600 scheduler proposal.
AD+later DPT are subsets of that original four-fit proposal, not two new3600
grants. All actual charges and any later DPT reservation must be reconciled by
the owner. No GPU/physical/fit resources are admitted by these estimates.

Remaining blockers are the necessary allocated CPU pass, actual final62-root
byte projection/AD+BC index admission, and exact owner code/profile/resource
receipts. They are not an uncompleted source run or an unused mean panel.
The separate DPT teacher profile, paid24 reuse candidate, teacher competence
and DPT fit admissions remain pending with no false review flag.
