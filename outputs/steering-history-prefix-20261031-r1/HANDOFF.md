# Transformer/data-admission handoff

Status: code and focused tests prepared; macOS AST/whitespace checks only.
No allocated tests, models, GPU, simulator actions or jobs were run by this chat.
This does not qualify the motor, teacher, student or scientific result.

One shared phase remains 28,800 GPU-s / 300,000 all-in actions / 64 GiB,
owned by `Plan VLA in-context learning / finish_assembly`. No allocation is
requested or created by this code. The prepared fit slice is four times 360 s,
1440 GPU-s total, subordinate to the shared parent clock. All readout actions
use the exact `student_evaluation` bucket (105,000), with no per-child grant.

## Code/API ownership

- New `meta_rl/steering_history.py`: canonical-source projection, independent
  AD/DPT admission, fitting command emission, completed-candidate admission,
  frozen history/controller adapter and per-K query NLL.
- Existing `meta_rl/vla_history.py`: optional complete-episode sampler windows.
  Defaults remain unchanged. At most nine complete episodes give at most eight
  earlier support episodes; unseen pre-window feedback is cleared.
- Existing `meta_rl/vla_history_train.py`: binds the schedule in resumes and adds
  `--max-episodes-per-window` / `--checkpoint-selection final`. No new learner.
- `tests/test_steering_history.py`, plus existing `tests/test_vla_history.py`:
  mandatory allocated Linux qualification before real fitting/deployment.
- Method contract: `docs/steering-history-prefix.md`.

## Live integration bindings

Read once: coordinator `outputs/frozen-pi05-endtoend-20261031-r1/{SCOPE.json,INTERFACES.md}`.
Use actual selected short arm `noise|kv|action`; native mapping/H32/five steps,
motor, normalization, feature and basis hashes must come from the qualified live
driver. `control_interface_sha256` is the DRIVER digest, not a hash of the
student's derived feature view. Coefficient bound is exactly 1.0. Native teacher
features remain 2062; student's observation is 2076.

`CoefficientSteering.student_observation(native_features)` is the sole online
projection/state owner. `DriverCoefficientStudent(history,control=control,
condition_id=...)` exposes the exact existing driver's `act`, `decision_metadata`,
and eight-argument `observe` callbacks. It does not create another simulator loop
or command-history state. The canonical typed `student_previous_agent_command`
is verified offline against the last valid pre-condition command. Query restore
must be followed by the existing physical reset; that driver resets the command
suffix while preserving only the selected policy memory snapshot.

The seven `steering_*`/update metadata keys were relayed to the driver/collector.
Actual actor descriptors may live in decision metadata; pure actor tensor hashes
are verified separately from full recovery file hashes. External immutable
actor archives are accepted only with their exact file SHA, kind and tensor hash.
No RNG seed is used as a coefficient/KV target.

## Prefix envelope

`PROFILE.prepared.json` deliberately contains null runtime bindings and cannot
be admitted as-is. Bind it ONCE to the real source profile, before opening query
cases. Six exact keys, with distinct seeds from `lineage_specs`, are:

```
sac_calibration_minus_train0
sac_calibration_plus_train0
sac_calibration_minus_train1
sac_calibration_plus_train1
sac_calibration_minus_validation2
sac_calibration_plus_validation2
```

Predeclared label teachers are TRAIN1 (minus/plus), matching the collector's
`teacher_label_selection`. One exact teacher labels BOTH partitions. A qualified
snapshot is the exact final24 actor qualified by A2. First scientific fit uses the common
24-episode immutable boundary; 12 remains pending data. 24 is the initial
collection target and 48 remains only the cap. No whole-lineage filtering.

The source prefix JSON has `format=frozen_vla_4d_source_prefix_v1`, `completed=true`,
`source_only=true`, `profile_sha256=content_hash(bound_profile)`, `source_task`,
the exact `interface`, `control_qualification` (absolute path/SHA), `lineages`, and
objective-specific `learning_qualifications` / `teacher_qualifications` maps.
Each lineage contains:

```
lineage_id, condition_id, partition, seed, source_only=true,
corpus_root, completed_episodes, initial_updates, final_updates,
initial_recovery={path,sha256}, final_recovery={path,sha256},
episodes=[{case_id,actual_seed,initial_observation_sha256,
           raw_trace={path,sha256}}, ...]
```

`raw_trace` is the existing canonical episode MANIFEST, not a new trace log.
Use its reset's `initial_observation.content_sha256`, actual reset seed and
physical case identity. Wrapped `source_coefficient_teacher_v1` recoveries are
accepted only at complete update boundaries and verified against their actual
source profile, mapping, motor, lineage and learner state. Both partitions must
contain full actual learning lineages for AD; DPT can retain poor behavior.
All original source arrays/raw/recoveries remain immutable.

Qualification maps reference JSON receipts bound to existing predeclared physical
A2/teacher-validation `evidence` files and fixed `paired_cases` outcome panels.
Each receipt binds `condition_id`, driver `interface_sha256`, and outcome rows
`{case_id,baseline_success,teacher_success}`. AD also binds exact initial/final
source recoveries and checks the predeclared improvement threshold. DPT binds
`teacher_kind=noise_sac`, `teacher_lineage_id`, full `teacher_recovery` and
`actor_parameter_sha256`, checking competence separately. Alternatively an actual
same-mapping `constant_4d_oracle` references a bounded coefficient/provenance JSON;
its physical proof must bind the exact mapping and executed four coefficients.
An external 14D inverse is not admissible. No new qualification panel/job is
requested; the prepared small-panel thresholds must be bound to the existing
predeclared pilot evidence and cannot establish robust competence.

The independent VAL learner supplies imperfect contexts, not a different teacher
for DPT labels. Labels use private fixed actor RNG, never source interaction RNG.
Only query targets change; history remains actual. AD and DPT exports/fits are
independent, so failed AD learning does not block qualified DPT oracle fitting.

## Fitting/readout

`COMMANDS.txt` supplies the exact entry points. Export each admitted objective
to a fresh directory, then emit its full/static argument vectors. The sole owner
runs those vectors under the ONE existing shared clock and GPU probe. Four
matched fits: seed0, width128/layers2/heads4, memory256, at most9 complete episodes
per window, batch8, 2500 updates, eval every250 with16 VAL batches, 360 s maximum per fit. If runtime
capacity does not reach the same 2500-update endpoint, preserve `latest.pt` and
report incomplete; no partial candidate is accepted for readout.

`load_student` admits only the completed exact pinned fit. `evaluation_plan`
predeclares 88 fresh queries plus 56 charged support episodes, 144 total. With
both task horizons600 and zero reset policy actions:86,400 actions. Actual native
source/donor horizons and reset bounds determine the exact cap; it must fit the
105,000 `student_evaluation` slice BEFORE queries open. No K8 substitution.

One 4-support prefix per full model/case is collected completely before querying.
Clone empty/K1/K2/K4 policy memory/RNG and run separate physical-reset queries;
query feedback never reaches later supports. K4 includes reset, reward blind,
consistent reward shuffle, wrong-condition, trained static and stationary retry/native
controls. Cross-task transfer uses two fresh donor supports per model/condition
and K2 queries with matched calibration. One fixed source-VAL-selected blind
policy supplies stationary retries for both conditions. Store exact case/model snapshot
bindings and paired seeds. The independent 32-case confirmation remains unopened
and outstanding for reliability/generalization claims. Report per-K query NLL,
K0 mixed-label ambiguity, actual queried outcomes and all physical costs.

## Qualification dispatch

Required on an allocated Linux CPU worker with actual zero visible GPUs:

```
python -m pytest -q tests/test_steering_history.py tests/test_vla_history.py --junitxml=<receipt-dir>/junit.xml
```

This includes real tiny CPU optimizer tests of the existing learners, raw corpus
roundtrips and model export/admission. Suggested bounded request: 300 CPU-worker
seconds WITHIN the phase's fresh 1,800 CPU-worker-second allowance, not a new grant.
Only the existing resource owner may dispatch and settle it. Retain actual source
commit, allocation/import/CUDA attestation, terminal result and elapsed cost.
No runtime PASS is claimed by this handoff.
