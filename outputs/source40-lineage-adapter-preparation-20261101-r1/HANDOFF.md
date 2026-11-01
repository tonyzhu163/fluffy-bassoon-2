# Source-only lineage adapter: code preparation

Status: **unqualified source**, no execution or resource authorization. No tests,
models, arrays, simulator, SSH, scheduler or resource work was executed here.
Static AST parsing and Git whitespace checks only. The pending 25-hour source
proposal remains unfunded and cannot be funded by splitting it into short jobs.

New files only: `src/fluffy_bassoon/meta_rl/vla_source40_lineage_adapter.py`,
`tests/test_vla_source40_lineage_adapter.py`, and this preparation directory.
Reviewed base: `f16820e3b34e268cd22c3bbb0aaee9cc7eedf890`; its advance from the
initial `8e9c06a8` base changed none of the adapter's existing dependencies.
Active b665/task44 code and every existing collector/trainer remain unchanged.

## Behavior

- Calls existing `vla_source40_training.load_source` with its original source40,
  motor, complete-case, state, physical-budget and paired-prefix checks intact.
  Keeps ALL40 TRAIN and existing ALL40/fixed8 VAL coverage; no small-pilot path.
- Keys whole trials by original independent `history_id`, allowing several
  seeds for one task/partition without concatenating overlapping prefixes.
  Preserves cross-partition case/state/vector-hash and lineage checks.
- Requires official OpenPI40 READY receipt `ec064c…d170`, entire motor
  `c565dd…9e9`, H10/tied32/fixed shared prefix4104 and the original feature
  pooling/runtime contract. Motus4D and RLinf130 cannot enter this path.
- Checks complete original raw episodes, actual requested noise and tiled
  expansion, actions/rewards/durations, reset identity, next-observation chain,
  replay order and distinct pre-action/per-step/post-episode update clocks.
  Uses actual `replay>500` and `chunks*20`; never synthesizes missing clocks.
- Compact episode actor/RNG identities must already be bound by the original
  raw episode payload. Gaussian warmup explicitly marks the actor unused and
  binds its real sampler configuration/RNG state. Missing historical identity
  is ineligible; it cannot be reconstructed from a desired update count.
- Requires the existing selected final full recovery and verifies its actual
  optimizer/train-state structure, complete replay contents/count, clock and
  sampler/global/private/motor RNG against the original history. Per-episode
  full snapshots are optional retained evidence, **not a required cadence**.
- AD admission separately binds an empirical improvement assessment with
  prospectively declared scope/panel/threshold, including an explicitly chosen
  zero threshold if the parent selects it. Assessment never filters ALL40 data.
  AD uses actual noise targets and does not instantiate a DPT teacher.
- DPT independently checks the selected TRAIN teacher's declared competence.
  A competent ceiling teacher with zero improvement is allowed. The exact
  qualified deterministic TRAIN actor labels both TRAIN and VAL; first-TRAIN
  label replay uses the existing `1e-5` backend tolerance. No VAL label actor.
- Writes only the existing seven-array NPZ format. Metadata and raw sidecars
  retain audit IDs; no task/condition/teacher/update identifiers become student
  inputs. Existing trainer fits/fixes normalization on TRAIN observations only;
  identical sorted TRAIN observations are shared across objectives and modes.
- Target44, screen/confirmation, task57 DEV, Plus/Pro and other target-domain
  records remain excluded, including source wrappers over original DEV rows.

## Late-bound ABI

The module docstring is the precise input-key reference. The sibling profile
in `outputs/libero-lineage-source-profile-20261101-r1/` is a requirements source,
not an executed declaration or a grant. It was read without modification.

Public future call:

```python
assemble(declaration_ref, bindings, output,
         objective="ad" or "dpt", dsrl_root=UNBOUND,
         expected_hostname=UNBOUND, source_commit=UNBOUND,
         ad_improvement_ref=UNBOUND)  # required for AD; no DPT requirement
```

`REF={path: absolute_original_path, sha256: actual_byte_sha256}`.
Declaration format: `source40_independent_lineage_declaration_v1`; keys are
`source_only`, `source_data_from_targets`, `source_training_task_keys`,
`source_validation_task_keys`, `motor`, `feature_contract`, `noise_contract`,
`teacher_gate`, `ad_improvement_gate`, `locked_query_cases`, `lineages`.
`teacher_gate` can be null for AD; `ad_improvement_gate` can be null for DPT.

Each lineage: `lineage_id`, `task_key`, `condition_id`, `partition`,
`learner_seed`, `cases`, `label_teacher_lineage_id` (null permitted for AD).
Original contract identities/cases must match exactly. Ordinary unconditioned
source uses null `condition_id`; no condition assignment is selected here.

Each binding: `lineage_id`, `index`, `motor_before`, `motor_after`, `chronology`,
`teacher_qualification` (null except selected DPT TRAIN teachers).

Chronology references original episode-ordered records with `raw_trace`,
`behavior_identity`, `actor_after_parameter_sha256`, `rng_before`, `rng_after`,
and optional `before_recovery`/`after_recovery`. The raw payload itself must
originally bind `behavior_identity`, `actor_after_parameter_sha256`,
`rng_before_ref`, `rng_after_ref`. Compact RNG pickles carry `sampler`,
`agent_rng`, `numpy_rng`, `python_rng`, and encoded `motor_rng`.

Concrete lineage/task-condition/reset-bank selections, all physical/runtime
receipts, checkpoint paths, execution commit/host/model root, resource limits
and grant stay UNBOUND. No source collection producer was added or changed.
The old collector does not emit the new compact identities: it cannot become
eligible through this adapter alone. Parent must review any future producer
work and execution independently.

## Prepared regression source and limits

32 synthetic cases cover ALL40 scope, multiple independent lineages, incompatible
motors/features, DEV role laundering, locked queries, TRAIN-only teachers,
raw chronological transitions, real warmup threshold crossing, distinct SGD
clocks, unchanged six non-target arrays, ceiling competence, separate AD
assessment, backend label drift, compact actor/RNG continuity, final replay
contamination, missing original identity and incomplete full recovery state.

The ALL40 assembly wiring fixture explicitly stubs source admission, compact
numerical decoding and teacher inference; it is not motor/source qualification.
Other focused cases exercise the real existing source loader/array writer and
adapter projection using synthetic raw episodes. No case has run. `STATIC.json`
records preparation hashes, **not** a qualification receipt.

Parent retains latest-main review/publication and all science/resource decisions;
finish_assembly remains sole runtime owner. This prep does not repair or gate
the prioritized SAC recovery work or the phone TRAIN audit.
