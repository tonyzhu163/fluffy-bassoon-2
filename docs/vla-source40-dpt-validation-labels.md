# DPT validation labels from the matching TRAIN teacher

The original source corpus remains immutable. Its validation behavior was
collected by independent validation SAC learners; its original DPT labels come
from those validation learners' final actors. Existing assemblies retain that
`legacy_teacher_transfer` interpretation. The new metric is an explicit derived
generation, not a new name for the old labels.

The opt-in assembly path accepts a completed, hash-bound
`source40_dpt_same_train_teacher_labels_v1` binding from
`vla_source40_dpt_same_teacher`. For each of the eight validation goals, the
writer binds the derived labels to the matching selected TRAIN checkpoint and
its pure actor digest. The assembly adapter rechecks this against the original
all40/fixed8 admission, the selected TRAIN DPT publication, and both source
indices. It verifies the original and derived validation arrays have identical
dtype, shape and bytes for every field except `target_noise`.

Only the combined **validation DPT target array** changes. All 40 TRAIN AD/DPT
histories, validation AD targets, source-history and episode order, rewards,
executed noise, observations, boundaries and behavior `teacher_steps` remain
unchanged. Each derived sidecar retains every original source metadata value;
the label teacher has a separate identity. Its metric is
`heldout_state_same_train_teacher` only after this admission succeeds.

The original validation learner's full physical collection cost remains a
behavior-stream cost. The matching TRAIN teacher cost is already present once
in the original training catalog. Derivation adds no simulator interactions and
does not count the TRAIN teacher again. Assembly costs and source indices stay
identical to the original corpus generation.

After the separate writer produces a complete fresh binding, use a **new**
assembly output directory:

```bash
python -m fluffy_bassoon.meta_rl.vla_source40_topup_assembly assemble \
  --admission /absolute/snapshot/admission.json \
  --admission-sha256 ORIGINAL_ADMISSION_FILE_SHA256 \
  --dpt-validation-binding /absolute/derived-generation/binding.json \
  --dpt-validation-binding-sha256 DERIVED_BINDING_FILE_SHA256 \
  --output /absolute/fresh-same-train-teacher-assembly
```

The resulting `topup-inputs.json` can use the existing `train` entrypoint with
its explicit file hash. It retains the fixed 25/100/500 update schedule and
final-update selection. The adapter changes no optimizer, model, source task,
checkpoint choice or robot evaluation contract. Existing commands without the
new binding arguments keep strict legacy admission and do not import the
derived-label writer.

Allocated Linux validation targets are
`tests/test_vla_source40_dpt_label_admission.py`,
`tests/test_vla_source40_topup_assembly.py`, and
`tests/test_vla_source40_training.py`, plus the writer's independent tests.
These adapter tests use a metadata-only writer test double and construct no
teacher, VLA or student model. Authoring/static checks on macOS do not establish
that these tests passed.
