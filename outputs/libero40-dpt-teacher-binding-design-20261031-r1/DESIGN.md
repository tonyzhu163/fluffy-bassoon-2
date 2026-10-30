# Next immutable500 generation: explicit label-teacher binding

**Design only; no implementation, relabeling, tests or allocation this turn.**
Prefer one preselected final-published **TRAIN SAC actor per goal** to label
both TRAIN and matching heldout VAL observations for the fixed8 source goals.
Name that DPT metric **`heldout_state_same_train_teacher`**. Freeze the latest
complete source-only TRAIN publication before labeling/fitting/target evaluation;
do not choose teachers using target results, validation NLL or claimed quality.
Actors remain candidates, not experts.

Keep original VAL behavior histories, frames, noises, motor actions, outcomes,
rewards and boundaries verbatim. Preserve the legacy100 generation and its
independent-VAL-teacher labels/metrics as **`teacher_transfer`**. Never compare
NLL across teacher/dataset generations as if its target distribution were fixed.
If no verified same-teacher corpus is ready, fixed500 fits may retain explicitly
reported `teacher_transfer`. Neither collection nor fixed500 checkpoint
selection is blocked or changed by this proposal.

The contract separates **behavior stream** from **label teacher**. A behavior
stream is an actual learning trajectory, potentially warmup plus many policy
versions; its final checkpoint is not the actor that generated every action.
Per-row `teacher_steps` therefore remains the actual behavior clock. The fixed
TRAIN label teacher has its own publication/update identity and full checkpoint
SHA. Prefer a separate pure-actor digest over
`serialization.to_state_dict(learner.agent._actor.params)`, using existing
`vla_libero.structure_digest`; include the image encoder and action head,
exclude optimizer/critic/replay/RNG/counters. Verify that digest on allocated
resources and before/after label inference. A full checkpoint hash alone is
not a pure-actor hash. No such digest or new corpus is claimed verified now.

## Existing data and code to reuse

- `vla_libero.py:103–106,439–450`: recorded `sac_observation` is public pixel64
  plus8D proprioception; rows also retain exact executed noise/actions and
  outcomes. No images/features need VLA recomputation.
- `vla_source40_corpus.py:485–494`: complete episode gzip/pickle files and
  receipt SHAs preserve row ordering. `:218–239` already applies deterministic
  `agent.eval_actions` to these stored observations and records label provenance.
- `vla_libero.py:314–327,389–417`: existing pinned `OfficialSAC` construction,
  full-state save/restore; `:645–663` supplies the canonical tree digest.
  `vla_source40_subset_evaluate.py:105–125` shows hash-bound source-only teacher
  loading. Use only this small SAC actor; construct no frozen VLA motor.
- `vla_source40_corpus.py:151–159` and `vla_history.py:69–91`: existing array
  assembly/export already isolates `target_noise` from the actual history.
- `vla_source40_topup_assembly.py:138–170` freezes original publications.
  `vla_source40_training.py:189–206` currently assumes label teacher equals
  the local behavior-stream publication. A new explicit derived-label admission
  must separate those roles; merely renaming metadata would be incorrect.

## Minimal future implementation

1. Start from the verified all40/fixed8 immutable snapshot and its selected
   TRAIN teacher catalog. Keep every origin index/sidecar/checkpoint hash.
2. On an allocated qualified runtime, load the8 matched TRAIN actors with the
   existing `OfficialSAC` path. Derive/check pure-actor identities and reuse
   TRAIN DPT labels only when their recorded teacher checkpoint matches the
   preselected binding. Other32 TRAIN histories remain unchanged.
3. Read only the selected complete VAL episode traces in receipt order. Apply
   the goal's fixed TRAIN actor `eval_actions` to recorded SAC observations.
   Write fresh DPT VAL labels; no SGD/replay insertion, simulator actions,
   policy sampling or VLA inference. Preserve every non-target array exactly.
4. Export fresh derived VAL DPT files/sidecars plus `binding-schema.json`'s
   proposed binding record. Add a thin explicit derived-label admission branch
   to the existing assembly loader, checking origin/non-target hashes and
   label-teacher bindings. Keep legacy admission strict by default; do not
   fabricate source publications or replace behavior provenance with TRAIN
   teacher provenance. A changed assembly gets a fresh generation/hash.
5. Existing trainers consume that generation with unchanged architecture,
   seeds, fixed validation windows and25/100/500 selection. Reporter records
   generation, metric ID and teacher binding. Legacy metrics remain separately
   identified; no mandatory reevaluation of the old generation is proposed.

`binding-schema.json` is a field/semantic-contract proposal, not a materialized
binding, executable validator or new framework. Verification is required only
before claiming the new metric. Even a common label teacher leaves different
behavior-induced observations/histories and heldout states; NLL remains an
imitation measure rather than robot-performance or memory evidence.

## Bound and accounting

Propose at most **600 cumulative GPU-seconds** for postcollection label prep,
including setup, loading, inference, writing and failed attempts. This comes
from the existing25h source contingency:84,480 collection +1,400 source checks
+3,000 fits +600 labels =**89,480s**, leaving**520s** under90,000s. No new action
or optimizer budget, no extra VLA inference and no storage-request increase.
The separate4.5h target component is unchanged; the combined29.5h/150GB request
remains pending. Implementation/inference belongs only on allocated resources.

Retain the TRAIN teacher's entire source training cost and all8 VAL behavior
generation costs once via existing unique stream/cost-group IDs. Do not add
the same teacher cost again because it labels two splits; do not erase VAL
behavior cost because another teacher supplies targets. The source1M action
ceiling and e93 production plan remain unchanged. At timeout preserve partial
derived files, retain the legacy `teacher_transfer` generation, and proceed
without a silent same-actor claim or an additional collection gate.
