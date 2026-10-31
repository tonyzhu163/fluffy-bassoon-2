# DPT validation uses independently trained teachers

Verified2026-10-31 from the existing immutable CPU snapshot, without raw-bank
scans, checkpoint unpickling, model construction, forward passes or tests.
No labels, selected checkpoints, active runs or pretraining data were changed.

**Yes: DPT TRAIN targets come from each TRAIN stream's final published SAC
actor; matching-goal VAL targets come from a separately trained VAL stream's
final published actor.** All8 matching pairs have different checkpoint
identities and different learner seeds:16 distinct full checkpoints/16 seeds.
The same frozen motor/features and disjoint nominal-state IDs remain verified.
All declared index, sidecar and publication identities matched the admission.

| Goal | TRAIN updates | VAL updates | TRAIN checkpoint | VAL checkpoint |
|---|---:|---:|---|---|
|libero_10:00|6620|11400|`f66bf972af0b`|`df56e22506e9`|
|libero_10:01|7120|12040|`27548ba19bfa`|`6b4eff0ae441`|
|libero_goal:00|5620|10780|`19eade9435fd`|`30cd4b4e6378`|
|libero_goal:01|5300|9780|`dcc08246a63d`|`0831580f1c45`|
|libero_object:00|29940|10740|`1f288d06e87b`|`7e9581b21792`|
|libero_object:01|5900|10280|`82af2ec8f704`|`6d8e0488aa79`|
|libero_spatial:00|28620|10500|`b95e004854ae`|`a8221cc5d9c4`|
|libero_spatial:01|5340|10200|`885d8af2ced2`|`49e27cc45a61`|

The full hashes, seeds, sidecar paths/hashes, teacher clocks and actual nominal
state IDs for all8 pairs are in `receipt.json`. Hashes above are shortened.
For example, Spatial00 uses28,620 TRAIN updates versus10,500 VAL; Object00
uses29,940 versus10,740. Other six TRAIN clocks are5,300–7,120 while their
independent VAL clocks are9,780–12,040. Thus teacher identity and optimization
clock are not held constant across this split.

Source-line evidence, reviewed code `01a47e09e4f00881f2bc9c5b02ced59508940258`:

- `src/fluffy_bassoon/meta_rl/vla_source40_corpus.py:218`: labels are the
  current stream learner's deterministic `agent.eval_actions` evaluated on
  every historical `sac_observation`. `:230–240` records that stream's source
  contract/partition, update clock and `source_actor_checkpoint_sha256`.
- The same file `:331–346` derives learner seed using **task and partition**,
  and explicitly declares independent validation SAC/replay. `:360–368`
  constructs/restores that stream's learner. No TRAIN-teacher relabeling of
  VAL observations occurs in this path.
- `src/fluffy_bassoon/meta_rl/vla_source40_topup_assembly.py:138–168` copies
  the selected immutable publication and its original producer sidecars;
  it does not replace a validation teacher with the training actor.
- `src/fluffy_bassoon/meta_rl/vla_source40_training.py:189–206` checks each
  partition's label-checkpoint provenance; AD/DPT retain identical actual
  histories and differ only in labels. It does not require a common teacher
  identity between TRAIN and VAL.
- `src/fluffy_bassoon/meta_rl/vla_history_train.py:74–82` recreates validation
  RNG734001 each time. Interim settings are4 batches×8 random causal windows,
  maximum128 positions (`vla_source40_topup_interim_fit.py:20–22`). The same
  frozen validation windows and teacher labels are reused at25 and100.
  Therefore changing validation samples/teachers during fitting does not
  explain the observed NLL increase.

This is **cross-teacher plus heldout-state imitation NLL**, rather than solely
heldout-state imitation of the same target policy. Different successful
steering policies can be valid robot solutions; increasing disagreement with
another teacher need not imply worse robotic control. This is a substantiated
alternative interpretation to overfitting, not a demonstrated cause of the
25→100 NLL increase. Source actors remain candidates; no expert/optimality or
individual actor competence claim follows from their update counts.

Important limit: the field named `source_actor_checkpoint_sha256` identifies a
**full learner checkpoint**, including actor/critic states, replay and RNG
(`src/fluffy_bassoon/meta_rl/vla_libero.py:389–397`). It is not a pure actor
parameter/function hash. Distinct independently trained teacher checkpoints
are established; the size or even existence of function disagreement on common
observations was not measured here. No counterfactual labels were generated.

Keep fixed-update exports and the existing robot readout. This evidence changes
how validation NLL is described; it supplies no selection change, relabeling,
new diagnostic run or source-scale approval prerequisite. The separate25h
source proposal remains pending and unchanged.
