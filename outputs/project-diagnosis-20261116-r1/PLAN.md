# Project diagnosis

Objective: continue diagnosis and analysis of the frozen-VLA in-context RL
project overall, using the current repository and verified external artifacts.
The completed teacher/transformer campaign remains closed. Initial work uses
Baige development CPU only; it does not reopen the 230 GPU-hour budget.

## Questions to resolve

1. **Scientific claim and task design.** What persistent unknown could earlier
   interactions reveal? Do training histories and evaluation histories share
   that structure? Distinguish general task adaptation, repeated-scene retries,
   initial competence and information-seeking behavior.
2. **Teacher/data quality.** Quantify successful coverage, source improvement,
   reset identities, post-success actions, actual actor-sampling exposure and
   train/validation dependencies. Do not mistake recorded stream count for
   independent teacher fits or task diversity.
3. **Objectives and representations.** Compare the four implemented adaptations
   with their primary specifications. Test whether the current observation
   identifies the source task; examine whether Q benefits from actual actions
   and whether history changes predictive information. Outcome ranking is only
   a proxy for action advantage, especially after first-success reward is spent.
4. **Native behavior and evaluation validity.** Review same-case stage baselines,
   history/reset/donor controls, replay discrepancies, horizons, control cadence,
   OOD meaning, fit uncertainty and outcome-dependent choices. Offline loss or
   sensitivity alone cannot establish useful adaptation.
5. **Project evidence and decision.** Reconcile RoboCasa, RoboTwin, synthetic
   histories and LIBERO controls. Produce an evidence-backed overall diagnosis,
   distinguish implementation failures from negative scientific results, and
   identify the smallest decisive next experiment before recommending scale.

## First targeted checks

- Read hash-bound six-task source arrays: distinct native reset seeds within
  streams; success timing and fractions of queries/controls after first success.
- Reconstruct the exact seed-0, batch-8, 10,000-update actor sampler for three
  and six tasks. Report query coverage and teacher-version/suffix exposure.
- A fixed nearest-centroid source-task probe uses current initial-query features
  only, with normalization/centroids fitted on training streams and scored on
  all validation streams. This measures visible task information, not policy
  competence or a proof that history can never help.
- Reclassify the existing frozen-value diagnostic into before-success and
  post-success queries. Preserve its original results; annotate limits rather
  than retroactively change the panel.
- Follow with frozen-model action/history interventions if they resolve a
  specific remaining ambiguity. Preserve checkpoints and raw evidence.

All numerical computation runs on allocated Linux. Record exact source pins,
input hashes, output identities and zero/additional GPU usage. Numerical checks
belong to this run; no tests for plans or bookkeeping are added. Respect other
owners' work and leave active runs immutable.
