# What the steering controller learns across trials

The research target is robotic success and interaction efficiency with a
frozen pi0.5 motor. The meta-learning task is a repeated-attempt robot trial,
not predicting a task ID or imitating arbitrary noise in isolation.

## Trial and adaptation

- A trial fixes the instruction/goal, initial scene state, perturbation and
  reset seed for K complete attempts. Different trials vary those conditions.
- The controller receives current public VLA features and robot state, plus
  its earlier executed steering actions, rewards and episode boundaries.
  Case IDs, split membership and simulator metadata are audit-only.
- Its action is a bounded 32-dimensional steering vector, repeated across
  the motor's ten-step initial-noise tensor. The motor remains frozen.
- Full-history evaluation retains context across attempts. Episode-reset
  evaluation reuses the same trained weights but discards earlier episodes.
  The static control is trained separately on the same data and has no past
  action/outcome context at inference.
- Meta-controller weights stay frozen during deployment. Ordinary online
  DSRL updates actor/critic weights within its deployment interaction budget.
- Perturbations may already be identifiable from the current observation.
  A trial definition alone does not establish that memory is useful.

## Source histories and objectives

- Each source trial starts from the same fresh official DSRL SAC prior.
  SAC updates between complete attempts; actual actions and outcomes are
  recorded before those updates.
- AD predicts the chronological action actually executed at each query.
  Advancing source optimizer updates are verified, but improving robotic
  competence is not assumed from update counts alone.
- DPT-style supervision replaces only the query label with the final
  candidate actor's action. Historical actions and outcomes stay actual.
  The final actor is not called an expert without independent evaluation.
- Full and static models within an arm use identical trials, labels,
  initialization seeds and optimization schedules. Source-SAC pretraining
  consumes the same training-only raw replay; it adds no robot interactions.
- Report source collection, validation and deployment interactions
  separately. Reusing a dataset for several models does not multiply its
  unique collection cost. Adding Plus data does increase collection cost.

## Distribution claims

| Source corpus | Target | Interpretation |
| --- | --- | --- |
| Original LIBERO only | Plus or Pro variants absent from source | Transfer to benchmark shifts unseen by the controller |
| Original LIBERO plus declared Plus training cases | Disjoint Plus cases | Within-benchmark generalization; stronger OOD claims require specified perturbation-family holdouts |
| Either source arm; no Pro pretraining | Pro | Transfer to a benchmark absent from controller pretraining |
| Either arm | New scene/perturbation with the same goal | Condition generalization, not new-goal generalization |
| Diverse training goals with disjoint target goal families | Held-out goals and conditions | The stronger task-generalization experiment, once implemented and frozen |

These statements concern controller training. The released RLinf backbone
is treated as exposed to all 130 original LIBERO tasks. Controller holdouts
on original LIBERO do not establish unseen tasks for that backbone.

## Current bounded smoke test

The first corpus uses one cream-cheese-to-basket goal family, four original
training scenes, two original local validation scenes, and an optional six
Plus training trials. This checks collection, training and closed-loop
deployment. It cannot establish broad task-level meta-learning.

The clean source episodes all succeeded. That supplies weak failure/recovery
information. The declared Plus/Pro development cases also have native-policy
success ceilings. Good imitation loss or high success on those cases is
insufficient evidence of useful adaptation. A performance claim needs
informative held-out trials and comparisons with the native motor, trained
static controller, context lesions, simple search and source-matched DSRL.
