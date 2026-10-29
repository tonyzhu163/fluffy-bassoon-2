# LIBERO as the starting benchmark for transferable steering

The controller learns on LIBERO-40: Spatial, Object, Goal and LIBERO-10,
ten tasks each. The primary question is whether its use of earlier attempts
and rewards transfers to LIBERO-90 and to perturbations in Plus/PRO. The
motor and controller weights stay fixed during in-context evaluation.

## Split and exposure

Use `python -m fluffy_bassoon.libero_transfer --source-root <pinned-LIBERO>
--output <fresh-manifest.json>` with `PYTHONPATH=src`. The builder reads static
BDDL files and hashes source bytes. It requires the clean official LIBERO
revision `87edbd1b823e464bc66557d213cc45e9f88aa543`. It never loads a model,
simulator, or executable initial-state pickle.

Do not infer disjoint full tasks from different suite names. The manifest
groups full normalized goals plus referenced object bindings and normalized
instructions transitively across all 130 entries. All 40 source tasks remain
in training; equivalent LIBERO-90 tasks become overlap controls. Report the
strict remaining test tasks as primary, all 90 as a secondary table, and
overlap controls separately. This syntactic audit does not prove exhaustive
semantic separation; shared scenes and atomic skills remain allowed.

Source teachers, meta-gradients, normalization fitting and checkpoint selection
use LIBERO-40 only. On those tasks, initial states 0–39 supply training,
40–44 validation and 45–49 familiar-task evaluation. Confirm the state count
at simulator qualification. LIBERO-90 is locked for evaluation. Trials and
their windows inherit their parent partition.

The released RLinf LIBERO-130 motor has seen the original test tasks. The
first experiment therefore measures generalization of the steering controller.
A motor trained only on the source tasks is a separate later experiment.
Do not migrate the existing 78/10/10 split or active collection runs into this
new contract; their data cannot silently become a LIBERO-40 corpus.

## Plus and PRO

Their supplied base suites are Spatial, Object, Goal and LIBERO-10. Consequently,
training on all four clean suites and testing their variants measures
perturbation robustness on familiar base tasks. It does not establish unseen
base-task transfer. Keep final variant identities, severity levels and seeds
separate from development before collecting outcomes. Existing development
cases inspected in another campaign are development data here too.

Joint task-and-shift transfer needs a separate family holdout within the 40
or validated perturbations of strict LIBERO-90 tasks. Declare that auxiliary
experiment independently; do not call supplied Plus/PRO variants of the 40
unseen tasks. Seen-shift and held-out-shift results must also be separate.

## Order and comparison

1. Reuse the released motor and qualified inference interface. On a small
   predeclared LIBERO-40 development panel, verify reset reproducibility,
   action changes under noise steering, and useful native-versus-oracle
   headroom. Test outcomes must not select arms or training recipes.
2. Start with an existing GRU/Transformer memory controller and static
   multitask steering at the same observation, language, motor and action
   interface. Keep simple search/UCB/Thompson sampling controls for discrete
   arm selection. A discrete selector is not an exact reproduction of
   continuous DSRL.
3. Train on complete multi-attempt trials from source tasks. Use K=8 for the
   first contract; keep a task/condition fixed across attempts while resetting
   the physical scene. Save genuine failures and actual feedback. Validate
   only on source-task development scenes; freeze all choices before test.
4. Compare frozen-controller adaptation with an independently cloned online
   DSRL baseline per test trial. Match target interactions and report source
   training, optimizer, motor-query, wall-time and GPU costs separately.
5. Evaluate strict 90 and locked Plus/PRO panels after the controller is fixed.
   Report per-attempt success, cumulative return and failures before success,
   with paired task/scene trials and multiple independent training seeds.

Evidence for adaptation requires gains beyond retry sampling and beyond a
trained episode-reset controller. Include memory reset, reward shuffle and
wrong-task history, preserving legal timing and current observations. A
reward intervention tests reward use; adaptation from state-transition
evidence alone must be described separately. Freeze deployment weights and
reset transient memory between trials.

Better held-out success alone establishes generalization. A CTM-specific
claim additionally needs matched GRU/Transformer comparisons. These are
success criteria, not a requirement that every experiment produce a win.

## RoboTwin follows qualification

Use [the VLRLA setup contract](robotwin-vlrla.md) after the LIBERO pipeline
demonstrates useful adaptation. Environment preparation can proceed now.
GPU qualification and longer collections retain explicit resource budgets;
an existing campaign's allocation is not a new experiment's allowance.
