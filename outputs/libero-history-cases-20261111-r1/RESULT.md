# Procedural LIBERO history cases

Created November 11, 2026 from main `43d0400fdff346c8d440fd2426c359f3392c85b6`.
This is a completed case-authoring prototype, not a simulation run or a learned
history result. No GPU was allocated; no motor or learner was run.

## Decision

Use LIBERO's scene/object/predicate machinery as the substrate and generate our
own history-learning sessions. LIBERO-Gen's released data and collection code
remain useful starting points, but the released fixed layouts are not the limit
of what LIBERO can express. The [official task-generation interface](https://lifelong-robot-learning.github.io/LIBERO/html/procedural_generation/task_generation.html)
supports programmatic object counts, initial regions and goal predicates.
The [LIBERO-Gen pipeline](https://github.com/real-stanford/behavior_prompting/blob/58125e682acceefef3d60c071212dc483a4db1c9/docs/liberogen.md)
also includes automatic demonstration collection. Its collector depends on task
metadata and existing replay segments, so arbitrary new scenes still need an
integration check.

The [custom-layout motor study](../libero-gen-custom-layouts-20261111-r1/RESULT.md)
found 0/8 successes on moved layouts and 2/2 in the paired native replay control.
That is a motor-transfer problem in the tested conditions, not a limitation of
the task language and not evidence against learning from history.

## What was generated

[histories.py](../../exploration/libero-scenes/histories.py) connects the existing
scene generator to the existing four-axis rule family. One session contains an
ambiguous demonstration, twelve distinct attempt scenes and one distinct query
scene. All fourteen share the same hidden rule and generic instruction.

| Split | Candidate sessions | Layouts | Available rule compositions |
| --- | ---: | ---: | ---: |
| Train | 1,024 | 192 | 2,184 |
| Validation | 256 | 32 | 364 |
| Test | 256 | 32 | 364 |
| Total | 1,536 | 256 | 2,912 |

The available rules are an inventory; the sampled sessions do not cover every
rule. Each scene supports ten nominal bowl/target behaviours. The four rule
axes are pick selector, conditional pick switch, anchor selector and conditional
anchor switch. No new offset or multi-step motor behaviour was introduced.

Scene seeds and rule compositions are disjoint across splits. Every primitive
axis choice occurs in the training rule pool. No two rules had identical
behaviour vectors across this nominal scene bank. This finite-bank observation
does not prove global non-equivalence.

Every candidate leaves at least three different continuations after the demo
and at least two possible query answers. On the test split, a median of 448 rules
remain after the demo within the full 2,912-rule family. The correct behaviours on the
twelve attempt scenes determine the query within the full family, without knowledge of split membership. Full corrections
are an information ceiling; binary feedback and imperfect execution may reveal
less. Rule/layout proposals are sampled uniformly before these geometry-based
constraints, so the accepted bank is conditional on them. Rejection counts are
in the [summary](cases-r2/summary.json).

## Evidence and reproduction

The bank is retained locally in `cases-r2/` (6.3 MB on disk), outside Git. Small
tracked evidence includes the [summary](cases-r2/summary.json), a complete
[example case](EXAMPLE.json), [static check](STATIC-CHECK.json), and
[file hashes](evidence/artifact-hashes.json). [REPRODUCE.sh](REPRODUCE.sh)
regenerates the scene and session files into a new directory using `uv`.

Authoring used Python 3.9 and NumPy 2.0.2 on macOS. Syntax compilation and static
artifact checks covered split disjointness, unique layouts within each session,
schedule field allowlists, matching schedule/evaluator records, referenced goal
membership and BDDL hashes. These do not qualify simulator or controller APIs.
No permanent bookkeeping tests were added.

## What is still needed for an experiment

1. Load a small deterministic development subset on allocated Baige resources.
   Check placement, settling, camera visibility and native predicate behaviour.
   The existing `admit.py` has not yet been executed on these generated scenes.
2. Bind rule labels to the actual settled reset and repeat the ambiguity checks.
   These candidates use requested table-frame coordinates. Sampling within
   initial regions can change near ties; table X/Y is not camera left/right.
3. Collect successful demonstrations and real attempted trajectories. Reuse the
   upstream scripted/planned collector where its assumptions hold; measure motor
   coverage separately. Preserve native failures, and do not select evaluation
   cases by policy success. A reliable new motor may require training on generated
   development scenes before it is frozen for the history experiment.
4. Fit the history learner on the training split, then compare the same frozen
   learner with own history, cleared history and matched donor history on held-out
   cases. Separately mask/shuffle outcomes on matched histories. Record query
   success by attempt, not just cumulative success from extra tries.

`schedule/` is orchestration metadata, not a student observation. `evaluator/`
contains hidden labels and must remain private to collection/scoring. The BDDL
placeholder goal must not decide stopping; use the same outcome-independent
horizon and padding across hidden-rule alternatives. Real rewards come only
from native predicates evaluated after executed actions, never from matching
intended symbolic behaviours.

This bank jointly holds out layouts and compositions and does not isolate those
transfer factors. It covers one family of bowl placement tasks, not unseen task
families. It is ready for simulator qualification, not for reporting robot
adaptation accuracy. The separately owned π0.5 motor check is unchanged.
