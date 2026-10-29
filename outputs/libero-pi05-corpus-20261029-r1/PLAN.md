# Frozen π₀.₅ LIBERO corpus split

Freeze [split.json](split.json) before π₀.₅ teacher collection. It contains
**78 training tasks, 10 validation tasks, the original 10 LIBERO-10 test tasks,
and 2 excluded LIBERO-90 tasks reserved as an auxiliary test**. Task 57,
cream cheese → tray, is in training and can qualify the technical pipeline.
This is a split specification, not evidence of successful robot learning.

The pinned [official LIBERO source](https://github.com/Lifelong-Robot-Learning/LIBERO/tree/87edbd1b823e464bc66557d213cc45e9f88aa543)
defines LIBERO-90 for pretraining and LIBERO-10 for downstream testing. We use
that split as the starting point, then remove semantic duplicates across it.
Two tasks belong to one family if their full normalized BDDL goal predicates
and referenced object bindings match, or their full normalized instructions
match. Equivalence is transitive. All family members share a partition.
Goals retain object identity, relation, region and direction; conjunction order
is normalized. Initial scene layout and atomic subgoals may overlap.

The audit excludes these LIBERO-90 tasks from every source-teacher and
meta-training stage:

| LIBERO-90 ID | Reason | Official test counterpart |
| --- | --- | --- |
| 38: right moka pot → stove | Same full goal `(On moka_pot_1 stove_cook_region) ∧ (Turnon stove)`, despite different instructions and initial stove state | LIBERO-10 ID 2: turn on stove and put moka pot on it |
| 77: book → back caddy compartment | Same full goal, bindings and instruction in another scene | LIBERO-10 ID 5: book → back caddy compartment |

No full-goal, instruction or family equivalence crosses the resulting
train/validation versus test boundary. Nine physical scene families remain
shared. The main claim is transfer to a held-out full task/goal composition;
shared scenes and familiar atomic skills are permitted. A new scene or new
primitive claim would require a different split. We make no claim that test
tasks were absent from the released VLA's pretraining.

The remaining 88 LIBERO-90 tasks are divided into 78 training and 10 validation
tasks using seed 0 and whole semantic families. The reproducible builder
shuffles sorted family IDs and chooses the first reachable whole-family subset
of ten validation tasks. It does not use model performance. The manifest
records every canonical task ID/name, full goal, family, BDDL SHA-256 and
initial-state file SHA-256. Its content hash excludes only its own hash field.

All 100 `.pruned_init` files statically encode 50 initial states. For training
tasks, states **0–39** supply source DSRL training and complete history trials,
**40–44** supply auxiliary familiar-task validation, and **45–49** supply the
auxiliary familiar-task test. The primary validation uses the ten validation
tasks: any validation-only teacher generation stays isolated on states 0–39;
own-history student rollouts on states 40–44 guide selection. Validation states
45–49 remain reserved. The official ten held-out tasks use all 50 states for
final evaluation, with any smaller pilot subset declared before rollouts.

Exclude every official test task and both auxiliary excluded tasks from
**all source DSRL teacher collection and updates, meta-gradients, feature
normalization fitting, distillation labels, curriculum and checkpoint
selection**. Validation-only teachers and labels cannot update a training
teacher or student. A separate ordinary DSRL online comparator may adapt during
held-out evaluation using its declared K-attempt budget; those interactions and
weights never return to pretraining or model selection. A frozen student may
use only its own earlier observations, actions, rewards and episode boundaries
within the current evaluation trial.

Keep each multi-episode trial intact. Every sampled training window inherits
its parent trial's partition. Do not randomly split sliding windows, episodes
from the same adaptation trial, or descendants of the same base-task family.
Use one released π₀.₅ motor receipt and one fixed shared feature contract;
per-task source learners must not silently introduce changing feature spaces.

**LIBERO-Plus and LIBERO-Pro are locked future tests.** Record each variant's
canonical base task, family and perturbation type before collection. All
descendants inherit the base family's partition. Perturbations of a training
base task measure robustness; variants of held-out bases additionally measure
task transfer. Final Plus/Pro feedback cannot choose teachers, curriculum,
features, checkpoints or hyperparameters. Any development variants need a
separately declared development partition before testing.

To reproduce from a checkout of the pinned source, run the static builder:

```bash
python3 build_split.py --source-root /path/to/pinned/LIBERO --output /fresh/output
```

The builder uses the task map, BDDL text and pickle shape opcodes only. It
does not import Torch, NumPy, LIBERO or a simulator, deserialize executable
pickle objects, compute model outputs, or launch jobs.
