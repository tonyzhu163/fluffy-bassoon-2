# RoboTwin: RLinf and VLRLA baselines

RoboTwin is an active benchmark arm for the frozen-VLA adaptation study.
Use the six source tasks from VLRLA, with RLinf providing ordinary online
RL and source learning histories, and VLRLA-bridge providing the direct
in-context baseline. RoboCasa tests whether a frozen policy uses the feedback
from a specific attempt. Both benchmarks use native simulator rewards.
The task split is already encoded in
[`configs/robotwin_vlrla.json`](../../configs/robotwin_vlrla.json).

## Shared task split

| Source/history training | Held out from history training |
| --- | --- |
| `handover_mic` | `adjust_bottle` |
| `pick_diverse_bottles` | `move_stapler_pad` |
| `place_a2b_left` | `place_a2b_right` |
| `place_object_stand` | `place_dual_shoes` |
| `place_phone_stand` | `place_empty_cup` |
| `scan_object` | `stamp_seal` |

Use [Sidney's broad multi-task SFT checkpoint](https://huggingface.co/SidneyXie/pi05_robotwin)
as the common initializer, pinned per run. The intended starting point has no
RL tuning; it may include all twelve benchmark tasks in its SFT exposure.
Use its published SFT results as reference values. A fresh SFT benchmark panel
is not a prerequisite for the PPO execution pilot. Qualify the model boundary
and actual source environments, then verify the scene banks before expanding
collection to all six source tasks. Keep the six held-out tasks for
transfer evaluation; their episodes must not enter source histories, student
gradients, normalization or checkpoint selection. "Held out" concerns
adaptation training; audit and disclose the base motor's task exposure
separately. The task-specific RLinf `adjust_bottle` checkpoint is useful for
runtime checks, but does not supply a common six-task initializer.

This split comes from [VLRLA Section 5 / Table 1](https://arxiv.org/html/2610.19517v1#S5),
whose comparison includes all twelve tasks. Its appendix's
`adjust_bottle` example does not change the six-task source split.

## Comparison and data roles

| Arm | What learns | Role |
| --- | --- | --- |
| Common SFT pi0.5 | No updates or retained history | Starting competence and headroom |
| RLinf PPO | Source action expert and critic; frozen VLM | Chronological improving histories, including failures; also report its ordinary RL learning curve |
| RLinf GRPO | Online policy parameters | Online-RL baseline used in VLRLA's main comparison |
| VLRLA-bridge | Recurrent low-rank KV bridge during supervised fitting; frozen original VLM/action expert | In-context baseline; deployment updates memory with all weights fixed |
| Our adaptation variants | Declared per run | Compare to the same motor, source corpus and evaluation cases; name the actual methodological difference |

PPO history collection and GRPO comparison are different roles in the
[VLRLA protocol review](../literature/vlrla-protocol.md). A GRPO smoke is
neither a PPO history corpus nor evidence that an in-context learner adapts.
The earlier Motus/noise-steering runs remain historical evidence in
[FINDINGS](../FINDINGS.md#robotwin-motus--vlrla-style).

## Protocol and interpretation

- Bind one common motor's checkpoint, normalization, transforms, action
  dimensions and training exposure across all six tasks and comparison arms.
  Preserve the ALOHA-AgileX embodiment and the native three-camera interface.
- Pin RLinf, RoboTwin and this repository for execution. The latest upstream
  recipe is an audit input; an older qualified runtime needs an explicit
  migration and actual-device qualification before using changed code.
- Use native simulator reward for source learning and support feedback.
  Evaluate the subsequent query with the native task predicate.
- Declare horizon, executed chunk, sampler and randomization identically
  across paired methods. The VLRLA collector specifies 400 controls, H50 and
  five flow steps. RLinf's released `adjust_bottle` pi0.5 recipe uses 200
  controls. Report either protocol by name; a horizon change is a reconstruction.
- Preserve whole source learning streams before splitting student data.
  Compare learning on separate development scenes before scaling collection.
  Save the actual executed actions, outcomes and policy-version chronology.
- Report per-task native success and success by deployment attempt, with
  matched scene and physical-action budgets, source/training costs and online
  update costs. Retain full/reset and reward-intervention controls for memory
  claims. The paper's 128-instance evaluation is a reference; its main-table
  history budget remains unspecified and must be declared in our contract.
- Use official VLRLA artifacts if verified. Otherwise label the bridge as a
  reconstruction and disclose missing checkpoint/evaluation details. Published
  paper percentages are reference values, not locally reproduced results.

RLinf's [training](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/robotwin.html)
and [evaluation](https://rlinf.readthedocs.io/en/latest/rst_source/evaluations/guides/robotwin.html)
guides explain the task presets and seed files. Generic task support does not
imply every VLRLA task has a shipped preset, seed bank or common checkpoint.
Audit those separately before launching a six-task run.

The new [restoration and objective-comparison plan](../../outputs/robotwin-restoration-20261106-r1/PLAN.md)
incorporates the joint Motus initializer candidate and the requirement to port
RoboCasa engineering to training and evaluation. Motus has native H32/delta
joints; the qualified Sidney reference has H50/absolute joints. Qualify and
name each motor/protocol separately, using one common motor per comparison.
The [engineering audit](../../outputs/robotwin-restoration-20261106-r1/ENGINEERING.md)
tracks batched inference, prefix reuse, device binding, memory capacity,
probability replay, execution durations, artifact I/O and recovery.

Separate the learning objective from the steering interface. For the new
RoboTwin comparison, action residuals provide observable teacher-minus-base
labels for AD, DIT-style weighting and offline actor-critic learning; the
learned-KV bridge remains a separately named interface comparison. DPT needs
qualified labels at the actual query observation. Improving PPO histories
support an improving-process AD claim; informative mixed-quality experience
can support DIT/offline value learning without that gate. All-zero rewards or
insufficient action coverage still require diagnosis before fitting.

Dated preparation, source snapshots and readiness (earlier audit):
[`outputs/robotwin-rl-baselines-20261105-r1/PLAN.md`](../../outputs/robotwin-rl-baselines-20261105-r1/PLAN.md).
Live ownership and jobs belong only in
[`outputs/coordination.md`](../../outputs/coordination.md).
