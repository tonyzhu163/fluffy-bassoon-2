# π0.5 history interface on LIBERO-Gen

Not launched. Created 2026-11-11 from main `1b4b9b1f`. This is the build plan for the main experiment: a history module meta-trained together with π0.5's action expert, with every weight fixed at deployment. The [Behavior Prompting motor check](../libero-gen-history-pilot-20261111-r1/PLAN.md) runs separately and comes first.

## Question

After one demonstration leaves a task ambiguous, does π0.5 with a trained history input choose and execute the right behaviour on a fresh scene, using its own earlier attempts and their native success bits, with no weight updates?

## What exists

| Piece | State |
| --- | --- |
| History into π0.5's action expert | A GRU, a projection into native action tokens and rank-8 query/value LoRA on the last four action-expert blocks, trained by flow matching through the frozen model. Written for the RoboCasa Human300 checkpoint in PyTorch through RLinf ([fit.py](../robocasa-three-ppo-dpt-lora-20261110-r1/fit.py)). Zero-adapter parity and nonzero gradients were verified on GPU. |
| That interface's only test | Own history did not beat a donor's history ([result](../robocasa-three-ppo-dpt-lora-20261110-r1/RESULT.md)). The histories there carried no hidden information, so this says nothing about the interface. |
| π0.5 on LIBERO-Gen | Behavior Prompting releases language-conditioned LoRA checkpoints, trained and served in JAX from their openpi fork (`liberogen` branch), with LeRobot-format datasets at 256×256. Not downloaded or run here. |
| Training recipe | The [toy](../learning-histories-fix-20261111-r1/RESULT.md) supports supervising every prefix length. It also shows a learner at 60% where the reference reaches 87% with outcome feedback, and a 9-point held-out gap with 99 training rules. |

## Hidden rules available in the release

From the static [release audit](../libero-gen-history-pilot-20261111-r1/RELEASE-AUDIT.json) of Spatial Combination (193 task files):

- 11 pick descriptions ("the black bowl next to the ramekin") and 10 main placement targets ("on the plate", "on the stove"), plus 11 bowl-in-bowl targets.
- 53 groups of tasks share one physical scene description but differ in goal. Eleven groups hold 6 to 12 alternative goals; 42 hold two.

So one scene supports up to 12 alternative behaviours, and a placement target persists as a rule across scenes. That clears the three-hypothesis minimum. It is still at most about 100 pick-and-place rules over roughly a dozen layouts, fewer than the toy's 99 training rules, which already allowed memorization. The release can show the effect. A claim about unseen rule families needs newly generated tasks.

Unverified: whether tasks in a group reset to identical states, and which tasks have demonstrations.

## Design

1. **Session.** Sample a hidden rule, one ambiguous demonstration, K attempts on fresh layouts, then a query on another layout. The instruction is generic ("do the demonstrated task") so language cannot reveal the rule.
2. **Attempts without new simulation.** A released demonstration of a *different* goal in the same scene is a competent execution of the wrong choice. Label it with the native success bit under the hidden rule's goal. Replay a sample in the simulator to confirm those bits before trusting them.
3. **History module.** Per event: frozen π0.5 vision features at the first and last frame, an action summary, role (demonstration, attempt, correction) and outcome. A small Transformer turns events into history tokens for the action expert. LoRA on the action expert is trained with it.
4. **Objective.** Flow matching on the expert query chunk under the hidden rule, at every prefix length in one pass. Failed actions are context only, never targets.
5. **Deployment.** All weights fixed. The model makes its own attempts; report native success after each.

Use the existing PyTorch interface if the released LoRA checkpoint converts with numerical parity (merge LoRA, convert, compare actions against the JAX server on saved observations). Otherwise add history tokens in the JAX fork.

## Gates, in order

| Gate | Pass condition | If it fails |
| --- | --- | --- |
| 0. Language motor | Released π0.5 checkpoint succeeds on at least 85% of episodes for each candidate task, by its documented evaluator | Use tasks it can do, or the Behavior Prompting motor only |
| 1. Rule inventory | At least three alternatives per scene with demonstrations and matching resets, and each rule appearing in at least three layouts | Generate tasks with the release's generator |
| 2. Conversion | PyTorch actions match JAX on saved observations | Build in JAX |
| 3. Offline separation | On held-out sessions, flow error with the session's own history is below error with a donor history of the same scene, interval excluding zero | Stop before any rollout; this is the test the RoboCasa pilot failed |
| 4. Closed loop | Success after K attempts above the same model with outcomes masked or shuffled | Report as a null with the motor ceiling |

## Comparisons at gate 4

Generic-prompt LoRA with no history (floor), the language-informed released motor (ceiling), demonstration only, attempts with outcomes, outcomes masked, outcomes shuffled, teacher corrections, and fine-tuning on the same session data as a separate arm.

## Budget

Gates 0 to 2: at most 10 GPU-hours. Gate 3 pilot fit: at most 20. Ask before gate 4 if the total would pass 50.
