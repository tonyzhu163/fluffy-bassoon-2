# Simulator PPO versus Flora PPO: retained-data investigation

The clearest measured mechanism is reward/metric mismatch. Simulator PPO gets
the exact terminal criterion used for evaluation. Flora optimizes a graded
visual proxy that sometimes calls simulator failures complete successes.
This is a plausible mechanism, not proof that it caused the small observed lead.

| Policy | OpenDrawer ID | CloseFridge OOD |
| --- | ---: | ---: |
| Untouched original |28/50|28/50|
| Simulator PPO |34/50|32/50|
| Flora PPO |31/50|28/50|

The S/F lead is only3 ID cases and4 OOD cases. Paired exact two-sided p-values
are0.508 and0.388. There is one training seed, and frozen-motor outputs are not
bitwise identical despite matching actual case/noise schedules and initial
public proprioception. Neither a reliable population advantage nor a unique
cause is established. Flora has no net OOD decline versus the original:9 paired
wins and9 losses. [Full paired audit](../robocasa-original-pi05-eval50-20261102-r1/TERMINAL-REVIEW.json).

Initial residual actor/critic tensors, learner configurations and feature
normalization match. Each arm completed384 TRAIN episodes,6 updates and360 Adam
steps. Flora recovery continued the original Adam counts through240→300→360.
No incomplete labeled batch entered SGD. KL and clipping diagnostics are finite,
and neither final residual head saturates its action bound on the fixed original
feature panels. A shorter or broken Flora run does not explain the result.

True TRAIN success is197/384 for S and201/384 for F. Both curves fluctuate around
half success, and neither improves its8-case DEV panel beyond the initial7/8.
The observed held-out gap is therefore not evidence that F simply failed to fit
while S clearly learned a stronger source policy.

Flora reward ranks actual TRAIN success reasonably well:AUROC0.925 over384
episodes. Mean normalized reward is0.859 on201 successes and0.441 on183 failures.
However,23 failures receive reward≥0.75; the highest receives0.96695, while one
success receives0.28915. The model explicitly emits `Score: 5` on58/183 failures
and196/201 successes. These are real full-success classification discrepancies,
not only partial-credit examples. [Grade confusion](GRADE-CONFUSION.json).

The native OpenDrawer predicate requires every target-drawer opening value to
be≥0.95 at the final step. Three-view frame sheets for the two highest-reward
failures show visible drawer opening; the low-reward success has substantial
occlusion/view limitations. Precise final drawer-joint values were not saved,
so the images do not prove whether each discrepancy is near-threshold opening,
target confusion or visibility. [High failure1](high-failure-1.png),
[high failure2](high-failure-2.png), [low-scored success](low-success-1.png),
[native predicate](SUCCESS-DEFINITION.json).

On the same saved F records and critic, changing only reward labels to binary
terminal truth changes normalized advantage signs on9.9–27.1% of decisions
across the six batches. In batch6 it changes23.1%, and the full-batch actor-gradient
cosine is0.308. Advantages are normalized in the real learner, so raw reward
scale alone is not an adequate explanation. This diagnostic keeps the F critic
fixed and takes no optimizer step; it is not a trained binary-reward ablation.

On identical original-policy feature panels, F's mean absolute residual is4.59%
of its bound on ID and7.07% on OOD; S's is6.72% and6.34%. This shows different
generalization of the heads, but does not establish closed-loop causes of wins
or losses. [All numerical evidence](REPORT.json), [all768 TRAIN episode rows](EPISODES.json).

The current reward `(E[grade]-1)/4` is not a calibrated probability of full
success. A distribution centered on grade4 maps to0.75 reward. Rubric clarity
and reward aggregation are separate design choices; neither should be changed
silently in an existing run. The follow-up [rubric audit](../robocasa-flora-rubric-audit-20261102-r1/AUDIT.md)
keeps physical thresholds in a human reference and uses observable language
for the candidate prompt.

All diagnostics ran on allocated Linux CPUs with no new simulator episodes,
GPU allocations or optimizer steps. The initial numeric run finished its JSON
outputs and then encountered a missing plotting dependency. The plot was
rendered separately in an isolated uv environment; shared model environments
were unchanged. [Execution receipt](COMPLETED.json).
