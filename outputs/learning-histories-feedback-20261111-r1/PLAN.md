# Frozen-checkpoint feedback diagnosis

Continue the synthetic exploration using the three existing `attempts` checkpoints
from `learning-histories-20261111-r1/grid01`, trained at commit `50508c86`.
No optimizer updates or changes to that run. The user authorized continuation
on November 11 after the initial results and recommended diagnostic sequence.

## Locked comparisons

- First reproduce both original 8,192-session evaluation curves exactly for each
  seed. This qualifies checkpoint loading and the inference path on the actual GPU.
- Then use fresh 8,192-session panels per split/seed, session RNG `30000+seed`.
  One persistent hidden rule, one demo, six attempts and a fresh query.
- Three history policies on the same scenes: the checkpoint's own greedy choices;
  the original posterior-sampling plus 20% random training actor; and uniform
  random choices independent of past outcomes. RNG seeds are in the evaluator.
- Freeze every history's scenes and choices before intervention. Evaluate all
  seven prefixes with real outcomes, masked outcome channels, outcomes shuffled
  across sessions independently per attempt index, or demo-only context. The
  random-actor panel eliminates indirect reward information in actor choices.
- Report paired gains/losses and prediction changes, within-seed paired normal
  95% intervals, and replication across all three independently trained seeds.
  Primary comparison: real versus shuffled outcomes at k=6 on fresh held-out
  random-actor histories. Also require real versus masked to have the same sign.
- Separate history usefulness from reading quality using exact enumeration on
  the same unmodified histories. Report both the original all-rule uniform prior
  and the evaluator's split prior, clearly labelled. The latter knows the split's
  candidate set and is an information diagnostic, not a fair learned baseline.
- Report all-rule and two-attribute-only subsets to avoid confusing arity mix
  with generalization. Measure query cross-entropy as well as accuracy.

These are frozen-input interventions, not alternate closed-loop rollouts or a
new exploration policy. Masked/shuffled inputs can be outside the training
support, and shuffled histories can be inconsistent. Interpret a positive effect
as beneficial model dependence on feedback on these panels, not full robotic ICRL.
No LIBERO-Gen work is part of this run.

## Execution

Use the latest reviewed main commit, immutable Git deployment, unchanged qualified
Torch CUDA environment, and three scheduler workers with one allocated GPU each.
Run the GPU probe with evaluation purpose. This task has no physics simulator;
the real workload qualification is exact replay of the original synthetic curves.
Cap each worker at 45 minutes: at most 2.25 additional GPU-hours. Prior exploration
accounted for 7.424 scheduler GPU-hours (7.620 submission-to-end upper bound).
Check completion artifacts immediately after launch, then every 20 minutes or
at expected completion. Keep checkpoint files read-only and retain new traces
in this run. Only small summaries/receipts go into Git; NPZ traces remain shared.
