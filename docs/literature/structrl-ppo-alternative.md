# StructRL reward alternative under a matched PPO schedule

Keep this experimental formulation on `codex/structrl-ppo`. It is a
StructRL-method adaptation to our existing atomic tasks, rather than a
reproduction of the paper's 16-composite-task success rates. This document
specifies the comparison; it is not an implemented reward backend or a launch
contract. The existing production learner and active runs retain their code.

## Shared experiment

The selected comparison uses **256 environments and 50 training iterations**
for both the simulator-sparse and structured-reward arms. Both arms must share
the same resolved baseline settings: atomic task set and sampling weights,
initial checkpoint and trainable interface, rollout rounds, PPO epochs,
minibatch size, optimizer, GAE, exploration, chunk length, horizons, evaluation
cases and checkpoint-selection rule. Resource placement is an execution
choice, distinct from the global environment count.

Do not inherit the old 32-environment / 18-iteration OpenDrawer pilot as the
current schedule. Do not substitute the paper's 100 iterations, 256×2 rollout
configuration or one PPO epoch into only the structured arm. Remaining baseline
choices should be copied from the finalized common producer recipe, rather
than guessed from an old config. Preserve the existing launch hold until its
owner binds that recipe; branch preparation does not release it.

If `R` denotes fresh rollout rounds per iteration, this schedule collects
`256 × 50 × R = 12,800R` complete training episodes. StructRL reports
`256 × 2 × 100 = 51,200`. Thus our schedule would match its episode count
at `R=4`, or use half at `R=2`. Episode count alone does not match primitive
controls, task difficulty or compute: task horizons and valid chunk counts
also matter.

**PPO epochs** are repeated optimization passes over one fixed rollout batch.
They add no fresh simulator experience. With `N` valid chunk samples and
minibatch size `M`, `E` epochs require approximately `E × ceil(N/M)` optimizer
steps, subject to the learner's batching rules and KL early stopping. Keep
the same requested epochs and KL rule in both arms; report actual completed
epochs and optimizer steps separately. Fresh rollout rounds and PPO epochs
must never be counted as the same kind of repetition.

## Reward construction to implement

Use the released method's dependency-gated, one-time completion rewards:

1. Ground a small set of genuine task-progress milestones to binary simulator
   predicates. Exclude predicates already true at reset or unreliable under
   replay. Proximity alone is not a qualified completion signal.
2. Organize predicates into ordered stages. Within a stage, completion order
   is free; later stages become eligible only after earlier stages complete.
3. Pay each eligible predicate once per episode. Reopening/closing or dropping
   and regrasping must not produce repeated payments.
4. With demonstration-derived reference gap `T_d`, use the released pacing
   curve `beta / (1 + T/T_d)`, where `T` is elapsed primitive controls since
   the previous credited completion. The release uses `beta=.6` and a
   one-time task-success reward of `2`. Match the terminal coefficient between
   sparse and structured arms; if rescaling, rescale and record it explicitly.
5. Detect events at primitive-control resolution, then sum rewards into the
   actual executed action chunk. Preserve the checked collector's boundary
   convention and first-success loss mask. Do not copy the upstream done-shift
   patch into our already checked collector.

The paper and release differ in timing details: the released builder uses
stage/completion-rank **median gaps** from demonstrations. Preserve the exact
chosen convention and primitive-control time units. `STRUCTRL_REWARD_VARIANT`
must select `structrl`/`smooth_band`; the release's older `timing` default
includes a persistent penalty that the selected paper reward removes.
[Reward guide](https://github.com/amazon-science/StructRL/blob/6caba1f9538167e0752a7338663e45aac8df965d/docs/REWARD.md),
[stage definitions](https://github.com/amazon-science/StructRL/blob/6caba1f9538167e0752a7338663e45aac8df965d/structrl/stage_config.py),
[timing builder](https://github.com/amazon-science/StructRL/blob/6caba1f9538167e0752a7338663e45aac8df965d/structrl/timing_reward_builder.py).

## Atomic-task qualification

The released stage table covers composite tasks, so it cannot be enabled
unchanged for our atomic task set. New predicates and timing references are
required. For OpenDrawer, measured partial drawer opening is a candidate
milestone; it needs reset-state, replay and completion checks before use.
Adding another payment at the existing final-success threshold provides no
earlier credit. A task without a reliable intermediate predicate should retain
sparse reward rather than receive an invented milestone or reference duration.

Simulator state is privileged **training-reward supervision**. It must not
enter policy observations, frozen VLA features, steering inputs or evaluation
success definitions. Evaluate both arms with the same native task-success
predicate, paired cases and uncertainty; shaped return is diagnostic only.

Before integration, numerically check prerequisite gating, repeat-payment
prevention, reset, simultaneous events, timing units, chunk aggregation and
terminal masking. Then verify the actual predicates in native episodes before
training. New numeric/core reward tests belong with the implementation;
bookkeeping tests and a new receipt framework do not.

See the [literature and recipe audit](robocasa-ppo-review.md) for external
results, baseline differences and the existing saved-batch GAE qualification.
