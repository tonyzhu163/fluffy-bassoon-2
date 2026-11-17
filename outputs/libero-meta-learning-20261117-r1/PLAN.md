# LIBERO meta-learning test

Design proposal, November 17, 2026. No job has been submitted under this
identity. This replaces further expansion of the narrow RoboCasa offset
study as the proposed main experiment. The existing campaign ceiling remains
200 GPU-hours, including the 2.36 conservative GPU-hours already charged;
this is not a new 200-hour allocation. Qualification and a measured cost plan
are required before committing to episode counts or a three-day completion.

## Claim

A learner trained across tasks can use a small amount of new experience to
improve its next attempt on a task excluded from its meta-training, without
changing any weights. Improvement must be attributable to relevant history
and transfer beyond one task or one calibration mechanism.

Separate three claims: initial competence, benefit from prior experience,
and dependence on reward feedback. An increase in first-attempt success is
not evidence for the second claim. Useful motion history without dependence
on rewards supports context adaptation but does not establish reward-driven
adaptation. Success on changed parameter values does not establish transfer
to a new task or mechanism.

## Benchmark and split

Use LIBERO-Pro2.0 as the primary candidate. Its published split provides
32 training base tasks and eight held-out base tasks across four suites.
For this experiment, propose 24 fitting tasks and eight development tasks
from that 32-task pool, balanced across suites and semantic skill coverage;
retain the official eight test tasks. This additional development split is
ours, not the official benchmark protocol. Freeze exact IDs after inspecting
the task manifest and semantic overlap, before outcomes guide selection.

Keep all descendants of a base task (instructions, assets, perturbations,
teacher streams and resets) in its split. Near-duplicate goals require an
explicit overlap audit. Report task-ID holdout honestly; a stronger skill-
family holdout needs a separately defined semantic split. Audit the frozen
VLA checkpoint's training exposure: these may be unseen only by the history
learner, not by the backbone. Do not reuse the RoboCasa motor's normalization
or controller contract for LIBERO.

Define perturbation families by mechanism, not by file ID or static/dynamic
label. Candidate training coverage includes geometry/layout, object/scene
appearance, camera geometry and execution timing. Candidate withheld
mechanisms include observation dropouts/corruption and grasp/drop recovery.
These assignments must be mapped to exact release IDs and checked for
overlap before collection. For example, training on fixed action delay makes
delay jitter a related timing variation, not an entirely unseen family.
An excluded family must also be absent from augmentation, teacher learning,
normalization and model selection for the history learner.

| Evaluation cell | Task | Environment change | Interpretation |
| --- | --- | --- | --- |
| ID | Fitting task, fresh reset/trial | Training family and distribution | Within-distribution history use |
| Task OOD (primary) | Official held-out task | Training family and distribution | Transfer of adaptation across tasks |
| Shift OOD | Fitting task, fresh reset/trial | Withheld mechanism | Transfer across change types |
| Combined OOD | Official held-out task | Withheld mechanism | Both sources of novelty |
| Composition extension | Held-out task | Untrained combination of individually trained changes | Compositional transfer, separate from family holdout |

All cells contain fresh trial identities. Publish task/family membership and
valid case lists before evaluation; do not admit test cases based on nominal,
teacher or learner success. Report the full valid panel, including floors
and ceilings. A nominally solvable subset may be descriptive only. Invalid
simulator cases need outcome-independent reasons and explicit counts.

LIBERO-Plus is a later external check on overlapping task/shift categories,
with overlap and frozen-backbone exposure documented. It is not automatically
an independent task holdout. Preserve official independent-episode scores
separately from our repeated-attempt protocol.

## Trial: what persists and what changes

One trial is a task plus a persistent environment condition, with three
attempts and no weight updates. Reset the simulator between attempts; retain
the task and condition but use preassigned fresh, valid initial states.
Reset all learner memory between trials. The retained condition may be a
camera transform, object geometry, action timing, or an intervention rule.
For dynamic interventions, retain the rule/distribution rather than replaying
an identical timed trajectory. Draw event randomness and policy noise from
separate paired streams. Do not require every change to be hidden: visible
changes provide useful tests of whether memory adds anything beyond current
observations. Completely independent one-off faults provide no persistent
condition to infer and must not be advertised as guaranteed history tasks.

Inputs are current cameras, public robot state, instruction, causal prior
observations, executed actions, native rewards and episode boundaries.
Exclude perturbation IDs/parameters, simulator internals, source checkpoint
identity, future outcomes and teacher targets. Match the physical-control
budget per attempt, action horizon and inference settings across arms.
Queries stop at termination. New attempts still start after previous success
so attempt-2/3 success has an unconditional denominator over all trials;
report a separate stop-on-first-success deployment metric if desired.

## Two complementary evaluations

**Matched experience comparison (primary, two attempts).** For every case,
collect one real support attempt using the same frozen SFT behavior policy.
Give this same public tape to all four learners. The query starts from a
fresh reset, paired across arms, with the condition retained. Compare:

- Relevant support history.
- Cleared prior-episode history, preserving the complete current-episode
  prefix; otherwise identical learner weights and inference.
- Donor history from the same task and family but a different condition,
  matched on support length and native outcome where the pool permits.
- A separately trained current-episode-only model of matched capacity.
- Frozen SFT and a frozen PPO checkpoint evaluated on the same query cases.

All methods receive the same allowed support and query interaction budget;
baselines may ignore support. Charge support collection even when its tape is
reused computationally. Donor selection is frozen using support metadata
only, never query outcomes. This comparison measures ability to use matched
experience, not the ability to collect informative experience autonomously.
Do not describe the shared SFT support as the learner's own trajectory.

**Autonomous adaptation comparison (three attempts).** Each learner generates
its own support experience. At attempts 2 and 3, fork evaluation from the
same preassigned reset with retained versus cleared prior-episode history.
Only the retained branch's actual trajectories enter the next support; other
branches are counterfactual evaluation costs, not additional agent experience.
Run the independently cleared policy through its own three-attempt trajectory
as well when reporting an end-to-end deployment comparison. SFT and frozen
PPO get the same three-attempt budget. This panel determines whether a
matched-support gain survives the learner's own failures and distribution
shift. Select any reduced set of methods for this more expensive panel using
development outcomes only and label coverage explicitly.

A reward-masked history intervention and a model trained with reward-masked
history are secondary diagnostics. Observations and termination can reveal
success, so masking the reward scalar alone cannot prove reward independence.
Matched donor history and the trained current-only baseline distinguish
specific experience from extra tokens and out-of-distribution memory removal.

## Training the four learners

Use one frozen, LIBERO-qualified VLA and the same history-to-KV interface,
context budget, actor initialization, actor update count and query sampler.
Keep the motor frozen during fitting; freeze the learner too at evaluation.
Use three independent fitting seeds for confirmatory comparisons, subject to
the measured budget. Report fewer seeds as exploratory, without treating
extra environment episodes as replacements for independent fits.

Build a shared multi-task source corpus with successes, failures, executed
control masks and native feedback. Keep task-condition identity consistent
through each history. Store real chronological source rollouts, including
intermediate PPO training stages, rather than sorting episodes by success or
joining unrelated scenes. PPO source training is a measured means of obtaining
useful variation, not an automatic requirement to train dozens of teachers.
Existing demonstrations can initialize or supplement training, but successful
demonstrations alone do not supply a reward-conditioned learning history.

| Method | Query supervision on the shared source states |
| --- | --- |
| AD | Executed behavior action in actual chronological source streams; demonstrate source improvement before claiming distillation of a learning process |
| DPT | A preselected competent teacher's action at the exact saved query observation; earlier actions and feedback stay as experienced |
| DIT-style | Executed actions weighted by estimated advantages, with causal value inputs and returns used as training targets only |
| IC-IQL | Causal Q/V fitting and advantage-weighted imitation of the recorded actions |

Store one shared compressed corpus and one copy of the frozen weights.
Methods reference the same observations instead of duplicating them. Keep
teacher checkpoints required for labels and a complete resumable latest
checkpoint; do not save a full motor at every PPO update. Retain videos for
diagnostics and a declared sample, not every rollout. Check storage capacity
and active dependencies before collecting or archiving artifacts.

DPT receives additional teacher labels; count that supervision and its cost
separately instead of calling all four data-identical. Teacher privilege, if
used, must be recorded and excluded from student inputs and all test-time
policy calls. No test task contributes teacher labels or PPO updates to source
fitting. Preserve the implementation distinctions documented in METHODS.md;
the names alone do not imply exact reproductions of published algorithms.

Before bulk collection, development evidence must resolve the failures seen
in the old experiment: teacher competence, same-condition histories, causal
inputs, actor query coverage, decoded adapter control, and whether estimated
advantages contain useful action information. A weak teacher or a broken KV
interface makes a null outcome inconclusive about meta-learning in general.
Use concrete diagnostics; no arbitrary success-rate admission threshold.

## Endpoints, uncertainty and decision

Primary endpoint for each algorithm: paired attempt-2 success difference
between relevant and cleared support on the task-OOD cell. Macro-average
over tasks and declared families, not over frames or variable-length tapes.
Report absolute SFT/PPO/learner success, paired gain, donor contrast and the
trained current-only contrast together. Correct the four primary algorithm
tests for multiplicity if making significance claims. Other cells and
attempt-3 results are separately labeled secondary outcomes.

For autonomous trials, report success at each attempt, mean successes over
the fixed interaction budget, and cumulative success separately. Never use
only cumulative success or condition the main result on earlier failure.
Paired resets, policy-noise streams and event streams reduce noise but do not
guarantee identical trajectories or cross-process numerical parity. Use
contemporaneous controls on the same qualified runtime/hardware family.

Report confidence intervals clustered by base task and trial, with independent
fit variation shown separately. Only eight test tasks limits task-population
claims regardless of the number of reset seeds. Publish per-task and per-family
results and paired discordant counts. Determine the final paired episode
count using development discordance and measured wall/GPU cost, then freeze
it before opening the confirmatory results. Do not keep extending a fixed
sample test until it becomes significant; a sequential alternative would
need a predeclared sequential-valid analysis.

Continue this recipe if relevant experience reliably improves held-out task
success over cleared and donor history, outperforms the trained current-only
control, survives autonomous trials, and offers a useful success/cost tradeoff
against SFT/PPO. Generalization is only as broad as the passing cells.
Stop scaling this recipe if a qualified interface and informative training
data yield a precise negligible history advantage, or relevant history is no
better than unrelated history. A broad interval is inconclusive, not a null
result. The minimum worthwhile gain must be declared before test inspection
from the measured additional latency/compute and the intended deployment
tradeoff; no numerical usefulness threshold is invented here. Failure of
this recipe does not disprove all forms of meta-learning.

## Execution order and capacity

1. Qualify one LIBERO motor and the KV adapter on development tasks, verify
   native baselines and deterministic pairing, and measure episode, fit and
   storage costs on actual allocated GPUs. Reuse existing code/artifacts only
   after checking their exposure and runtime contracts.
2. Freeze semantic task/family splits and the corpus design. Collect a small
   development sample with actual failures and useful teacher labels. Train
   all four small fits to check the learning signal before scaling the corpus.
3. Reserve confirmatory evaluation capacity first. Publish episode counts,
   fitting seeds and a cost worksheet within the remaining197.64 GPU-hours.
   For N shared-support cases, four methods and three seeds, retained/cleared
   query arms alone cost24N episodes, plus N supports and2N SFT/PPO queries.
   Donor, current-only, autonomous and additional OOD panels cost extra.
   Parallel execution improves elapsed time, not the total GPU-hour budget.
4. Fit the frozen comparison, evaluate the predeclared panel, and publish
   the full per-case results and a continue/stop/inconclusive assessment.

The three-day objective is a decision-oriented experiment, not a promise to
complete every cell, baseline and seed before runtime measurements exist.
Protect the primary task-OOD comparison and its controls when reducing scope;
never silently substitute a small perturbation study for it. Frozen offline
PPO is the baseline here. Online test-task PPO adaptation would be a separate
comparison with matched support controls, environment interactions and an
explicit optimization budget, not an uncharged source of test experience.

## Primary sources

- [LIBERO-Pro2.0 evaluation and split](https://github.com/LIBERO-Pro-Team/LIBERO-Pro2.0/blob/main/docs/remote-evaluation-guide.md).
- [LIBERO-Pro2.0 perturbation mechanisms](https://github.com/LIBERO-Pro-Team/LIBERO-Pro2.0/blob/main/PERTURBATIONS.md).
- [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus).
- [Repository research question](../../docs/PAPER.md) and [method contracts](../../docs/METHODS.md).
