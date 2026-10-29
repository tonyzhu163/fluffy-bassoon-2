# Three design axes for in-context steering of frozen VLAs

We study how training method, memory architecture and steering interface
affect a frozen VLA's ability to improve from deployment experience.

The [ICRL survey](https://arxiv.org/html/2502.07978) separately discusses
supervised/reinforcement pretraining, test-time context and architectures.
The three-axis organization here adapts that perspective to a frozen robot
motor; it is not the survey's verbatim taxonomy.

| Axis | Question | Candidate choices |
| --- | --- | --- |
| Training | How are the adaptation weights learned? | Direct reinforcement pretraining (RL²-style); AD-style distillation of ordered learning histories; DPT-style expert action prediction from experience contexts |
| Architecture | How is interaction history represented and processed? | GRU, Transformer-XL, CTM; other cores after a useful baseline |
| Interface | What can the learned controller change? | Initial diffusion/flow noise; action residual; native features/KV |

Context contents, construction, causal timing, truncation and reset rules
must be explicit for every configuration. Architecture alone does not specify
these. Task distribution, motor exposure, reward, normalization and evaluation
budgets form the common experimental contract, rather than additional knobs
changed during an architecture comparison.

## Initial comparison

Fix the pretrained motor, current-observation interface and **per-action-chunk
noise steering**. Start with **GTrXL** and direct RL²-style PPO training.
Ordinary DSRL is the practical gradient-based reference. Use a separately
trained episode-reset controller to isolate benefit from across-episode
memory under the same recurrent architecture and optimizer.

An RL²-style controller receives current observation features and previous
noise, reward and boundary flags. It produces the noise passed to the frozen
VLA. Training samples tasks/conditions, runs several episodes per trial with
memory retained, and optimizes trial return. Evaluation holds weights fixed,
starts with empty memory per new task trial and retains memory between its
episodes. The primitive environment resets while task identity stays fixed;
identical physical starting scenes are not required.

DSRL's noise interface does not prescribe a meta-training algorithm.
Original DSRL commonly uses SAC; an RL²-style recurrent noise controller can
use a suitable outer optimizer such as PPO. This is an RL² learner through
the same interface, not a claim to reproduce the native DSRL learner.

## Run order

1. Verify the common motor, real simulator/rendering and ordinary DSRL on the
   selected tasks. Record an actual improvement curve, useful failures and
   frozen-motor integrity. A successful optimizer step alone is insufficient.
2. Obtain the first noise-interface result: native motor, static/frozen DSRL,
   target-updating DSRL, recurrent XL and a separately trained episode-reset
GTrXL. Use paired task instances, fixed deployment interaction budgets,
   normal/reset/reward interventions and per-attempt outcomes.
3. Hold the training recipe/interface fixed and compare GRU, GTrXL and CTM.
   Match observations and training interactions; report parameter counts,
   optimization cost and wall/GPU time explicitly.
4. Hold a competent architecture/interface fixed and compare direct RL²,
   AD-style distillation and DPT-style training. Do not treat AD and DPT as
   identical supervision: AD predicts the source learning process; DPT needs
   appropriate expert target actions conditioned on experience contexts.
5. Investigate other steering interfaces after the initial result provides
   a reason. A fixed noise arm per episode is a distinct interface from
   fresh noise chosen each chunk and must be reported separately.

## Data and claims

Direct online meta-RL does not require a teacher dataset. If distillation is
used, retain the teacher's early, intermediate and late experience, including
failures; preserve task/case identity, observations, chosen noise, executed
actions, reward, boundaries, source-policy version, ordering and valid masks.
Checkpoint/update identifiers are provenance, not privileged student inputs.
Store complete resumable teacher state separately from student context.

A frozen VLA limits what steering can express. First establish that available
noise can produce useful alternatives. Earlier experience must reveal
information useful to subsequent decisions, such as hidden dynamics,
ambiguous goals or strategy outcomes. Fully observed tasks can favor static
policies, and near-ceiling panels leave little room for measurable gains.

Report zero-shot performance, success by attempt, cumulative trial return,
success at matched interaction budgets, and total training cost separately.
Use task/case-level paired analysis and independent training seeds. Better
held-out first-attempt performance supports generalization. Improvement
attributable to useful history supports in-context adaptation. A CTM claim
requires evidence against matched memory architectures.

Published VLRLA details live in [its protocol record](vlrla-protocol.md).
Existing measured starting evidence and the dated decision are in
[the first-results record](../outputs/vlrla-protocol-20261029-r1/FIRST-RESULTS.md).

## Initial sampling, horizon and PPO contract

Sample the six VLRLA source tasks uniformly, using equal task weighting in
the training objective. A finite rollout batch need not divide exactly by
six; use a balanced cycling schedule or compensate its task counts explicitly.
Hold task identity fixed for a **four-episode training trial**, with fresh
initial states from that task's randomized reset distribution. Keep all
successes and failures. No new hidden-physics distribution is a prerequisite
for this first baseline; whether ordinary RoboTwin experience provides useful
adaptation is an empirical question.

The default evaluation is an **eight-episode trial**, with results at episode
indices **1, 2, 4, 8** and cumulative trial outcomes/costs. Episode one has no
past episode; episode eight has seven past episodes. These are episode
indices, not one/two/four/eight support episodes plus an extra query. Episodes
five through eight are trial-length extrapolation for a K4-trained model.
Evaluation prefixes belong to the same trial; do not restart memory at each
reported budget. A future K8-training comparison must match interactions.

Use VLRLA's source PPO scale as a reference: 25 iterations, 256 logical
environment lanes, four episode rounds, a 400-primitive-step episode limit,
five optimization epochs and a global minibatch equivalent to 2,048 chunks.
At a native 50-action horizon this gives 8,192 nominal chunk slots/update,
204,800 across training, and 10.24 million primitive-step slots. This is a
reference capacity, not a resource commitment or measured executed data.
If the selected motor has another native horizon, recompute and explicitly
record the changed chunk counts rather than changing its action contract
merely to reproduce this arithmetic. Physical execution batches follow
measured allocated-resource capacity while preserving the saved global batch.

Initial PPO loss settings follow Table S2: gamma 0.99, GAE lambda 0.95,
normalized advantages, clipping 0.2, dual clipping 3, value clipping 0.2,
Huber parameter 10, value weight 1, no entropy/reference-KL bonus and gradient
norm cap 1. Recurrent replay preserves sequence order and its initial memory;
it must not shuffle individual chunks as independent samples. Learning
credit continues across within-trial episode resets; the done signal remains
an input, while task/trial boundaries govern the meta-return. A small randomly
initialized steering controller's learning rate needs qualification separately
from the paper's 5e-6 rate for its pretrained action-expert teacher.

GTrXL uses relative attention, pre-normalization and gated residuals. This
repo's compact implementation was qualified on the Can pilot; that does not
qualify its VLA/RoboTwin integration. See the
[GTrXL paper](https://arxiv.org/abs/1910.06764).
Original [RL²](https://arxiv.org/pdf/1611.02779), Sections 2.2–2.4, uses a
fixed MDP within each trial, fresh episode starts, persistent GRU state and
TRPO. Its navigation experiment trains on two episodes and tests up to five.
Our GTrXL/PPO/K4-to-K8 configuration is an adaptation of that training
principle, not an exact RL² or VLRLA reproduction.
