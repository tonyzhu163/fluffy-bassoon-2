# VLRLA: published experimental protocol

This is a factual reconstruction of **arXiv:2610.19517v1**, rather than an
executable reproduction or a record of experiments performed here. It covers
the simulation study, real-robot studies, architecture, data construction,
optimization and evaluation. Unspecified settings remain unspecified.

Primary sources: [paper](https://arxiv.org/abs/2610.19517v1),
[HTML](https://arxiv.org/html/2610.19517v1),
[PDF](https://arxiv.org/pdf/2610.19517v1).
The [source audit](../outputs/vlrla-protocol-20261029-r1/README.md) records
retrieval identities, release searches and the public demonstration manifest.
Our proposed DSRL experiments are a separate
[design](frozen-vla-icrl-design.md).

## 1. Experimental stages and what learns

The simulation pipeline has two distinct optimizations:

1. **Generate learning histories.** Improve a source VLA with online RL;
   retain trajectories from successive policy versions, including failures.
2. **Train the in-context policy.** Fit a recurrent model on these histories
   with supervised action prediction. This model learns its weights before
   deployment; deployment changes its interaction memory.

Three data routes are demonstrated: online-RL histories in simulation,
increasing human intervention for fruit placement, and hindsight assembly of
successful demonstrations for an ambiguous plate-choice task. They are
different datasets and protocols, not interchangeable collection stages.

| Component | History generation in simulation | VLRLA-simp training | VLRLA-bridge training | VLRLA deployment |
| --- | --- | --- | --- | --- |
| Vision-language encoder | Frozen | Frozen | Frozen | Frozen |
| Original action expert | Updated by source PPO | Replaced by recurrent action module | Frozen; differentiable path to bridge | Frozen where present |
| New recurrent module | Not the source teacher described in A.2.1 | Trained by action MSE | Trained by flow-matching loss | Weights frozen, memory updated |

Sources: [Sections 3–4](https://arxiv.org/html/2610.19517v1#S4), Appendix A.2.1.

## 2. RoboTwin tasks and exposure split

Table 1 names **12 evaluation tasks**, with source adaptation histories from
six seen tasks. Preserve the machine identifiers exactly:

| Seen: adaptation-training tasks | Unseen: held out from adaptation training |
| --- | --- |
| `handover_mic` | `adjust_bottle` |
| `pick_diverse_bottles` | `move_stapler_pad` |
| `place_a2b_left` | `place_a2b_right` |
| `place_object_stand` | `place_dual_shoes` |
| `place_phone_stand` | `place_empty_cup` |
| `scan_object` | `stamp_seal` |

"Unseen" here describes the adaptation-training split. The paper does not
provide a complete task-exposure manifest for the initial SFT motor; do not
infer that its original training excluded all six held-out tasks.
Appendix A.2.1 uses `adjust_bottle` as its configuration example, although
Table 1 places it in the unseen set. This does not establish that its
trajectories entered the six-task bridge-training dataset.

Source: [Section 5, Table 1 and Section 5.1](https://arxiv.org/html/2610.19517v1#S5).
The machine-readable split is in
[robotwin-datasets.json](../outputs/vlrla-protocol-20261029-r1/robotwin-datasets.json).

## 3. Simulation history collection

### 3.1 Environment, motor and saved record

| Item | Reported setting |
| --- | --- |
| Platform | RoboTwin; RLinf for online RL |
| Embodiment | ALOHA–AgileX |
| Initial policy | Existing pi0.5 checkpoint configured for ALOHA |
| RGB inputs | Head, left wrist, right wrist |
| Action | 14-dimensional vector; 50-action chunk |
| Sampler | Five flow integration steps |
| Exploration | Flow-SDE, noise coefficient 0.3; stochastic exploration at one integration step |
| Updated parameters | Action expert and associated policy parameters, without low-rank adapters |
| Frozen parameters | Vision-language backbone |
| Value estimate | Pooled action-expert features, detached before value head |
| Record per decision | Instruction, three RGB observations, proprioception, action chunk, reward, termination and truncation arrays |
| Retained outcomes | Successes and failures |

Within a task, collection samples randomized environment instances. It does
**not** require returning to an identical initial state after every episode.
Collection follows source-policy update order: rollout, save experience,
update teacher, repeat. Source optimizer steps are distinct from subsequent
supervised student updates.

### 3.2 Reward and PPO settings

| Item | Reported setting |
| --- | --- |
| Reward | Sparse environment success; first success receives 1 |
| Reward scaling / relative rewards | 1 / enabled |
| Reward and log-probability reduction | Action chunk |
| Advantage | GAE; normalization enabled |
| Discount / GAE lambda | 0.99 / 0.95 |
| Actor clipping | Ratio range 0.8–1.2; dual-clip coefficient 3.0 |
| Critic loss | Clipped Huber; value clip 0.2; Huber parameter 10 |
| Critic weight | 1.0; actor and critic equal weight |
| Entropy / reference KL terms | Neither |
| Optimizer | AdamW |
| Actor / value learning rate | 5e-6 / 1e-4 |
| Adam beta / epsilon | (0.9, 0.95) / 1e-8 |
| Weight decay / gradient norm cap | 0.01 / 1.0 |
| Actor / base environment seed | 1234 / 0 |

### 3.3 Collection, optimization and teacher evaluation

| Item | Reported setting |
| --- | --- |
| Source PPO iterations | 25 |
| Training environments | 256 |
| Collection rounds per iteration | 4 |
| Wrapper episode limit / round horizon | 400 / 400 environment steps |
| Chunk decisions per environment per round | 8 |
| Nominal chunk slots per iteration | 8,192 |
| Post-termination slots | Excluded from the optimization objective with a mask |
| Optimization epochs per iteration | 5 |
| Global optimization minibatch | 2,048 chunk samples |
| GPU microbatch / accumulation | 32 samples / 8 accumulation steps |
| Distributed setup | Eight GPUs on one node, FSDP |
| Offloading | Actor, rollout workers and environment workers |
| Teacher evaluation cadence | Before training, then every five PPO iterations |
| Teacher evaluation | 128 environments, one round, 400-step horizon |
| Teacher evaluation initialization | Separate seed list; fixed reset-state identifiers |
| Table S2 memory-detach entry | 12 chunks, labeled "2 episodes"; interpretation unresolved |

The paper reports nominal capacity **204,800 chunk slots** per full
25-iteration run. Multiplying by 50 gives **10,240,000 primitive-step slots**
(derived capacity, not measured executed interactions or retained valid data).
Do not multiply this by six and present it as an observed dataset size:
per-task manifests and post-mask counts are not supplied.

Sources: [Section 4.2](https://arxiv.org/html/2610.19517v1#S4.SS2),
[Appendix A.2.1 / Table S2](https://arxiv.org/html/2610.19517v1#A1.SS2.SSS1),
PDF pp. 15–16.

## 4. Recurrent policy and supervised training

### 4.1 Shared causal interaction contract

At each decision, the policy receives the current visual-language information
and proprioception, plus previous action chunk, reward and boundary feedback.
Only completed past interactions enter context. Transformer-XL-style memory
connects local causal segments and persists between episodes of a trial.
Parameters remain fixed during deployment. Trial initialization and memory
cache limits need an executable specification beyond this general contract.

### 4.2 VLRLA-simp

Use frozen visual-language prefix embeddings as the perception interface.
A recurrent Transformer and MLP predict deterministic action chunks, trained
with action-reconstruction MSE. Sections 5.2–5.4 use this variant for backbone,
context and transferred-history experiments. The paper does not provide a
complete simp architecture/optimizer configuration comparable to Table S1.

### 4.3 VLRLA-bridge

Freeze both VLM and original action expert. Learned summary queries attend
over valid positions and layers of the native KV cache; action queries read
the summaries together with historical information. Pool their outputs into
a context representation and use a low-rank adapter to modify the native KV
cache. The adapter conditions on current cache content and context, includes
position/layer/key-or-value embeddings and masks padding. Its output bases
start at zero, preserving initial base-policy behavior.

States and individual actions use separate linear token projections;
reward, termination and truncation use distinct binary embeddings. Learned
embeddings identify summary/action query slots, within-chunk action positions
and trial transition positions, followed by a shared hidden projection.

Training predicts the flow velocity toward Gaussian noise from an interpolated
action target. Gradients pass through the frozen action expert into the
bridge; the expert weights do not update. This is supervised flow matching,
not online PPO optimization of the bridge.

| Bridge setting | Table S1 |
| --- | --- |
| Layers / attention heads | 6 / 8 |
| Hidden / feedforward width | 512 / 1,024 |
| Summary / action queries | 36 / 24 |
| Residual rank | 64 |
| Dropout | 0.1 |
| Output initialization | Zero |
| Sequence / segment length | 48 / 6 transitions |
| Optimizer / maximum LR | AdamW / 1e-4 |
| Weight decay / gradient norm cap | 0.01 / 1.0 |
| LR schedule | 10% warmup, cosine decay |
| Global batch | 64 |

Exact simulation history window assembly, episode counts per training
sequence, checkpoint spacing, student update count and cache detachment
implementation are not fully specified. The real-robot configurations below
must not silently fill these simulation gaps.

Sources: [Section 4.1](https://arxiv.org/html/2610.19517v1#S4.SS1),
[Appendix A.1.1 / Table S1](https://arxiv.org/html/2610.19517v1#A1.SS1.SSS1).

## 5. Simulation evaluation and controls

### 5.1 Main comparison

Evaluate success over **128 randomized instances**. Table 1 distinguishes
six seen and six unseen tasks and reports averages for each group and all
12 tasks. The paper does not identify the bridge episode index or history
budget used to produce the scalar Table 1 scores.

| Method | Training / evaluation contract |
| --- | --- |
| SFT pi0.5 | Common supervised-finetuned base model |
| GRPO | RLinf's implementation; parameter updates; reported training hardware is 8 A100 80GB GPUs |
| RoboTTT | Authors' reimplementation; same six-task history dataset as bridge; reward and termination omitted in training and deployment |
| VLRLA-bridge | Same six-task history dataset; reward and boundary feedback available; accumulated history during deployment |

Reported reference success percentages (SFT / GRPO / RoboTTT / bridge):
seen **48.6 / 59.2 / 57.8 / 66.2**; unseen
**53.3 / 60.8 / 58.6 / 60.6**; overall
**50.9 / 60.0 / 58.2 / 63.4**. These are published values, not independently
reproduced results. Five-seed bands in the following figures do not establish
that Table 1 was averaged over five training seeds.

### 5.2 Backbone, history and transfer experiments

Use **VLRLA-simp**, on `adjust_bottle` and `place_empty_cup`.
Figures 2–4 plot success against episode index **1–25**; shaded regions are
one standard deviation over **five seeds**. Seed roles and exact seed values
are not fully described.

| Experiment | Conditions |
| --- | --- |
| Backbone (Fig. 2) | pi0.5, LingBot-VLA, FastWAM; figure also includes a PPO reference curve |
| Context (Fig. 3) | h=4, h=8, h=8 with history zeroed, h=8 with first-episode history reused |
| Transfer (Fig. 4) | Capture rollout supplies history; warmup loads it for another configuration of the same task; full is the ordinary rollout |

The units and exact implementation of the context parameter h are not
specified sufficiently to equate it with an action horizon or episode count.
The history interventions alter evaluation context; they are not documented
as separately trained reset-policy controls.

Appendix A.4 documents a restricted FastWAM interface: final video hidden
states rather than the native multilayer video KV representations. It yields
120 tokens of width 3,072 from the combined three-camera image, versus the
pi0.5 prefix's 968 tokens of width 2,048. The proposed explanation for its
lower cup-placement score is representational; it is not an isolated
architecture comparison with matching input information.

Sources: [Sections 5.1–5.4](https://arxiv.org/html/2610.19517v1#S5.SS1),
[Appendix A.4](https://arxiv.org/html/2610.19517v1#A1.SS4), PDF pp. 6–8, 21.

## 6. Fruit-placement real-robot protocol

### 6.1 Collection and sequence construction

Six subtasks combine apple/grape/starfruit with bowl/plate. The collection
curriculum increases human control from none, to the initial one-third,
to the initial two-thirds, to the whole episode.

Appendix A.2.2 reports this recorded episode pool:

| Stage | Episodes | Successful episodes | Recorded success percentage |
| --- | ---: | ---: | ---: |
| 0 | 99 | 35 | 35.35 |
| 1 | 95 | 53 | 55.79 |
| 2 | 99 | 80 | 80.81 |
| 3 | 98 | 91 | 92.86 |
| All | 391 | 259 | 66.24 |

Each episode has 12 decisions, with 20 seven-dimensional actions per chunk;
observations contain main RGB (540x960), wrist RGB (480x640), a seven-dimensional
state and an instruction. Total: 4,692 decisions / 93,840 action vectors.
Fruit, destination and success have human annotations. Failures are retained.

Uniformly choose one subtask, then independently sample one matching episode
from each stage, with replacement, and concatenate in stage order. This
produces 48-transition trials; sampled episodes need not share a physical
trajectory or initial scene. No success filtering or pool subsampling.
Use the frozen checkpoint's image/text transforms and quantile statistics;
bridge states/action histories have width seven, while expert targets are
padded to width 32.

Section 6.1 separately mentions approximately 80 teleoperation episodes for
base SFT and 3x20 HITL episodes. Their relationship to the detailed
391-episode pool is not reconciled in the paper.

### 6.2 Bridge fitting

Use the Table S1 bridge structure, with 24 action queries and rank 64;
four episodes x 12 decisions, six-transition segments, eight segments per
update. Table S4 adds:

| Item | Setting |
| --- | --- |
| Objective | Flow velocity MSE |
| Memory detach | 12 chunks, labeled "2 episodes"; conflicts with 12 decisions per episode |
| Optimizer | AdamW; beta=(0.9, 0.999) |
| Maximum LR / decay / clipping | 1e-4 / 0.01 / 1.0 |
| LR schedule | 10% linear warmup, cosine decay |
| Global / per-GPU batch | 64 / 8 |
| Precision | BF16 mixed precision |
| Updates | 1,680 |

### 6.3 Deployment evaluation

Four successive episodes with retained history and fixed weights; first
episode starts without previous interaction history. Seen fruits use 45
runs, 15 per fruit. The unseen transfer uses red pepper and a metal pot,
neither in the training data, with 25 runs. Figure 6 gives approximate seen
success 0.47 for base and 0.42/0.49/0.60/0.62 across episodes; unseen gives
0.24 for base and 0.36/0.36/0.48/0.68. These are figure labels, not raw data.
Vessel balancing, intervention-free reset details and uncertainty estimates
are not supplied.

Sources: [Section 6.1](https://arxiv.org/html/2610.19517v1#S6.SS1),
[Appendix A.2.2 / Tables S3–S4](https://arxiv.org/html/2610.19517v1#A1.SS2.SSS2),
PDF pp. 8–9, 17–18.

## 7. Ambiguous-goal cross-episode memory protocol

### 7.1 Demonstrations and preprocessing

Franka, power adapter, green/yellow plates. The policy receives an instruction
without target color; target correctness determines binary terminal reward.
Source demonstrations were successful under explicit color instructions;
ambiguous-instruction interactions need not be collected for training.

| Item | Appendix A.3 / Table S5 |
| --- | --- |
| Source episodes | 185: 93 green, 92 yellow |
| Training / held-out episodes | 169 (85/84) / 16 (8/8) |
| Frequency / episode duration | 15 Hz / 180 actions = 12 seconds |
| Source size | 33,300 actions, about 37 minutes |
| RGB views | Front, side, wrist |
| State / action width | 15 / 7 |
| Actions | End-effector translation/rotation deltas plus gripper |
| Chunks | Six nonoverlapping chunks, 30 actions each |
| Observation timestamps | 0, 30, 60, 90, 120, 150 |
| Image transform | 480x640 to padded 224x224, aspect ratio retained |
| State transforms | Unit quaternion; sign selected by nonnegative largest-magnitude component |
| Normalization | Frozen checkpoint quantile statistics; matched gripper conventions |

### 7.2 Feedback-dependent history assembly

Sample an initial demonstration and feedback bit b. Its achieved plate is
the hidden target if b=1; the other plate is the target if b=0. Replace
color-specific instructions with the ambiguous instruction. Append K-1
demonstrations reaching that hidden target; relabel terminal rewards to
produce (b,1,...,1).

Create paired trials sharing the initial demonstration and augmentation
randomness but opposite feedback. Their continuation targets consequently
differ. This explicitly teaches dependence on feedback for an otherwise
identical initial history. Split source episodes before dynamic assembly.
The sampling probability of b is not explicitly supplied.

### 7.3 Adapter training and curriculum

| Item | Table S5 |
| --- | --- |
| Frozen base | pi0.5 |
| Bridge layers / hidden / FFN | 6 / 512 / 1,024 |
| Summary / adaptation queries | 72 / 48 |
| Residual rank | 64 |
| Memory detach | 12 chunks = two six-chunk episodes |
| Optimizer | AdamW, beta=(0.9, 0.999), decay 0.01 |
| Maximum LR / updates | 1e-4 / 2,790 |
| Global / per-GPU batch | 64 / 8 trials |
| Precision | BF16 |
| LR schedule | 4% warmup; cosine decay after 60% boundary to zero |
| Short-trial stage | Updates 1–1,256: K=2 |
| Mixed-trial stage | Updates 1,257–2,790: alternate K=2 and K=6 |
| Front image crop / rotation | 95% of each dimension / uniform -5 to +5 degrees |
| Brightness / contrast / saturation multipliers | 0.85–1.15 / 0.8–1.2 / 0.8–1.2 |

The A.3.1 prose instead places the curriculum switch after the first third
of updates. Table S5's switch is about 45% (1,256/2,790). Preserve both;
there is no source justification to silently choose one.

### 7.4 Evaluation

Figure 7 reports six episodes with **44 runs per episode**, fixed model
weights and feedback carried across episodes. Figure-labeled success rates
are 0.341, 0.886, 0.909, 0.955, 0.932, 0.977. This includes a decrease at
episode five; it is not strictly monotonic. The held-out demonstrations are
a validation source split, distinct from these physical evaluation runs.
Live target-color sampling and detailed reset randomization are not specified.

Sources: [Section 6.2](https://arxiv.org/html/2610.19517v1#S6.SS2),
[Appendix A.3 / Table S5](https://arxiv.org/html/2610.19517v1#A1.SS3),
PDF pp. 9, 19–20.

## 8. Reproduction gaps and dataset acquisition

The following require source code, artifacts or author clarification before
claiming an exact reproduction:

- Exact base checkpoints, normalization/input transforms, software commits,
  task configurations and domain-randomization distributions.
- Original SFT motor task exposure; six-task history manifests; valid lengths,
  masks, policy checkpoints and ordering used for student examples.
- Complete simp configuration and simulation bridge training duration,
  trial sampling, validation/selection and memory lifecycle.
- Main-table deployment history budget, episode aggregation, GRPO interaction
  budget and statistical repetitions; full RoboTTT implementation details.
- Interpretation of the Table S2/S4 "12 chunks / 2 episodes" entries, the
  fruit-collection counts and the Table S5 curriculum discrepancy.
- Raw outcome records and histories, rather than plotted averages alone.

The dated release search found no identified official VLRLA code, checkpoint
or custom learning-history dataset. This is a bounded search result, not a
claim that release is impossible or will never occur.

Official RoboTwin demonstrations are available through its
[collection guide](https://robotwin-platform.github.io/doc/usage/collect-data.html)
and [dataset repository](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0).
The audit pins per-task paths, sizes and SHA256 values. Those demonstrations
can support motor preparation; they do not contain the paper's full PPO
learning process or substitute for DSRL learning histories.
