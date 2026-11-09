# RoboCasa PPO: literature and released-recipe audit

Search and source inspection: November 2–3, 2026. This review asks which
published results actually demonstrate online RL on RoboCasa, which recipes
can be reconstructed from released code, and what they imply for this
repository's unsuccessful OpenDrawer PPO positive control and planned frozen
velocity/KV steering learners. It reports author results; none of the external
training runs was reproduced here.

## Findings that change our next experiment

1. **A public config is not a demonstrated successful recipe.** RLinf supplies
   RoboCasa and RoboCasa365 examples, but its legacy RoboCasa guide explicitly
   supplies no fixed success-rate table. Our run combined that framework with
   a different model, action cadence, critic placement and short budget. Calling
   it a reproduction of a published RoboCasa PPO result overstates fidelity.
2. **The best released comparison found is StructRL's sparse-reward PPO
   baseline and structured-reward PPO on RoboCasa365.** Its reported π0.5
   results are 39.3% SFT, 41.1% sparse PPO and 45.8% structured PPO across
   16 Composite-Seen tasks. Sparse PPO is therefore a useful control, but the
   paper does not support expecting a large gain from sparse reward alone.
3. **Our usual PPO constants are plausible.** Actor/value learning rates
   5e-6/1e-4, γ=.99, λ=.95, clipping .2, and no reference-KL/entropy term
   appear in relevant implementations. Stronger concerns are temporal
   granularity, exploration, critic inputs, reward/boundary alignment,
   initialization and the amount of genuinely new experience.
4. **Successful base-policy RL does not qualify a frozen steering interface.**
   PPO studies generally update an action expert or its LoRA weights. They do
   not establish that our fixed random 32-D velocity/KV maps are sufficient.
   MiDAS provides relevant frozen-base residual evidence, but uses value-based
   off-policy learning, demonstrations and replay rather than our PPO recipe.
5. **Read launchers and patches, not YAML alone.** StructRL's released π0.5
   launchers override both chunk length and critic placement. Its paper and
   released launcher disagree on the π0.5 chunk length; its code also contains
   a termination-shift patch whose applicability depends on buffer indexing.
   These are reproduction questions to resolve, not fixes to copy blindly.

**Nov 4 update.** The [recipe audit](../../outputs/robocasa-ppo-recipe-audit-20261104-r1/AUDIT.md)
compares the five-task 10M Flow-SDE sources with πRL's ablation tables and
RLinf's shipped π0.5 configs. Our H50 chunk, K10 denoising steps, noise 0.3
and action-expert critic all sit on the losing side of πRL's ablations. Together they make
the 600-coordinate chunk ratio ~8–10× more sensitive than published recipes
(first-update approx-KL .151, clip fraction .262). This revises finding 3
above: the scalar constants are fine, but the Flow-SDE cadence/noise/critic
recipe is not.

Primary starting points: [StructRL paper](https://arxiv.org/html/2609.36352v1),
[released reproduction guide](https://github.com/amazon-science/StructRL/blob/6caba1f9538167e0752a7338663e45aac8df965d/docs/REPRODUCE.md),
[RLinf RoboCasa365 guide](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/robocasa365.html),
and [πRL critic design](https://arxiv.org/html/2510.25889#S4.SS3).

## Search scope and evidence rules

The search included RoboCasa/RoboCasa365 with PPO, GRPO, online RL, flow
policies, residual control, frozen VLAs, reward shaping and policy fine-tuning;
paper titles, author project pages, official repositories, configs and model
cards; arXiv metadata queries; and backward/forward discovery from relevant
papers and RLinf examples. English and Chinese recipe queries were used.
Search queries, candidate dispositions and repository pins are recorded in
[the search record](../../outputs/robocasa-ppo-literature-20261102-r1/SEARCH.md).

This is a comprehensive scoping and implementation review, not a preregistered
systematic review or a claim of complete database coverage. In particular,
arXiv metadata queries miss papers whose abstracts never name the benchmark;
full-text searches and citation chaining supplied important additional papers.
Mirrors, generated paper summaries, surveys and GitHub forks were discovery
aids; the claims below use the primary paper, author code or official docs.

Evidence is separated into: **direct PPO**, **direct RoboCasa RL using another
algorithm**, **adjacent mechanisms**, and **excluded look-alikes**. A result
counts as direct only when RoboCasa robot outcomes, rather than reward-model
accuracy, video prediction or another simulator's success, are reported.
No cross-paper ranking is inferred from unmatched success percentages.

## Benchmark and metric distinctions

| Distinction | Why it changes the interpretation |
| --- | --- |
| Original RoboCasa / 24-task studies vs RoboCasa365 | Different task inventories and distributions. A 24-task average is not the official 50-target-task score. |
| RoboCasa-GR1 vs mobile PandaOmron | Different embodiment, action space and datasets; GR1 tabletop gains cannot be read as an OpenDrawer mobile-manipulation reproduction. |
| Atomic-Seen vs Composite-Seen/Unseen | A single drawer skill and a multi-stage kitchen task have different credit-assignment and exploration problems. |
| `pretrain` vs `target` | Kitchen/object distribution selection is separate from a task's seen/unseen category. Target-task adaptation can be legitimate, but must be declared and evaluated on disjoint instances. |
| Predicted action horizon vs executed chunk length | A model predicting 50 actions may replan after 5, 10, 16 or 50. These settings alter open-loop error, decisions per episode and PPO sample count. |
| First success vs success at the final control | Transient completion may count under one metric and fail under the other. Preserve both readouts. |
| Simulator controls vs PPO samples vs updates | A million primitive controls with H50 produces far fewer decisions than the same controls with H5. Repeated SGD epochs add no new environment evidence. |
| Changed expert vs frozen expert with a steering head | Their action spaces, exploration laws and useful learning rates differ. A successful expert update does not validate a steering head. |

The official repository records a 1.5× horizon increase in RoboCasa 1.0.1.
Version, task horizon and evaluator must therefore accompany every score.
[Official release history](https://github.com/robocasa/robocasa#updates),
[benchmark conventions](https://robocasa.ai/docs/build/html/benchmarking/benchmarking_overview.html).

## Direct PPO evidence

### StructRL: strongest released recipe and sparse-PPO comparator

StructRL evaluates π0.5 and GR00T-N1.5 on 16 RoboCasa365 Composite-Seen
tasks, 100 evaluation episodes per task. Reported π0.5 success is
39.3→41.1 with sparse PPO and 45.8 with structured PPO; GR00T is
38.6→40.2 and 49.1 respectively. Training jointly samples tasks. Verified
subtask predicates, dependency gating and completion pacing replace terminal-only
credit in the structured arm. These are simulator-grounded rewards, not a
validated use of an unconstrained video scorer.

The paper specifies 100 iterations and 512 rollouts per iteration. It is a
large training study, not evidence that a small OpenDrawer pilot must improve.
Its code release is unusually useful, but the reconstruction issues below
prevent treating the released defaults as an already verified exact replica.
[Paper, §4 and Appendix A](https://arxiv.org/html/2609.36352v1),
[code at inspected commit](https://github.com/amazon-science/StructRL/tree/6caba1f9538167e0752a7338663e45aac8df965d).

| Setting | Released π0.5 sparse/structured PPO launcher and inherited config |
| --- | --- |
| Model and initialization | Target Composite-Seen SFT π0.5; action expert trainable, VLM frozen |
| Executed chunk / flow steps | Launcher overrides to 50 / 10; **paper states π0.5 chunk 10** |
| Exploration | Flow-SDE, configured noise .5 |
| Critic | Launcher sets `value_after_vlm=True`, `detach_critic_input=True` |
| New experience per iteration | 256 environments × 2 rollout epochs = 512 rollouts |
| PPO epochs / global minibatch | 1 / 512; microbatch 16 |
| Actor / value LR | Launcher overrides to 5e-6 / 1e-4 |
| γ / λ / policy clip | .99 / .95 / .2 |
| AdamW / norm clip | β=(.9,.95), ε=1e-8, wd=.01 / 1.0 |
| Entropy / reference KL | 0 / 0 |
| Boundary handling | `done_shift.enabled=True` in a patched environment worker |
| Horizon | Common rollout limit 2900; evaluator uses task-specific horizons |

Read the [model config](https://github.com/amazon-science/StructRL/blob/6caba1f9538167e0752a7338663e45aac8df965d/configs/model/pi0_5_robocasa365.yaml),
[PPO config](https://github.com/amazon-science/StructRL/blob/6caba1f9538167e0752a7338663e45aac8df965d/configs/robocasa365_ppo_openpi_pi05_composite_seen.yaml),
[sparse launcher](https://github.com/amazon-science/StructRL/blob/6caba1f9538167e0752a7338663e45aac8df965d/launch/train_pi05_sparse_rl_ppo.sbatch),
[structured launcher](https://github.com/amazon-science/StructRL/blob/6caba1f9538167e0752a7338663e45aac8df965d/launch/train_pi05_structrl_ppo.sbatch)
and [upstream pins](https://github.com/amazon-science/StructRL/blob/6caba1f9538167e0752a7338663e45aac8df965d/docs/INSTALL.md)
together. Bare YAML has different LRs, critic placement and chunk defaults.
The inherited `max_epochs=1000` is a cap, while the paper's experimental
endpoint is 100; a reproduction must set its endpoint explicitly. Evaluation
resets are not seeded in the release, according to its reproduction guide.

### PolicyWeave: PPO can improve task experts, but this is a different control

PolicyWeave studies 18 RoboCasa365 atomic tasks with GR00T N1.5, task-specific
rank-8 LoRA experts, and a shared action normalization/encoder/decoder interface.
Its reported merged policy rises from 64.7% after SFT to 74.1% after expert RL;
joint RL reaches 60.7%. These are merged-expert results, not direct evidence
for frozen expert steering. Evaluation uses 50 episodes/task and RoboCasa
1.0.1 target scenes. PPO uses Flow-SDE, γ=.99, λ=.95, clip .2, two update
epochs, actor/value LR 1e-5/1e-4. Selected individual continuations total
23.74M environment steps; checkpoint selection is separate from final
evaluation. The paper and project were inspected; a runnable author recipe
was not established in this audit.
[Paper, Appendix C](https://arxiv.org/html/2609.33125v1),
[project](https://policyweave.github.io/).

### SEES: PPO improves bottleneck skills inside a hierarchical system

SEES trains family-shared action-head LoRA adapters using restored failure
states and sparse atomic-success predicates. GR00T Composite-Seen performance
is 10.2% for the base, 9.5% with execution-only decomposition, and 13.1%
after three evolution rounds. Thus the comparison includes routing and
termination behavior, not just PPO on the original composite command.
Its PPO recipe uses 32 environments, four rollout epochs, two update epochs,
40 updates per family/round, H16, four flow steps, actor/value LR
6e-6/1e-4, γ=.99, λ=.95, clip .2, and no entropy/KL penalty. The backbone
is frozen but action-head LoRA changes the expert. An author-linked
[checkpoint card](https://huggingface.co/huangtianyu0815/robocasa-n15-subtask-termhead/blob/main/README.md)
names code; the linked Git repository could not be fetched anonymously.
[Paper, Table 1 and Appendix C.4](https://arxiv.org/html/2609.32698v1).

### RLinf examples: implementation references with limited outcome evidence

The [legacy CloseDrawer guide](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/robocasa.html)
uses π0 and explicitly states that no fixed success-rate table is published.
The [benchmark-native OpenDrawer example](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/robocasa365.html)
uses the dataset registry. At inspected RLinf commit
`cc6bd2588f2ef6e6e7e4da62c3fef96526a16cb1`, its
[OpenDrawer YAML](https://github.com/RLinf/RLinf/blob/cc6bd2588f2ef6e6e7e4da62c3fef96526a16cb1/examples/embodiment/config/robocasa365_opendrawer_ppo_openpi.yaml)
has π0, H5, five flow steps, 32×4 rollouts, global batch 2048, four PPO
epochs, training horizon 400, evaluation horizon 750, and `split=pretrain`.
This supplies 10,240 decision slots per iteration. Our H50 variant supplies
1,024. Both use the familiar actor/value LRs and GAE settings. The YAML's
1,000-iteration cap is not a measured convergence requirement.

### Other direct PPO reports with weaker recipe evidence

The [DRILL paper](https://arxiv.org/html/2610.30814v1) reports a RoboCasa
Sim-PPO control at 12.4±1.9%, alongside world-model methods. Appendix D.8
uses 30 simulator episodes per outer update, a recent-600-episode replay
buffer, four PPO epochs, minibatch 64 and LR 2e-6. This is a small-budget
control with replay, not a conventional fresh-on-policy reproduction target.
Its results do not isolate our sparse OpenDrawer or frozen-steering problem.

The [RoboCasa-RL repository](https://github.com/micmia/RoboCasa-RL) supplies
state-policy PPO experiments with substantial hand-shaped rewards, task
wrappers and a coursework report. Its 3M-step examples are useful evidence
that RoboCasa integration exists, but do not establish a comparable
π0.5 PPO positive control. A from-scratch state MLP recipe must not be
transferred wholesale to a pretrained flow VLA.

## Direct RoboCasa RL using other algorithms

| Work | What was actually shown | Relevance and limits |
| --- | --- | --- |
| [Z-1](https://arxiv.org/html/2606.31846v1) | π0.5 GRPO on 24 original RoboCasa tasks; reported mean 67.4→80.6%. Scene-specific SFT, task-specific RL, prefix branching, reward decay and selective joint VLM training. | Positive flow-policy RL evidence, **not PPO**. Evaluation budgets differ: 48 SFT vs 64 RL rollouts/task. Author code/weights for the RL recipe were not located on the inspected paper page. |
| [Simple Recipe Works](https://arxiv.org/html/2603.11653) | π0 on four RoboCasa training tasks: 18.9% initial to 29.5±3.0% sequential-training average; 3,840 episodes/task, horizon 480. | Appendix specifies **GRPO**, LoRA rank 32, group 8, LR 2e-5, global batch 8192. The [released repository](https://github.com/UT-Austin-RobIn/continual-vla-rl/tree/dc1b1c8a7fb630c8d9aaf349376ae5a49b575b4e) exposes LIBERO support; no RoboCasa training config was found in its tracked tree. |
| [MiDAS, revised v3](https://arxiv.org/html/2608.11363v3) | Frozen π0.5 base plus value-based residual adaptation after one demonstration; 89.3±8.7% RoboCasa average over **three** tasks. A fourth, unsolved dishwasher task is excluded from that average. | Closest conceptual evidence for frozen-base action correction. Uses demonstrations, critic/replay and successful-episode reuse; it is **not PPO** and not a validation of our fixed velocity/KV projections. |
| [Adaptive Q-Chunking](https://arxiv.org/html/2605.05544v1) | GR00T N1.6 on 24 RoboCasa-GR1 tasks: 47.6→63.2%, 200 evaluation episodes/task. | Supports learning useful action timing/selection with a critic. The RoboCasa pipeline uses substantial SFT and collected rollouts; it is not mobile π0.5 PPO. |
| [VaDar](https://arxiv.org/html/2610.26629v1) | Off-policy OGPO with subgoal-disaggregated critics on three long-horizon RoboCasa365 tasks, starting from non-VLA BC policies under fixed initial-state conditions. | PPO-style denoising-chain extraction inside an off-policy critic method, rather than ordinary on-policy PPO. Separate frozen-policy best-of-4 selection improves ArrangeTea from 23% to 55%; inference uses subgoal-completion indicators. Relevant critic/selection evidence with different observation and supervision assumptions. |

For Z-1, the published GRPO recipe uses 64 environments×4 rollout epochs,
group size 8, global batch 1024, microbatch 16, AdamW LR 5e-6, clip .2,
and no additional KL regularizer. Prefix chunks and all-success group
handling change the optimization problem. Merely changing our advantage
estimator to GRPO would not reproduce those gains.

For MiDAS, retain the failed fourth task when discussing coverage; 89.3%
is not a four-task average. The [author project](https://minimal-data-adaptation.github.io/)
was inspected, but this review does not establish an executable released
RoboCasa recipe. Its evidence motivates a separate off-policy baseline,
not silently replacing the agreed PPO comparison.

## Audit against our completed OpenDrawer run

The inspected local experiment is the
[Flow-SDE positive-control plan](../../outputs/robocasa-opendrawer-rlinf-sim-ppo-20261102-r1/PLAN.md),
[actual result](../../outputs/robocasa-opendrawer-rlinf-sim-ppo-20261102-r1/RESULT.md)
and [config](../../benchmarks/robocasa/configs/robocasa365_opendrawer_pi05_sim_ppo.yaml).
It completed 18 updates / 2,304 TRAIN episodes / 921,600 TRAIN controls.
TRAIN first-three vs last-three success was 57.55→58.33%. Held-out first
success was 35→43/64, exact paired p=.18493. This is an uncertain positive
evaluation point estimate with a failed declared learning gate, not evidence
that PPO cannot improve this model.

| Item | Our run | External evidence | Assessment |
| --- | --- | --- | --- |
| PPO constants | Actor/value LR 5e-6/1e-4, γ=.99, λ=.95, clip .2, dual clip 3, no entropy/KL | Matches several relevant code/paper recipes | No demonstrated misconfiguration here. Avoid a blind LR sweep as the first response. |
| Critic placement | `value_after_vlm=False` for π0.5 | πRL explicitly uses VLM output for π0.5; StructRL launchers set True | Concrete recipe mismatch worth testing. It is not proof that an expert-feature critic is invalid; VLRLA used a different critic on RoboTwin. |
| Execution cadence | H50, ten flow steps | RLinf OpenDrawer H5/five; SEES H16/four; StructRL paper H10, launcher H50/ten | High-priority controlled choice. SFT horizon 50 need not mean executing all 50 before replanning. Preserve the correct likelihood/action slicing when changing it. |
| Exploration | Flow-SDE noise .3 | StructRL's π0.5 model config .5; VLRLA's .3 comes from RoboTwin | A plausible ablation, not an established explanation. Match a chosen reference end to end rather than mix coefficients across tasks and samplers. |
| Fresh data / update reuse | 128 episodes/update; four PPO epochs; 18 updates | StructRL 512/update, one PPO epoch, 100 reported iterations; SEES 128/update, two epochs, 40/family/round | Our run is a small pilot. Match decisions, valid masks and rollout novelty, not optimizer-step totals or wall time alone. |
| Initialization | Human300 multitask π0.5 | StructRL target Composite-Seen SFT; Z-1 scene-specific SFT; PolicyWeave target experts | The starting policy and available demonstration exposure differ. Stronger reported results cannot be attributed to PPO hyperparameters alone. |
| Sparse reward and boundaries | First-success latch; post-success loss masks | Similar goal, but StructRL adds an explicit termination-shift patch | Our inspected saved batch has correct boundary indexing and exact native return agreement; do not copy that patch into this collector. See the bounded numerical check below. |
| Eval protocol | 64 paired cases, horizon 400, first/final success | Different horizons, trial counts and task sets across papers | Retain our paired inference. A literature percentage is not a matched baseline or a reason to lower the learning gate. |
| Trainable interface | Whole action expert updated | Future main route freezes expert and trains velocity/KV controller | Full-expert learning success would still leave steering controllability and optimization unqualified. |

### Termination/GAE: check the buffer convention before changing anything

StructRL's [RLinf patch](https://github.com/amazon-science/StructRL/blob/6caba1f9538167e0752a7338663e45aac8df965d/patches/rlinf_core.patch)
claims success termination cuts earlier credit and shifts it one chunk later.
Its supplied synthetic check constructs a T+1 done array by appending a final
row. Standard buffers may instead begin with the initial state's done row.
Those conventions are different: `dones[t+1]` is correct when it denotes
the boundary **after action t**. A synthetic test with a shifted input can
produce an apparent bug in a correct recurrence.

An allocated-Linux CPU check used the deployed RLinf commit
`8a1233c3fae6f5905adfda7684c455842b0ecfb8` and a saved 8×64×50 behavior
batch from the earlier Flow-SDE qualification run. It extracted the native
preprocessing/GAE functions without changing the runtime. Recomputed returns
matched all 512 saved chunk returns exactly (maximum absolute error 0).
For a real success at chunk 3, done rows were
`[0,0,0,0,1,1,1,1,1]`, and the loss mask retained chunks 0–3. Setting
values to zero gave returns `[.831910,.884540,.940500,1,0,0,0,0]`.
Injecting reward 9 into every later chunk changed none of the returns
through chunk 3. The success reward therefore propagates backward correctly
and post-terminal rewards cannot cross that boundary in this checked batch.

This rules out the alleged indexing failure for the inspected collector and
batch; it is not a full audit of every rollout, action-level reward mode,
auto-reset behavior or timeout bootstrap. Preserve those distinctions before
generalizing. **Do not apply StructRL's done-shift patch to our collector on
this evidence.** The compact [numerical record](../../outputs/robocasa-ppo-literature-20261102-r1/GAE-CHECK.json)
identifies the batch, source and function hashes.

## Adjacent literature and exclusion decisions

| Source | What it contributes / why it is not a successful RoboCasa PPO recipe |
| --- | --- |
| [πRL](https://arxiv.org/abs/2510.25889) | Essential Flow-SDE likelihood, hybrid ODE/SDE and model-specific critic reference. Reported main benchmarks are LIBERO, ManiSkill, MetaWorld and CALVIN; do not label its scores RoboCasa. |
| [RLinf-VLA](https://arxiv.org/abs/2510.06710) | Framework and PPO practice reference, with reported LIBERO/ManiSkill/RoboTwin results; framework support for RoboCasa is separate from a benchmarked RoboCasa recipe. |
| [ReinFlow](https://arxiv.org/abs/2505.22094), [author code](https://github.com/ReinFlow/ReinFlow) | Tractable stochastic flow-policy RL and released metrics. Original visual manipulation benchmark is RoboMimic, not RoboCasa. |
| [VLRLA](https://arxiv.org/abs/2610.19517) | PPO history generation and frozen-expert KV bridge are relevant, but simulation is RoboTwin; bridge fitting is supervised flow matching, not PPO on a RoboCasa KV head. See the [protocol reconstruction](vlrla-protocol.md). |
| [PolicyTrim](https://arxiv.org/html/2606.22540v1), [author code](https://github.com/INCEPTIONwang/PolicyTrim) | Chunk-length/execution-efficiency mechanism. Original benchmarks are not a direct RoboCasa result; StructRL separately evaluates its GRPO design on RoboCasa365. |
| [Robo-Dopamine](https://arxiv.org/html/2512.23703) | Important reward-shaping and reward-hacking evidence. RoboCasa appears in reward-model data/evaluation; policy-learning simulation results are LIBERO/RoboTwin. Its 95% real-robot result is not RoboCasa PPO. |
| [LLM-Guided Future Hypotheses](https://arxiv.org/html/2605.29864v2) | Reports BC+RL gains on CloseDrawer and OpenSingleDoor; future-conditioning and privileged scene grounding change the setup. An identifiable PPO implementation/hyperparameter recipe was not established in the inspected paper. |
| [OopsieVerse](https://arxiv.org/html/2606.31993v1) | Safety-oriented PPO/latent steering with detailed hyperparameters, across RoboCasa and OmniGibson. Modified tasks, observations and damage rewards; not a π0.5 OpenDrawer sparse-reward positive control. Backend must be traced per experiment before borrowing a score. |
| [LoopWAM](https://arxiv.org/html/2610.18873v2) | Deployment-rollout policy improvement; not an inspected PPO/GAE recipe. |
| [PearlVLA](https://arxiv.org/html/2606.17924v2) | Latent planning/refinement; not an inspected online PPO recipe. |
| [EBRL](https://arxiv.org/html/2609.27547v1) | Executes OpenPI/RoboCasa PPO and GRPO for systems throughput comparisons. Its displayed learning-success comparison is LIBERO/GR00T; a throughput speedup is not a demonstrated RoboCasa learning gain. |
| [FutureWorlds](https://arxiv.org/abs/2610.01019) | RL improves a video/world model; RoboCasa video-prediction metrics do not establish robot policy success. |
| [Soft prompting](https://arxiv.org/abs/2609.11310) | Frozen π0.5 adaptation on three atomic RoboCasa tasks demonstrates placement and overfitting effects, but uses demonstrations rather than PPO. |
| [Official benchmark policy support](https://robocasa.ai/docs/build/html/benchmarking/policy_learning_algorithms.html) | Authoritative SFT/IL and evaluator references, not evidence that benchmark checkpoints were PPO-trained. |

GitHub issue [#1044](https://github.com/RLinf/RLinf/issues/1044) independently
reports oscillating RoboCasa π0.5 success, but generic replies offer no tested
fix or successful reproduction. It is a failure report, not a tuned recipe.
Duplicated RLinf fork READMEs, course advertisements, simulator-asset-only
papers, and evaluation-only VLA leaderboards were excluded from the recipe
evidence.
