# Pretraining scale: primary-source audit

These are reported experimental configurations, not universal minimum data
requirements for in-context RL. An offline context dataset, a complete learning
history, an episode, an environment transition, and an optimizer update are
different units. Repeated training epochs do not increase unique interactions.
Page numbers below are one-based PDF pages.

## Original RL²: online outer-loop training

Duan et al., *RL²: Fast Reinforcement Learning via Slow Reinforcement Learning*,
[arXiv v2](https://arxiv.org/pdf/1611.02779v2), uses TRPO to train a recurrent
policy through online interaction, rather than a fixed offline pretraining
corpus. A trial keeps one sampled MDP fixed across episodes; recurrent state
persists across those episodes and resets between trials (§2.2, pp.2–3).

| Experiment | Reported TRPO batch size | Policy-iteration limit | Location |
| --- | ---: | ---: | --- |
| Bandits | 250,000 | Up to 1,000 | Appendix A.1, Table 1, p.13 |
| Tabular MDPs | 250,000 | Up to 10,000 | Appendix A.2, Table 2, p.13 |
| Visual navigation | 50,000 | Up to 5,000 | Appendix A.3, Table 3, p.14 |

The appendix labels these values “batch size” without explicitly printing its
unit. Multiplying batch size by the iteration limit is not a reported realized
interaction total.

For navigation, training samples from **1,000 maze-layout/target configurations**;
each trial has **two episodes**, each with horizon **at most 250**. Evaluation
uses **1,000 separately generated configurations**, with additional larger-maze
and longer-trial tests (§3.3, p.7).

## DPT: offline contexts and optimal-action labels

Lee et al., *Supervised Pretraining Can Learn In-Context Reinforcement Learning*,
[NeurIPS 2023 camera-ready](https://papers.neurips.cc/paper_files/paper/2023/file/8644b61a9bc87bf7844750a015feb600-Paper-Conference.pdf),
Appendix A.4–A.5, p.21:

| Environment | Collected context datasets | Train/validation division | Calculated training context transitions |
| --- | --- | --- | ---: |
| DarkRoom | 100,000 × 100 transitions | 80,000 contexts across 80 goals; 20,000 across the remaining 20 goals | 8,000,000 |
| Miniworld | 60,000 × 50 transitions | 80% train / 20% validation: 48,000 / 12,000 contexts | 2,400,000 |

Each supervised example combines a context with a query state and an optimal
action label. Miniworld has only **four task identities** and does not require
new task identities at evaluation. The DarkRoom transformer context is **100
steps** (Appendix A.2.2, p.20).

Version difference: [arXiv v1, Appendix A.5](https://arxiv.org/html/2306.14892v1)
reports **40,000 Miniworld contexts**, split **32,000 / 8,000**. The camera-ready
counts above supersede those values for this audit. Basic-bandit pretraining
uses **100,000 samples**, an **80/20 split**, and **300 epochs**; linear-bandit
pretraining uses **1,000,000 samples**, **200-step contexts**, an **80/20 split**,
and **200 epochs** (camera-ready Appendix A.3, pp.20–21).

## Algorithm Distillation: offline learning histories

Laskin et al., *In-Context Reinforcement Learning with Algorithm Distillation*:

- Main experiments report **2,000 learning histories** for the Dark environments
  from A3C with **100 actors**, and **4,000 histories** for Watermaze from DQN
  with **16 actors**. These counts are histories, not individual episodes
  ([arXiv v1](https://arxiv.org/pdf/2210.14215v1), §5, p.7).
- The [ICLR 2023 camera-ready](https://openreview.net/pdf/c985c5523f4d0b869ac3914fad93d499e71fcb5a.pdf),
  Appendix J, p.18, reports a single-stream experiment with **2,048
  DarkKey-to-Door tasks × 2,000 episodes/task**, retaining every tenth episode:
  **200 retained episodes/task**. Calculated totals are **4,096,000 generated
  episodes** and **409,600 retained episodes**. At the **50-step episode
  length** shown in Figure 7, p.8, these correspond to **204.8 million generated**
  and **20.48 million retained transitions**.
- Figure 7, p.8, compares contexts from **20 to 200 transitions**; the paper
  reports its strongest in-context learning with **two to four episodes** in
  context. Appendix I, p.18, finds no held-out in-context learning with **1, 9,
  or 18 training tasks**, some with **37–151**, and best performance with
  **1,212 / 2,424** in that domain.

The camera-ready appendices I/J correspond to H/I in arXiv v1, p.16. These
task-count results do not establish thresholds for robotics.

## Transfer limits

Online outer-loop interactions and offline corpus sizes are separate costs.
Gridworld steps and robot steering chunks also differ in information and
runtime; published counts alone do not determine VLA GPU-hours. A frozen,
pretrained motor already supplies visual and manipulation skills, which can
reduce the steering module's learning burden. That is a transfer rationale,
not a measured reduction established by these papers. Required case diversity,
learning-history quality, and data scale must be measured for the chosen motor
and robotic task distribution.

## Robotics comparisons

[VLRLA](https://arxiv.org/html/2610.19517v1), Appendix A.2.1, reports a
simulation source run with 25 PPO iterations, four rollout rounds, 256 parallel
environments and eight chunk slots per round/environment: 204,800 nominal
chunk slots. Post-termination slots are masked; this is not 204,800 distinct
valid samples. Appendix A.2.2's narrower real-robot curriculum has 391 episodes
across six subtasks, 4,692 chunk decisions, and stage success proportions
35.35/55.79/80.81/92.86%. It trains the bridge for 1,680 updates at global
batch 64. This shows that a focused demonstration can use thousands of
decisions when its histories have structured feedback and behavior changes;
it does not establish broad new-task generalization from that quantity.

[Vintix](https://arxiv.org/html/2501.19400v2), Table 1, reports 670,000 episodes /
67 million transitions for its 45 Meta-World training tasks; the full 87-task
cross-domain bank totals about1.6 million episodes / 222.7 million transitions.
These are state-based control data, not VLA steering chunks or a required
minimum for a frozen-VLA attachment.

The completed RLinf130 technical pilot has 16 clean training episodes /
230 unique steering chunks. Adding the declared Plus source partition yields
40 training episodes /608 chunks, all in one goal family. Ten independent
training trials and 500 repeated optimizer updates per model do not establish
a substantive meta-learning corpus. The new source40 protocol must measure
distinct cases, learning-history quality and scale separately.
