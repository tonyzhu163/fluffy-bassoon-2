# Offline in-context RL for a frozen VLA: evidence and experiment design

Primary-source scoping review, 7 November 2026. Question: what can our
RoboTwin PPO/GRPO experience teach a history-conditioned controller while
the deployment VLA stays frozen? This complements the broader
[ICRL review](in-context-rl-review.md), the earlier
[steering map](vla-steering-review.md), and the direct
[RoboTwin PPO recipe audit](robotwin-ppo-review.md).

**Assessment:** continue DIT-style weighting and offline IQL alongside source
PPO engineering. Strong PPO improvement is valuable for algorithm
distillation, but is not a prerequisite for every offline ICRL objective.
Informative action/reward coverage, a usable control interface and an honest
online adaptation test are prerequisites. The current corpus supports a
single-task pilot; it cannot yet establish cross-task meta-learning.

For the immediate decision, distinguish **teacher optimization**, **student
training objective**, and **control interface**. PPO/GRPO generate experience;
AD/DPT/DIT/offline value learning determine what the student learns from it;
residual actions, KV modification and proposal ranking determine how the
student controls the frozen motor. Evidence for one choice does not select
the other two. Our recommendation is a matched residual-interface comparison
with trained static and history-imitation controls, alongside source PPO
restoration. This is an experimental recommendation, not an established
RoboTwin result.

## Scope and strength of evidence

The review separates three problems: generating experience, learning from it,
and changing robot behavior at deployment. A successful paper on one does not
automatically solve the others. Core method sections and appendices were read
for AD, DPT, DIT, offline IC-IQL, Vintix II, VLRLA and piRL. Author code was
checked for IQL weighting and the released RLinf configuration. Adjacent
steering methods were checked at the mechanism level. This is neither a
systematic search nor an independent reproduction; external results are
author-reported. Search terms, inspected versions, implementation hashes and
the dated handoff are in the
[source record](../../outputs/robotwin-restoration-20261106-r1/OFFLINE-ICRL-REVIEW.json).

## What the central methods actually learn

| Work and inspected version | Learning signal and relevant evidence | Consequence for this project |
| --- | --- | --- |
| [Algorithm Distillation, Laskin et al., 2022, §3–4](https://arxiv.org/pdf/2210.14215v1) | A causal transformer predicts actions from chronological, across-episode histories of a source RL algorithm. The experiments distinguish this from distilling an expert endpoint and show in-context improvement in discrete environments, including pixels. | Preserve the action from the policy version that actually generated each transition. A one-version corpus can train history-conditioned imitation, but does not instantiate the original improving-process data construction. |
| [Decision-Pretrained Transformer, Lee et al., NeurIPS 2023, §3–5](https://arxiv.org/pdf/2306.14892) | Predict a good/optimal action at a query state from a task-specific interaction context. Context can be suboptimal. Its posterior-sampling interpretation explains exploration without imitating a teacher's chronological update sequence. | It requires action supervision at the actual query observation. Success elsewhere in the corpus is insufficient. A final PPO teacher can supply approximate labels if queried there; its extra supervision and measured quality must be disclosed. |
| [Decision Importance Transformer, Dong et al., ICML 2025](https://proceedings.mlr.press/v267/dong25d.html), [full text §4–5](https://arxiv.org/html/2601.20116v1) | Estimate behavior advantages with context-conditioned values, then use exponentially weighted action likelihood. It avoids DPT's optimal-action labels; experiments include bandits, navigation, Meta-World reach variations and Half-Cheetah velocity variations. | Particularly relevant to mixed-quality experience. Failures can remain useful context while their actions receive low imitation weight. Reliable values and coverage still matter. Continuous robot/VLA transfer remains an experiment. |
| [Yes, Q-learning Helps Offline In-Context RL, Tarasov et al., v5, 2026, §3–4 / Appendix B](https://arxiv.org/pdf/2502.17666v5) | Compares explicit RL objectives with AD across more than 150 datasets, including continuous IC-IQL and IC-TD3+BC. Reports about 30% average improvement on its metrics and stronger robustness with offline regularization; evaluates adaptation curves and normalized AUC. | Strong support for comparing objectives on matched data. Its numerical gains are not RoboTwin success-rate predictions. The shared backbone is trained by value losses; continuous actor gradients are detached from it. Our independent encoders are a different implementation. |
| [Implicit Q-Learning, Kostrikov et al., 2021, §4](https://arxiv.org/html/2110.06169) | Uses an upper expectile value, in-dataset Bellman regression and advantage-weighted policy extraction. Value fitting avoids querying unseen actions. The implementation uses clipped double Q. | A suitable offline objective to adapt to causal history and decoded residual commands. It does not create absent behavior or guarantee good values under a deployment distribution shift. |
| [Vintix II, Polubarov et al., v1, 2026, §3–4](https://arxiv.org/html/2604.05112v1) | Scales DPT with a flow action head: 709.7M transitions, 209 training tasks across ten domains and 46 held-out tasks. Noised demonstrator rollouts are relabeled at their visited query states. Distinguishes online interaction from offline demonstration prompts. | DPT is no longer supported only by small discrete examples. Its oracle-label requirement remains, and its data scale is far beyond our pilot. A flow head is a plausible later capacity comparison, not a substitute for collecting the labels. |
| [Scalable In-Context Q-Learning, Liu et al., ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/9e76870b97621b3153b84782062143ef-Abstract-Conference.html), [v3 §3–4](https://arxiv.org/html/2506.01299v3) | A pretrained dynamics/reward model encodes task prompts; a shared multi-head transformer learns Q, upper-expectile V and advantage-weighted policy extraction. It evaluates discrete and continuous tasks with parameters fixed during interaction. | A close value-based baseline, not a frozen-VLA result. Our separate encoders and raw history do not reproduce its learned task-prompt construction. Any prompt comparison must account for auxiliary world-model training. |
| [State-Action Distillation, Chen and Paternain, TMLR 2025; v3 §4–5](https://arxiv.org/html/2410.19982v3) | Builds query-action labels using random-policy evaluations within a trust horizon. Experiments cover bandits and navigation, including visual Miniworld. Its MDP guarantee assumes that random-policy and optimal Q functions select the same maximizing actions. | A counterexample to requiring a strong trained teacher for every ICRL method. Its query-state resets, action enumeration and return coverage are additional resources; these are not present automatically in our continuous robot logs. |
| [VLRLA, Shi et al., v1, October 2026, §4 / Appendix A](https://arxiv.org/html/2610.19517v1) | Adds cross-episode memory to a VLA. The bridge modifies frozen VLM KV conditioning and trains through the frozen action expert with an action-level flow loss. Its RoboTwin PPO collector uses 25 iterations, four rounds of 256 environments and five optimization epochs. | The closest architectural comparison. An action target can train an internal bridge without a recorded latent command. Frozen expert weights must still permit input gradients during bridge training. The main-table GRPO baseline and PPO history collector have different roles. |

DIT is an ICML **2025** publication even though the inspected arXiv full-text
identifier begins `2601`. Bibliographic year and arXiv upload year should not
be conflated. Vintix II and the latest offline IC-IQL revision also make older
blanket claims that ICRL only works in tiny discrete domains inappropriate;
neither supplies our missing frozen-Motus RoboTwin result.

The additional sources sharpen the teacher-quality distinction. SAD obtains
new labels by evaluating actions at selected query states; it does not turn an
arbitrary fixed failure buffer into expert supervision. S-ICQL moves part of
task inference into learned dynamics/reward prompts, adding a training stage
and an ablation requirement. Its authors also retain a long-horizon prompt
length limitation. Neither result justifies dropping coverage checks or
calling a single-task fit cross-task meta-learning. Read each method's own
versioned specification when comparing architectures: another paper's
related-work description is insufficient implementation evidence.

## Source PPO versus GRPO: what the evidence licenses

The official [RLinf RoboTwin guide](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/robotwin.html)
reports clean `adjust_bottle` pi0.5 success of 85.94% SFT and 96.09% PPO.
Its OpenVLA-OFT GRPO results use randomized settings and another policy family.
Those tables support feasibility; they do not establish a matched pi0.5
PPO-versus-GRPO ranking on our task.

The [released PPO YAML](https://github.com/RLinf/RLinf/blob/77a521298f1d215971246c2d7a423720d9e76652/examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi_pi05.yaml)
uses task-specific SFT, H50, horizon200, group1, 256 environments, four rounds,
global minibatch2048 and five epochs. Our joint Motus phone experiment changes
the initializer, task, cadence and grouping. `max_epochs: 1000` is a ceiling,
not a recovered training duration for the reported result.

[What Can RL Bring to VLA Generalization?](https://papers.nips.cc/paper_files/paper/2025/file/8c0fabe372177d2aded596be2d3b4544-Paper-Conference.pdf),
§4.1, favors PPO over GRPO in its OpenVLA/LoRA manipulation comparison, with
two training seeds. This supports choosing PPO as a starting point; it does
not prove that PPO histories are better supervision for our student.
[piRL v3](https://arxiv.org/html/2510.25889v3) supplies flow-policy likelihood,
noise and critic-design evidence, with several manipulation benchmarks and
single-task studies. These are adjacent optimization results rather than a
joint Motus/RoboTwin reproduction.

The schematic PPO advantage is

\[
\delta_t=r_t+\gamma b_tV(o_{t+1})-V(o_t),\qquad
\widehat A_t=\delta_t+\gamma\lambda c_t\widehat A_{t+1}.
\]

Bootstrap mask \(b_t\) and continuation mask \(c_t\) need not be identical.
The clipped policy surrogate uses the likelihood ratio
\(\rho_t=\exp(\log p_\theta-\log p_{\rm old})\).
For Flow-SDE, the pinned sampler's transition likelihood and denoising/action
reduction must be preserved; this expression does not assert an exact marginal
likelihood for a decoded physical command.

[RLinf-VLA §3.4 / §4.2](https://arxiv.org/html/2510.06710v1) defines GRPO groups
by common task and initial state and assigns a normalized trajectory score to
their actions. The qualified local native formula uses

\[
\widehat A_i=\frac{R_i-\overline R_g}{s_g+10^{-6}},\qquad
s_g^2=\frac{1}{G-1}\sum_{j\in g}(R_j-\overline R_g)^2.
\]

Here the score follows the native masked, undiscounted reward recurrence;
multiple episodes require done-aware handling. All-tied scores give zero
reward advantages. Additional advantage normalization, masking and loss
reduction are part of the recipe, not optional arithmetic details. The local
[independent numerical reference](../../outputs/robotwin-sidney-matched-baselines-20261106-r1/numerical_qualification.py)
checks these choices. A zero reward advantage does not imply every possible
regularizer or optimizer-state contribution vanishes.

**Design inference:** use PPO first and a matched GRPO arm. Match physical
interaction budget, initializer, reward, scene groups, cadence and evaluation.
Report both teacher learning and downstream student adaptation. Do not adopt
the earlier informal “75/25” preference as an evidence-based allocation.

## The controller interface is a separate choice

For the decoded-residual pilot, define the frozen native proposal and command

\[
B_t=\pi_0(o_t,\ell;\epsilon_t),\qquad
\Delta_t\sim\mu_\theta(\cdot\mid H_{<t},\phi(o_t,\ell),B_t),\qquad
A_t=B_t+\Delta_t.
\]

The observed teacher target is \(A_t^{\rm data}-B_t\), paired with the same
observation and frozen proposal. Conditioning on \(B_t\) matters when it is
stochastic. Log the full issued command and separately its executed prefix,
reward, duration, boundaries and next observation. These are our interface
requirements, not a claim that all cited methods use residuals.

| Interface and primary evidence | Fit and limit |
| --- | --- |
| Action residual — [Policy Decorator](https://arxiv.org/html/2412.13630v1) refines fixed imitation policies with a learned residual and controlled exploration. | Directly labelable from decoded teacher actions. Its online refinement results are not reward-conditioned, frozen-student ICRL evidence. Measure residual magnitude and saturation; a broad residual can largely replace the base behavior. |
| Internal KV bridge — VLRLA above | Reuses the action expert more directly. Action supervision can train through it. Advantage-weighted flow matching would be a proposed adaptation, not identical to DIT's weighted log likelihood. |
| Initial noise — [DSRL](https://arxiv.org/html/2506.15799v1) learns in a frozen diffusion/flow policy's latent-noise space. | Useful for aligned collection with the same frozen motor. Noise that generated an action under a weight-updated teacher is not automatically a steering label for the original motor; inversion or another action-level training route needs qualification. |
| Candidate ranking — [V-GPS](https://arxiv.org/html/2410.13816v2) learns an offline value function and reranks frozen-policy action samples. | A diagnostic for proposal coverage and critic usefulness. A history-conditioned version is our extension. Compare the cost of all proposals; selection cannot recover a good action absent from the candidate set. |

[ICRT](https://arxiv.org/abs/2408.15980) conditions robot behavior on
demonstrations, which differs from learning from its own reward feedback.
[RoboTTT](https://research.nvidia.com/labs/gear/robottt/) scales temporal context
using gradient-updated fast weights at inference. It is relevant long-context
robotics evidence; its fast-weight mechanism must be reported separately from
our no-test-time-gradient controller.

## Mapping the published objectives to the implemented pilot

For a common history representation \(h_t\) and residual target \(\Delta_t\),
the proposed comparisons are

\[
L_{\rm imitate}=-\mathbb E\log\mu_\theta(\Delta_t\mid h_t,B_t),\qquad
L_{\rm weighted}=-\mathbb E\left[w_t\log\mu_\theta(\Delta_t\mid h_t,B_t)\right].
\]

Chronological AD, DPT labels, static imitation and DIT-style extraction differ
in supervision or conditioning, even when some loss expressions coincide.
DIT targets behavior advantage; IQL instead alternates upper-expectile value
fitting and in-dataset Bellman Q targets before extracting the actor. For
episode-level values our adaptation uses

\[
L_V=\mathbb E\left[|\tau-\mathbf1_{u<0}|u^2\right],\quad
u=\overline Q(h_t,\Delta_t)-V(h_t),\qquad
L_Q=\mathbb E\left[\left(Q(h_t,\Delta_t)-r_t-\gamma b_tV(h_{t+1})\right)^2\right].
\]

The following differences were found by reading the actual
[decoded learner](../../src/fluffy_bassoon/meta_rl/chunk_value.py) and
[weight helper](../../src/fluffy_bassoon/meta_rl/history_value.py), rather than
assuming an algorithm name establishes equivalence:

| Item | Published/reference choice | Implemented pilot and consequence |
| --- | --- | --- |
| IQL extraction scale | Author [actor code](https://github.com/ikostrikov/implicit_q_learning/blob/master/actor.py) uses `exp(beta * advantage)`; the [MuJoCo config](https://github.com/ikostrikov/implicit_q_learning/blob/master/configs/mujoco_config.py) sets beta3. | Our helper divides by `temperature`, default3: beta is effectively1/3. For advantage1 the uncapped weights are about20.09 versus1.40. This is a different setting, not an equivalent spelling. Reference beta3 corresponds to our temperature1/3. |
| IQL conservatism | Original [critic code](https://github.com/ikostrikov/implicit_q_learning/blob/master/critic.py) uses the minimum of two Q estimators. | One Q model plus an EMA target, with separate actor/Q/V history encoders. Call this an IQL-style adaptation. A double-Q comparison and the shared-backbone IC-IQL architecture remain untested here. |
| DIT value context | The inspected paper supplies past state/action/return labels to its value estimators; it regresses returns and adds Bellman regularization. | Our causal values read shifted physical feedback, with Monte Carlo regression, a conventional Bellman auxiliary, target EMA and bounded weights. Keep the name **DIT-style**. Return labels remain critic supervision, never future-information actor tokens. |
| Actor density | Different papers use categorical, Gaussian or flow policies. | Our actor is a bounded, tanh-transformed diagonal Gaussian over the full32×14 residual with a density Jacobian and fixed radii. Its capacity/support is a hypothesis to test. |
| History and Bellman boundaries | Cross-episode memory and value horizons are distinct design decisions. | The pilot carries memory across episodes but stops Bellman edges at success or the declared H192 timeout. Gamma .99 is per physical chunk decision, not per physics integration step. This values within-episode behavior; it does not explicitly optimize the future benefit of support-episode exploration. |

The temperature discrepancy should be resolved explicitly before calling a
fit an original-IQL reference. An existing pinned contract must not be edited
mid-run. Retain its declared beta1/3 pilot identity or create a fresh,
predeclared beta3 comparison after checking advantage/weight distributions.
Neither setting is known to be superior on our data. Numerical correctness
against our own formulas does not establish reference-method equivalence.

## What our data can and cannot establish

The independently checked
[physical export](../../outputs/robotwin-restoration-20261106-r1/DECODEDHISTORY01-COMPLETE.json)
contains one `place_phone_stand` policy version, 1,024 episodes and 5,958
positive-duration decisions; 186 post-terminal no-ops are excluded. There is
nonzero success coverage. This permits an informative-data pilot, not a
presumption that values or residuals will improve behavior.

Important limitations are concrete:

* The 32 exported streams are ordered by collection round then replica. Every
  policy0 stream contains **four distinct scene seeds**, each with eight
  replicas, according to the
  [case split](../../outputs/robotwin-restoration-20261106-r1/STUDENT-CASE-SPLIT.json).
  Simultaneous replicas have an imposed offline order. This is neither an
  online student learning history nor 32 attempts on one fixed scene.
* Only policy0 is present: no chronological teacher improvement sequence and
  no qualified DPT query labels. Use DIT-style/IQL-style/static-imitation names
  for this pilot. More updates to the same buffer do not add policy versions.
* Student TRAIN/DEV/TEST streams are separated, but their scenes were all
  available to the source teacher as TRAIN. They do not establish teacher-case
  transfer. The separate native DEV panel and held-out task evaluation have
  different roles.
* The [fitting profile](../../outputs/robotwin-restoration-20261106-r1/DECODED-FIT-BOUNDS02.json)
  uses TRAIN feature statistics and joint envelopes, with a global radius
  margin selected on DEV. The entire profile is not TRAIN-only. TEST was not
  used to select that margin; selected-profile TEST coverage is unverified.
* The [Motus card](https://huggingface.co/motus-robotics/pi0.5_robotwin2)
  describes joint SFT on 50 tasks / 27,500 episodes and native H32 delta-joint
  control. Tasks held out from our RL/student training may already be in base
  SFT. Audit exact exposure; do not call them unseen to the pretrained VLA.

The [full-architecture CPU qualification](../../outputs/robotwin-restoration-20261106-r1/CPUDECODEDFIT01.json)
verifies a few real-data updates, strict frozen exports and exact fresh-process
learner/optimizer/sampler/RNG recovery. It does not test a trained student in
the simulator or establish an adaptation gain. The negative small-panel PPO
endpoints and the collector timeout are documented separately in the
[PPO review](robotwin-ppo-review.md#what-actually-happened-in-this-repository).
