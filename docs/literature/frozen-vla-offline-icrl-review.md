# Offline in-context RL for a frozen VLA

This review compares the supervision required by published offline objectives, the interfaces that can steer a frozen motor and differences from the repository implementations. [Research design](../PAPER.md) · [Verified findings](../FINDINGS.md)

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

See the [local methods](../METHODS.md) for the implemented data contracts and the [RoboTwin findings](../FINDINGS.md#robotwin) for the measured native outcomes.
