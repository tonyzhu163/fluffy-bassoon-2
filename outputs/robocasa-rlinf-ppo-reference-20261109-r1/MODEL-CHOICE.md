# Likelihood clarity and the frozen-VLA project

Research requested November9: verify a pi0.5 PPO formulation for RoboCasa and
consider OpenVLA,prioritizing theoretical clarity. No model switch or new RL
training has been launched. These are recommendations,not an adopted change
to PAPER.md or a claim of demonstrated controller learning.

## What exists

[piRL](https://arxiv.org/html/2510.25889v2) explicitly develops Flow-Noise and
Flow-SDE PPO for pi0 and pi0.5. Its likelihoods describe stochastic denoising
transitions or an augmented denoising trajectory. They must not be described
as a directly available exact marginal density of decoded physical actions.
The hybrid sampler uses one stochastic transition among deterministic ODE
steps. Mathematical assumptions and the actual sampled/replayed variables
must agree; replay equality alone does not establish this entire equivalence.

[StructRL](https://amazon-science.github.io/StructRL/) reports pi0.5 PPO and
GRPO on RoboCasa365 and releases a direct
[sparse PPO launcher](https://github.com/amazon-science/StructRL/blob/6caba1f9538167e0752a7338663e45aac8df965d/launch/train_pi05_sparse_rl_ppo.sbatch).
Its launcher overrides execution to50 controls and10 denoising steps; the
paper describes10 executed controls. Its target-composite SFT initializer and
large joint-task training budget also differ from our Human300 multitask
checkpoint and task-specific pilots. Resolve those discrepancies before
claiming exact reproduction. The C5 verified in the separate native SFT
campaign is its original evaluator setting,not a universal PPO requirement.

## Which policy is easiest to define probabilistically

| Trainable policy | Likelihood | Main qualification |
| --- | --- | --- |
| Original autoregressive OpenVLA | Product of categorical action-token probabilities | Log the actual sampling distribution,token masks and complete sampled action tokens; verify an appropriate RoboCasa checkpoint and embodiment |
| pi0.5 with denoising RL | Gaussian denoising-transition or joint-path probabilities | Specify the augmented process,all optimized random variables,slicing and hybrid sampling assumptions |
| Stochastic controller with a frozen pi0.5 motor and fixed steering map | Explicit controller density over its sampled steering actions | Motor,decoder/basis and any other randomness must remain fixed with respect to the PPO parameters |

Original [OpenVLA](https://arxiv.org/html/2406.09246v3) uses autoregressive
discrete action tokens. [OpenVLA-OFT](https://arxiv.org/abs/2502.19645) instead
uses parallel continuous L1 action regression in its main recipe; OFT does
not automatically supply the original categorical policy likelihood. A
continuous Gaussian head can be defined,but that is an explicit additional
policy choice. The OpenVLA name alone does not settle the likelihood question.

## Recommendation for this project's frozen motor

Keep the verified pi0.5 motor while evaluating an explicit controller policy
q_psi(z|H),where H is the available observation/action/feedback history. A
Gaussian (or a Gaussian followed by a fixed bounded map) gives a tractable
sample density. With a frozen decoder F,the motor executes a=F(o,z),and PPO
uses q_new(z|H)/q_old(z|H). No inverse of F or physical-action Jacobian is
required: z is the action in the transformed environment. This follows the
[DSRL latent-action MDP formulation](https://arxiv.org/html/2506.15799v2#S4.SS1).
It is a probability-model recommendation,not a guarantee that PPO improves.

The fixed-map condition matters. Jointly learning a decoder/KV basis while
using only the probability ratio of its input coefficients would leave an
unaccounted parameter-dependent action transformation. Learn/qualify and
freeze such a map before that PPO phase,or define the policy distribution
directly over the actual perturbations with a valid density. A deterministic
history-to-KV bridge trained by supervised flow matching is a separate
legitimate objective; it should not be relabeled as this Gaussian PPO policy.

The existing paper route currently learns full-expert source histories and
distills them into a frozen-motor bridge. Replacing its source learner with
controller PPO would be a design change. Retain full-expert piRL as a clearly
labeled comparison/source option until that decision is explicit. If direct
VLA-parameter PPO with the simplest likelihood is the central requirement,
prefer an autoregressive token policy such as original OpenVLA,subject to
first obtaining and verifying a competent checkpoint for the benchmark.

Finally,likelihood clarity is distinct from controllability. Previous random
steering maps did not establish useful adaptation. Require real control and
held-out learning evidence; use a fixed stochastic untrained controller as a
baseline and never equate zero Gaussian variance with a regular density.
