# Retain frozen pi0.5 for in-context RL

The user's final instruction is to keep pi0.5: this project studies in-context
RL with a frozen policy,not VLA fine-tuning. The earlier autoregressive
OpenVLA migration recommendation is withdrawn. No model migration,cadence
training ablation or new action-expert PPO/GRPO job has launched.

## Method and adaptation boundary

- Keep the original pi0.5 VLM and action expert frozen during the ICRL
  history/adapter learning stage and at deployment.
- Train the history model and learned bridge offline with the declared
  action supervision/flow-matching objective through the frozen expert.
  Gradients reaching adapter inputs do not require updating VLA weights.
- At deployment,freeze the history model and bridge too. Adaptation occurs
  through completed interaction history and its derived conditioning/KV
  activations. No deployment optimizer or parameter updates.
- A separate support episode precedes the query episode. New trials clear
  history; query outcomes cannot serve as their own support.
- Historical full-expert PPO/GRPO runs may supply teacher histories or external
  controls. Their gains do not establish ICRL,and this decision does not
  authorize new VLA fine-tuning.

The core criterion is an improvement caused by actual history at fixed
weights,on held-out cases/tasks. Compare own-history with reset-history,
reward-blind and appropriate fixed-policy controls using matched episodes.
A stronger SFT policy or an offline-improved teacher is a different result.

## What the likelihood research establishes

[piRL](https://arxiv.org/html/2510.25889v2) supplies published Flow-Noise and
Flow-SDE PPO formulations for pi0.5. [StructRL](https://amazon-science.github.io/StructRL/)
reports direct RoboCasa365 pi0.5 PPO and GRPO and releases a sparse PPO
launcher. These references answer whether such methods exist; they do not
require the ICRL motor to be fine-tuned.

Original autoregressive [OpenVLA](https://arxiv.org/html/2406.09246v3) has
explicit token probabilities,whereas [OpenVLA-OFT](https://arxiv.org/abs/2502.19645)
uses continuous L1 action regression in its main recipe. That distinction
matters for direct policy-gradient updates,but it is not a reason to replace
the frozen pi0.5 motor in our current project. AD/DPT action supervision and
flow matching through a frozen expert do not require computing its decoded
physical-action marginal likelihood.

If PPO is later chosen for offline training of a stochastic controller,
define its actual sampled action and normalized likelihood explicitly.
With fixed motor and decoder F,a sampled Gaussian steering action z has
ratio q_new(z|H)/q_old(z|H); F is absorbed into the transformed environment,
as in the [DSRL latent-action MDP](https://arxiv.org/html/2506.15799v2#S4.SS1).
If a decoder/KV basis is trained jointly,scoring only its input coefficients
does not account for that changing transformation. That requires a suitable
full policy definition; it does not justify a blanket model switch.
A deterministic bridge trained by supervised flow matching is a separate
legitimate objective.

## Reproduction settings remain separate

The original Human300 SFT evaluator uses prediction50/execution5,ODE10 and
pretrain scenes. The current native SFT evaluation keeps these settings.
The direct StructRL PPO launcher uses C50/K10,while its paper describes C10;
its initializer and joint-task protocol also differ from our checkpoint.
Thus C5 is the original SFT setting,not a universal PPO setting.

Retain the verified native pi0.5 motor. Its complete atomic reproduction
is verified; the composite continuation was intentionally closed after the
user clarified atomic-first scope. Atomic tasks are the current learning
stage,and composites are deferred transfer evaluation.
Do not interpret evaluator correctness,teacher improvement or clean
controller likelihoods as evidence of held-out in-context adaptation.
