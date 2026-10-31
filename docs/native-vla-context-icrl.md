# Reward-driven adaptation inside a VLA's context

Status: prototype implemented in `rl/pi05_context_policy.py` and the bounded
`meta_rl.pi05_context_pilot` driver. The real-model gradient/parity/replay gate
has passed; the bounded end-to-end pilot completed without demonstrated
adaptation or a nonzero PPO gradient. Verified results are tracked in
[EVALUATION.md](../src/fluffy_bassoon/meta_rl/EVALUATION.md).
This is a separate experimental arm alongside the existing frozen-motor CTM
and Transformer-XL bridges. It tests whether the VLA can implement adaptation
through its own attention, without a separate deployment-time learner.

## Question and claims

Can a pretrained VLA learn to explore and improve across attempts on a new
hidden task, using public interaction history and fixed learned parameters
at deployment? The intended contribution is a generalizable learning strategy
inside the VLA. Removing an external module or inserting prompt tokens alone
does not establish that contribution or predict an ICML acceptance.

Distinguish three mechanisms:

| Mechanism | Deployment update | Interpretation |
| --- | --- | --- |
| Direct history conditioning | Append public experience; fixed model and adapter weights | Candidate in-context RL |
| Latent search | Optimize a prompt or its sampling distribution from returns | Test-time optimization with a frozen backbone |
| Learned latent inference | A trained encoder maps experience into a belief | PEARL/MELD-style inference; an external encoder remains a learner |

PEARL learns a posterior encoder and a latent-conditioned actor/critic. MELD
learns a reward-predictive latent state model and a belief-conditioned policy.
Their deployment adaptation is inference, rather than policy-gradient ascent
on an independently optimized latent. DSRL learns an auxiliary policy over a
frozen generator's noise. TTT-VLA optimizes latent prompts with a self-supervised
proxy. VLRLA adds a history-processing module to a VLA. These are relevant
comparisons; this proposal is not a claim of exhaustive novelty.

## Native policy interface

Use the qualified pi0.5 motor, decoder and simulator from the
[native bridge protocol](pi05-live-bridge-ppo.md). Start with the existing
four-attempt, 80-step trial and five-step execution chunk to make the first
comparison interpretable. Each attempt resets the physical scene. The hidden
task persists until trial end; public history resets only at the next trial.

The policy is `pi(a_chunk | current_observation, instruction, H_public)`.
The initial history representation is chronological scalar feedback, embedded
by a small linear input projection into the native prefix. Each record has
the existing eleven public fields: the previous requested seven-dimensional
command, previous latched success, attempt boundary, and public attempt/chunk
clocks. Preserve the one-chunk delay of feedback and never expose future
rewards. The requested command is after native action decoding/preparation and
before the private actuator rotation. The full executed command chunk is kept
in audit artifacts; the eleven-field learner view retains the existing
last-command convention.

This first variant sees current native images/proprioception and the complete
scalar history. It does not see historical images. Report that limitation
explicitly. A later visual-history variant may insert deterministically pooled,
frozen vision-tower features before native language attention; compare it with
a bridge receiving the same additional information. Fix pooling, timestamps,
token limits and eviction before its runs. Do not silently truncate a trial.

Insert history before the native transformer builds its per-layer prefix K/V.
The existing action expert then consumes the resulting prefix through its
normal attention. Concatenating arbitrary values into a final-layer K/V cache
is not equivalent. Audit native attention masks, positions/RoPE, prefix/suffix
visibility and context limits in the pinned RLinf source before implementation.
Older history tokens cannot see future records. The current decision can see
all preceding public records and the current observation.

For the first trainable variant, add LoRA to native prefix attention and train
the feedback projection/type embeddings. Keep base tensors, the vision tower,
action expert and decoder frozen. Zero-initialize LoRA output factors. Use a
null-history bypass to reproduce the native empty-history policy; zero-valued
extra tokens alone do not guarantee parity because they change attention.
At deployment, LoRA is part of the fixed VLA computation; no separate recurrent
actor or learned context updater runs. A training-only value head is allowed.

Native hooks and gradient paths are an implementation gate, not an assumption
that the present RLinf wrapper already supports this interface. Training through
a frozen action expert still requires autograd through its input-dependent
operations; an unconditional `no_grad` wrapper would break prefix learning.

## Tasks and information

Use hidden actuator rotation as the initial interface/controllability task,
reusing the existing training theta and overlay split. That split holds out
rotation values and overlays, not object categories or semantic task families.
Because images/proprioception reveal motion, reward-blind improvement on this
task can be valid system identification. It cannot by itself prove reward-driven
learning.

A reward-essential task is required for the stronger paper claim. A proposed
example has two reachable destinations, with the rewarded destination sampled
once per trial and absent from instruction, pixels and reset metadata. The
initial visible distribution is identical under either reward rule. Success
must be scored against the hidden rule rather than a visible goal marker. Both
destinations need a qualified competent motor before training this arm; the
current single-plate simulator is not assumed to provide that capability.

Disjoint trial/task splits must be defined before training. Later paper-scale
evaluation should test new reward rules or task combinations and an independent
environment/backbone where practical. New scene seeds alone are not a new task
family. The reward-essential simulator and broader evaluations are proposed
scope extensions, not already implemented experiments.

## Training stages

1. **Qualify conditioning and gradients.** Show native empty-history parity,
   correct reward timing and masking, no cross-trial state, nonzero gradients
   through native attention to the history interface, and unchanged frozen
   base tensors. Fit a small labeled controllability problem before a large RL
   run. Task labels may supervise actions during training but never enter the
   policy inputs. Qualify label-policy competence with live rollouts.
2. **Collect immutable public histories.** Record imperfect behavior, failures
   and exploration, with the actual observations, requested actions, rewards
   and boundaries. Keep training-only targets separate. Store raw/preprocessed
   observations or frozen input embeddings sufficient to reconstruct every
   trainable layer, plus preprocessing versions, masks, actions, noise/time,
   labels, task splits and hashes. Do not overwrite existing bridge datasets.
3. **Warm-start action prediction.** Apply native conditional flow matching to
   competent query-action labels with the public prefix. For the existing
   motor, supervise only the five decoded steps and seven active channels;
   leave the three unexecuted model steps outside that loss. Sample fresh flow
   noise/time where the saved observations permit it. Validate on complete
   held-out trials. Action targets never replace history actions.
4. **Optimize complete trials with RL.** Continue from the warm start and
   compare with direct RL from the same base initialization. Optimize expected
   sum of raw success over all attempts, with gamma/lambda one for the initial
   bounded trial. Physical attempt boundaries are observation tokens, not
   return terminals. Bootstrap zero at trial end. Use the pinned runtime's
   supported stochastic flow policy and qualify its likelihood replay before
   PPO; deterministic flow outputs do not supply a tractable action density
   by themselves. Preserve any supervised anchor as an explicit objective term
   and report its coefficient.
5. **Freeze and evaluate.** No parameter, prompt-optimizer or normalization
   updates occur within evaluation trials. The only changing input is public
   history. Keep validation for checkpoint selection separate from final test.

Privileged action labels are permitted during training, as in the existing
DPT-inspired warm start. However, an oracle can prescribe different actions
for identical public histories, causing averaging or task memorization. On a
reward-essential task prefer a teacher conditioned on the same public belief,
or a stochastic target matching that ambiguity. Optimal-action imitation alone
does not train information-seeking exploration; trial-level RL must test it.
Keep behavior-policy mixtures independent of attempt index and include varied
outcomes, so a prescribed improvement schedule is not the learning signal.

## Replay and implementation boundary

The current `make_dpt_record` saves final-layer detached native K/V and suffix
features. The current bridge PPO replays those features without rerunning the
motor. Neither is sufficient to train changing prefix attention: the saved
representations would be stale. Existing histories also omit raw observations.
Reuse the trial/environment contracts and label transformation, not the
cached-feature trainer unchanged.

The new driver must rebuild the prefix under current adapters at every replay,
and recompute the native suffix as required by the chosen stochastic flow
likelihood. Frozen vision inputs may be cached only when no trainable operation
precedes the cache boundary. Save sampled stochastic variables and the exact
behavior-policy log likelihoods. Reproduce those likelihoods before the first
optimizer step and monitor joint KL/clip fraction after updates. Do not replace
the flow-policy density with the current bridge's auxiliary residual density.

Measure simulator steps, teacher calls, trainable parameters, actual memory,
forward/backward cost, seconds per update and decision latency. Native attention
training is likely more expensive than a detached external bridge; parameter
counts alone cannot support an efficiency claim. Tune execution microbatches
without changing saved global batch or objective weighting.

## Latent optimization baseline

First validate a controllable latent interface on training tasks, using native
flow noise or a trained soft-prompt interface. Freeze that interface before
evaluation and count its preparation cost. Failure of arbitrary untrained
prompt tokens is not a strong latent-adaptation baseline.

For a baseline without a learned updater, initialize a small diagonal Gaussian
`q_eta(z)` from a fixed training-selected prior at each new trial. Sample a
latent for one attempt, run the frozen policy, and update distribution parameters
from its return using a score-function estimate
`(R - b) * grad_eta log q_eta(z)` or a predeclared evolution-strategy rule.
The return baseline must not depend on the current latent sample; no simulator
gradient is needed. Keep variance/prior constraints, search dimension, step
size and update frequency fixed from validation. Count every candidate rollout
against the same interaction budget, including failed searches. Sparse identical
returns can supply little information; measure this rather than assuming rapid
adaptation. This is explicit test-time optimization, not fixed-weight ICRL.

## Comparisons, controls and reporting

| Comparison | Purpose |
| --- | --- |
| Unadapted native motor | Starting competence |
| Same internal adapters with no cross-attempt history | Ordinary RL fine-tuning at matched training budget |
| Existing CTM and Transformer-XL bridges | External-learner comparison under matched public information |
| Latent search with a frozen decoder | Explicit optimization comparison at matched interaction budget |
| Public-information structured teacher and privileged oracle, separately named | Exploration reference and competence ceiling |
| Warm start alone, direct trial RL, warm start plus trial RL | Contribution of each training stage |

Evaluate normal history, history wiped between attempts, reward blinding,
reward shuffling within predeclared matched groups, and prior-history swaps
between different hidden tasks while holding the current observation fixed.
Same-checkpoint lesions are mechanism diagnostics; separately trained
reward-blind/no-history controls address distribution shift from lesions.
On the reward-essential task, correct reward history should produce a useful
advantage and conflicting history should alter the task-specific behavior.

Primary outcomes are total trial reward and success at each attempt, including
early exploration cost. Report deterministic and stochastic evaluation when
applicable, complete per-trial results, uncertainty across independent training
seeds and task instances, and gains over controls at matched interactions.
Report wall time/latency separately; more compute must not masquerade as more
sample-efficient learning. Predeclare the final cohort after measuring pilot
variance; small two-lane diagnostics are qualification, not a paper result.

## Decision gates

- **Interface gate:** parity, masks, information timing, usable conditioning,
  gradient/replay correctness and allocated-GPU memory/runtime checks pass.
- **Adaptation gate:** live frozen-weight improvement on unseen trial instances
  exceeds a same-budget no-history policy. Supervised loss alone is insufficient.
- **Reward gate:** informative rewards change adaptation on a task where the
  reward rule is otherwise hidden. Rotation alone cannot pass this gate.
- **Paper gate:** repeatable gains/generalization, credible exploration and a
  useful cost/quality tradeoff over strong internal/external baselines. Merely
  removing the bridge remains an architectural observation.

If the interface is inert, diagnose the attention/gradient path before RL. If
warm-start loss improves but live behavior does not, inspect labels, coverage
and policy distribution shift. If history helps only on seen tasks or reward
lesions have no effect where reward is necessary, narrow the claim rather than
scaling the same result. Resource caps, exact revisions, numerical thresholds
and launch receipts belong in a dated plan, not this method contract.

## Sources

- [PEARL: Efficient Off-Policy Meta-RL via Probabilistic Context Variables](https://proceedings.mlr.press/v97/rakelly19a.html).
- [MELD: Meta-RL from Images via Latent State Models](https://arxiv.org/abs/2010.13957).
- [DSRL: Steering Your Diffusion Policy with Latent Space RL](https://arxiv.org/abs/2506.15799).
- [TTT-VLA: Test-Time Latent Prompt Optimization](https://arxiv.org/abs/2606.03127).
- [VLRLA: Deployment-Time Adaptation through In-Context RL](https://arxiv.org/abs/2610.19517).
- [Existing warm-start contract](ctm-warmstart.md), including privileged-label
  limitations and the distinction between imitation and live adaptation.
