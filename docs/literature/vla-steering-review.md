# Steering a frozen VLA: scoping review

Scoping review, 2026-11-01. Question: what are the known ways to steer or
adapt a frozen pretrained VLA, or a diffusion or flow-matching robot policy,
at deployment without updating its weights? For each family: the mechanism,
whether the VLA stays frozen, and what training labels it needs. The labels
matter most when the steering is learned from histories of PPO runs that
updated the VLA's own weights. The goal is to choose the steering interface
for the in-context meta-learner on a frozen pi0.5 ([PAPER.md](../PAPER.md)).

**Coverage and limits.** This is a scoping map, not a systematic review. It
used 15 web searches on 2026-11-01, one per candidate family plus follow-ups
on specific papers, and read abstracts and search snippets, not full texts.
Paper descriptions here come from abstracts. Several 2026 papers were found
only on alphaxiv or arXiv listings and are marked *(listing only)*.
RoboMonkey's arXiv ID was not confirmed; its CoRL 2025 PMLR page is linked
instead. There was no screening protocol or second reviewer, so the
PRISMA-ScR protocol and reporting checklist are not satisfied and remain
with the research team.

## Answer

There are about ten families, not three. The earlier three are A (direct
actions), B (conditioning residual) and C (noise). Six families give a
learner a continuous per-step output that can be a meta-learner's action,
and all six keep pi0.5 frozen. Three of them can learn directly from
full-weight PPO actions: **direct actions**, **action residuals**, and
**conditioning or velocity residuals** trained through the frozen expert.
**Noise steering** can too, by inverting PPO actions through the frozen flow
decoder to recover noise labels (UniSteer). The other families
(language, activation steering, gradient guidance, sample-and-select) either
give a coarse or discrete control, need a differentiable objective, or use
the meta-learner as a scorer rather than a controller.

## Families

| # | Family | Mechanism | VLA weights | Labels from full-weight PPO actions `a_t` | Representative work |
|---|---|---|---|---|---|
| 1 | **Initial-noise steering** | Learner picks the noise the frozen flow/diffusion sampler starts from | Frozen | Indirect: invert `a_t` through the frozen ODE to a noise target (UniSteer). Inverted noise can sit outside the bounded range DSRL uses; LP-DS reports that unconstrained noise magnitudes degrade decoding | DSRL [2506.15799](https://arxiv.org/abs/2506.15799); LP-DS, a trust-region noise residual `w = ε + Δθ(s)` [2606.01151](https://arxiv.org/abs/2606.01151); UniSteer [2605.10821](https://www.alphaxiv.org/abs/2605.10821) *(listing only)* |
| 2 | **Conditioning / KV / prompt residual** | Learner writes a residual into the frozen model's conditioning: prefix KV cache, soft prompt tokens, or action-expert input tokens | Frozen | Yes: flow-matching loss on `a_t` backpropagated through the frozen action expert | VLRLA-bridge [2610.19517](https://arxiv.org/abs/2610.19517); soft prompts on frozen pi0.5 in RoboCasa (prompt on action-expert input needed; VLM-prefix-only prompt about 1.7%) [2609.11310](https://arxiv.org/abs/2609.11310); prefix/suffix prompt tokens [2512.02902](https://arxiv.org/abs/2512.02902); retrieved past context in prefix K/V, training-free [2603.07647](https://arxiv.org/abs/2603.07647) |
| 3 | **Velocity residual inside the denoiser** | Zero-initialized module reads the frozen prefix KV and adds to, or blends with, the flow velocity | Frozen | Yes: flow-matching loss on `a_t` | Residual corrector [2609.21216](https://arxiv.org/abs/2609.21216); RL²-VLA velocity blending [2607.26991](https://www.alphaxiv.org/abs/2607.26991) *(listing only)* |
| 4 | **Action residual** | Small policy adds a bounded correction to the frozen policy's action | Frozen | Yes: target `a_t − base action`. The base action depends on its sampled noise, so record that noise | ResiP [2407.16677](https://arxiv.org/abs/2407.16677); Policy Decorator [2412.13630](https://arxiv.org/abs/2412.13630); object-centric residual on a frozen VLA [2606.18953](https://huggingface.co/papers/2606.18953); value learning in residual RL [2602.10539](https://arxiv.org/abs/2602.10539) |
| 5 | **Direct action head** | Separate head outputs actions from frozen VLA features; the action expert is bypassed | VLM frozen, expert unused | Yes: MSE or flow loss on `a_t` | VLRLA-simp [2610.19517](https://arxiv.org/abs/2610.19517) |
| 6 | **Observation-latent perturbation** | Learner perturbs a bottleneck latent between the encoder and the frozen action generator | Frozen online, but the bottleneck must be added during offline training | Not directly; trained with RL on the latent | ZPRL [2605.19919](https://arxiv.org/abs/2605.19919) |
| 7 | **Language / instruction steering** | A policy rewrites or chooses the instruction given to the frozen VLA | Frozen | No action labels; needs search or RL over instructions | Language feedback policy [2606.12299](https://arxiv.org/abs/2606.12299); VLA Grounder [2607.04517](https://www.alphaxiv.org/abs/2607.04517) *(listing only)*; SteerVLA, driving [2602.08440](https://arxiv.org/abs/2602.08440) |
| 8 | **Activation steering** | Add or gate directions in hidden activations | Frozen | No; directions come from interpretability or sparse autoencoders | Häon et al., OpenVLA and pi0 [2509.00328](https://arxiv.org/abs/2509.00328); SAE features on VLAs [2603.19183](https://arxiv.org/abs/2603.19183); COAST [2605.17144](https://www.alphaxiv.org/abs/2605.17144) *(listing only)* |
| 9 | **Gradient guidance during sampling** | Gradient of an external objective guides each denoising step | Frozen | No; needs a differentiable objective (dynamics model, temporal logic, VLM reward) | DynaGuide [2506.13922](https://arxiv.org/abs/2506.13922); temporal-logic guidance [2606.22729](https://arxiv.org/abs/2606.22729); VLS [2602.03973](https://arxiv.org/abs/2602.03973) |
| 10 | **Sample and select** | Draw K action chunks from the frozen policy and pick one with a value function, verifier or world model | Frozen | Scores, not actions: a value target (PPO's critic) or which sample is closest to `a_t` | V-GPS [2410.13816](https://arxiv.org/abs/2410.13816); RoboMonkey [CoRL 2025](https://proceedings.mlr.press/v305/kwok25a.html); bidirectional decoding [2408.17355](https://arxiv.org/abs/2408.17355); FOREWARN [2502.01828](https://arxiv.org/abs/2502.01828) |

Not steering, but related: PA-RL [2412.06685](https://arxiv.org/abs/2412.06685)
samples and ranks like family 10, then distills into the policy weights.
In-context imitation ([ICRT 2408.15980](https://arxiv.org/abs/2408.15980),
[Instant Policy 2411.12633](https://arxiv.org/abs/2411.12633)) trains the
whole policy to read prompts, so it is not steering of an existing VLA.

## Fit for this project's meta-learner

The meta-learner is a plain causal decoder that must output one continuous
control per step, keep pi0.5 frozen, and learn from Flora-rewarded PPO
histories.

| Family | Fit | Why |
|---|---|---|
| 1 Noise | Good, already built | Low-dimensional and bounded. Works with frozen-motor PPO or DSRL directly, and with full-weight PPO through inversion, with the out-of-range risk |
| 2 Conditioning residual | Good | Natural for full-weight PPO labels. Matches VLRLA. A soft-prompt result exists on frozen pi0.5 in RoboCasa. The earlier RoboTwin KV-head attempt here scored 0/5 |
| 3 Velocity residual | Good | Same labels as 2. Acts in the denoiser rather than the conditioning |
| 4 Action residual | Good, simplest | Direct labels, small output. Bypasses nothing. Needs the base sample's noise recorded |
| 5 Direct actions | Possible | Simple labels, but discards pi0.5's action expert, which weakens the frozen-VLA story |
| 6 Latent perturbation | Poor | Needs a bottleneck trained into the base policy |
| 7 Language | Poor for this question | Discrete and slow; a different research question |
| 8 Activation | Poor | Few known directions; not a learned per-step control |
| 9 Guidance | Poor | Flora is a 27B video model, not a cheap differentiable objective |
| 10 Sample and select | Possible as a complement | The meta-learner would output scores; labels come from a critic, not from actions |

The label rule from this review: **the meta-learner's output must be
something the PPO histories can label.** Frozen-motor PPO or DSRL labels
noise (family 1) directly. Full-weight PPO labels families 2–5 directly, and
family 1 only through inversion.

## References

All links are in the tables above. Primary arXiv abstract pages were used
where found; *(listing only)* marks entries seen only on alphaxiv or similar
listings.
