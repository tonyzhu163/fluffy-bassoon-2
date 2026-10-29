# METTA paper-based network implementation

This independently implements the network in [METTA v1](https://arxiv.org/html/2610.31656v1),
Sections3.1–3.2 and AppendixB. The authors' implementation is not publicly
available. This code does not establish reproduction of Figure7, its speed,
or its learning performance. No long training run is implied by this change.

## Architecture and memory semantics

- Inputs are state, previous executed action, previous reward and episode
  boundary, embedded into128dimensions. There are two stacked TTT-MLP layers.
- Each fast memory is a two-layer GELU MLP with residual query input and layer
  normalization: `f(u) = u + LN(GELU(u W1 + b1) W2 + b2)`.
- Key/value/query projections are learned slow parameters. The fast weights
  minimize `.5 * ||f(key) - value||²` using differentiable inner SGD. They are
  tensor states rather than mutable model parameters.
- TTT chunks contain128tokens. All writes inside a chunk differentiate the
  chunk-initial fast weights. Each read uses only its causal prefix of writes;
  the final updated memory initializes the next chunk. Chunking does **not**
  detach outer gradients or reuse behavior-policy states.
- Fast weights reset to learned initial values each physical episode. The
  previous episode's final within-episode representation is stored before the
  reset; it remains available to later episodes in the trial. Trial resets
  clear the complete history. First-episode context is its current representation.
- Default retrieval uses attention over completed episode summaries followed
  by `LN(current + retrieved)`. No episode positions are added. EMA with
  alpha.5 and no cross-episode aggregation are explicit alternatives.
- Actor and critic have independent tanh64,64heads and a separate normalized
  current-state input. Gaussian latent actions use learned standard deviations
  bounded to[.1,1.5], initially1.0; actual simulator commands are clipped[-1,1].
  Latent actions and their unsquashed likelihoods are retained for PPO.

Analytical inner derivatives allow online writes in `torch.no_grad()` and
`torch.inference_mode()`. They retain higher-order dependencies when outer
autograd is enabled. Slow parameters are unchanged during rollout/evaluation.
Summary history is an explicit batched tensor with capacity128episodes;
overflow raises instead of discarding history. All recurrent-state entries
have batch as their first dimension, supporting the existing reset/collector API.

Episode boundaries use the wrapper's boundary bit, separately from network
input. Following Algorithms1/2, previous action/reward/done inputs are zeroed
at the beginning of an episode; the old episode's last pre-action representation
becomes its summary. No extra terminal-observation token is synthesized.

## Unspecified choices and differences from the paper's full stack

The paper specifies hidden width128, two layers, GELU residual MLP and
chunk128, but does not fully specify MLP expansion, attention heads, inner-rate
parameterization, initialization or embedding activation. This implementation
uses expansion4, exact GELU, single-head scaled attention with learned Q/K/V,
a linear input embedding, inner MLP weights initialized Normal(0,.02), zero
biases, and learned scalar token rates `sigmoid(Linear(h)) * base_lr / width`.
The rate projection starts at zero and base_lr=1. Layer-norm epsilon is1e-6
inside the MLP and PyTorch's default1e-5 for summary fusion. These choices are
explicit assumptions, not verified author settings. No language-model RoPE,
convolution or extra language-model MLP blocks are silently imported from TTT-LM.

The ML10 adapter retains the existing task pools, native dense reward times.1,
success scoring and running state moments, frozen throughout collection/PPO
and updated after each complete update. It does not implement the paper's
running discounted-return reward scaling or its unspecified online observation
normalizer. It omits public clocks from network input. Parameter matching to
the250kCTM budget is disabled to preserve the paper's width/depth; manifests
record actual parameters and mark `parameter_matching=false`.

## Training adapter and accounting

The entry point is `python -m fluffy_bassoon.meta_rl.metaworld_metta` with
`--benchmark ml10 --core metta` and the common source/output/resource arguments.
It uses the shared persistent-worker collector, success evaluations, full
policy/optimizer/normalizer/sampler/RNG checkpoints and cost counters.

AppendixB.1PPO reconstructs the **entire current prefix** for each chronological
episode segment, under the latest slow parameters. Loss is restricted to the
current episode; gradients pass through earlier summaries and fast updates.
No graph or memory from before an optimizer step is reused. Cached100 TBPTT
is explicitly rejected. Activation checkpointing in32token blocks trades
recomputation for memory without truncating gradients or changing the TTT rule.

For50lanes/K5/H500, ten epochs and ten lanes per minibatch, this schedule means
250optimizer steps per round (20,000over80rounds) and100Mloss samples over
the10Mtraining run; full-prefix
reconstruction additionally processes300Mpositions (before activation
checkpoint recomputation). `forward_samples` counts loss positions and
`reconstructed_prefix_samples` counts actual prefix positions. These differ
from the previous CTM cached100 learner's update schedule. This reference
implementation prioritizes auditable semantics; no published speed is assumed.

## Qualification

`tests/test_metta.py` compares inner writes, causal reads, chunk carries and
outer parameter/input gradients against an independent float64 autograd
implementation of the inner objective. It checks frozen slow parameters,
no-grad/inference-mode updates, episode resets/summary causality, padding and
permutation invariance, EMA/empty history, cross-episode gradients, lesions,
capacity limits, collector likelihood replay, checkpointed/full gradients,
fresh-prefix PPO work counts and exact next-update recovery including Adam/RNG.

Allocated gates additionally exercise the actual width128/two-layer/chunk128
network on real ML10 episodes and a B10/L128 forward+backward capacity check.
These establish implementation behavior and a bounded execution path, not
10Mlearning performance or VLA readiness. All model/tests run in an allocated
Linux job; macOS only edits and parses syntax.
