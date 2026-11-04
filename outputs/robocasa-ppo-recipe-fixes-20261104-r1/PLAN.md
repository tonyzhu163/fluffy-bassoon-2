# Requested PPO recipe implementation

The human requested these controls on Nov4. Preserve the completed Stage1
PPO and the approved original GRPO arm; no active source checkout is changed.
Implement named candidates, then qualify on allocated Linux CPU. A changed
recipe is a new experiment, not a silent continuation of Stage1.

| Config suffix | Executed / predicted horizon | Denoising | Noise | Global minibatch | Epochs | Iterations | TRAIN controls | Adam steps/rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ppo_reuse4` | 50 / 50 | 10 | .3 | 256 | 4 | 40 | 3,072,000 | 960 |
| `ppo_long` | 50 / 50 | 10 | .3 | 256 | 1 | 160 | 12,288,000 | 960 |
| `ppo_short_chunk` | 10 / 50 | 5 | .5 | 512 | 1 | 40 | 3,072,000 | 600 |

All use the VLM-prefix critic, ignore_last=true and warmup0. The checked
StructRL pi0.5 config and sparse launcher, RLinf pi0.5 LIBERO-10 and RLinf
pi0 OpenDrawer configurations all use zero critic-only steps.100 warmup
steps were an unvalidated proposal, not a reference requirement. The local
optimizer-rewrite draft was removed without deployment. See the
[reference check](../robocasa-ppo-grpo-meta-stage1-20261104-r1/WARMUP-REFERENCE-CHECK.md).

The short-chunk candidate preserves the checkpoint's network horizon50 and
executes only the first10 controls. Actor/rollout chunk sizes, denoising
counts and likelihood slices must agree; the chunk log ratio includes120
executed coordinates.128episodes ×60decisions yields7,680chunk samples;
batch512 gives15optimizer steps/epoch. With2actor ranks and microbatch16,
native gradient accumulation uses16microbatches per global step. Batch1024
would not divide7,680 and is rejected; it requires a separately sized rollout
layout. No action-repetition or post-hoc slicing of the saved global loss.

Gamma.99/lambda.95 remain per-decision reference coefficients. Shorter
chunks therefore change their physical-time horizon; this combined recipe
cannot isolate cadence, noise, batch or discount effects. Higher noise and
fewer likelihood coordinates are candidates, not a proof of improved learning
or bounded marginal action ratios. Numerical parity and real-device reset/
likelihood checks are required before trusting a GPU run.

The current64GPUh approval covers original Stage1. Benchmark new-recipe
batch capacity, update time and storage before preparing a concrete compute
request; no new training allocation is authorized by this implementation
receipt. H10 creates five times as many decisions per episode and can cost
more despite fewer denoising steps. Do not extend the original PPO to160 or
change original GRPO settings. Keep200GiB PFS floor and full resumable state.
