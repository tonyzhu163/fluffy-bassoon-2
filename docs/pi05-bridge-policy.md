# Stochastic flow-residual policy and exact PPO replay

`rl/pi05_bridge_policy.py` wraps the
[native KV feature bridge](pi05-feature-bridge.md) with a stochastic policy and
shared-feature critic. It provides rollout and likelihood replay primitives;
the live collector owns scene splits, perturbations, rewards, PPO updates and
checkpoint boundaries.

## Augmented action distribution

At each native Euler pass, the frozen motor computes its velocity from the
recorded noisy action and current observation. The bridge predicts a residual
mean from its recurrent features, detached native suffix states, noisy actions
and time. The first seven normalized model-action channels receive independent
Gaussian residual samples with this mean and a fixed declared sigma. All eight
native horizon positions participate: later positions can affect earlier ones
through the native action expert's attention. The remaining 25 channels receive
the native conditional velocity with no direct bridge residual or exploration.
Their values can still change indirectly when earlier active-channel noise
changes later native suffix passes.

For four denoising passes and horizon eight, the augmented action has
`4 × 8 × 7 = 224` Gaussian scalars per chunk. The chunk log probability is the
**sum** over all 224 densities, not an average. The complete latent flow
trajectory is the policy action for PPO; its joint density is tractable. It is
not the marginal density of the seven decoded simulator action coordinates.
The frozen decoder and environment may map multiple latent trajectories to the
same physical action without invalidating the joint score-function gradient.

The residual mean uses the bridge's `residual_scale * tanh(...)`; the Gaussian
sample is unbounded. There is no tanh Jacobian, because tanh transforms the mean
rather than the sampled action. The distribution is defined on residual
velocities, so no Euler-step Jacobian is needed. A next-state density would
include a fixed step-size change of variables and would be a different recorded
action convention. Native flow velocities must be float32. Sampling and replay
use the identical float32 computational mean.

The initial noise supplied to the native flow model remains a parameter-free
base distribution. Its density cancels in policy ratios and is omitted.
Sigma is fixed, so the summed Gaussian entropy is constant and cannot train an
exploration scale. A bounded pilot should explicitly save residual scale and
sigma; suggested exploratory settings are residual scale 1.0 and sigma 0.05 or
0.1. These values require live headroom checks. The zero output head preserves
the native motor only in deterministic evaluation; stochastic rollout adds
exploration immediately.

## Collection and replay API

Construct `StochasticPi05BridgePolicy(bridge, noise_scale=..., active_dims=7)`
and keep it in eval mode during both collection and gradient replay. Eval mode
disables stochastic module behavior; it does not disable autograd. The wrapper
owns a zero-initialized linear critic and exposes `config()` for the saved run
contract. A state-dict buffer records sigma, active support and residual scale;
loading a differently configured policy fails.

`sample_chunk(model, prepared_observation, feedback, memory, rng=...)` returns
`(model_actions, next_memory, record)`. The observation must already pass through
the native RLinf input transform and `preprocess_observation`. Decode the result
with the unchanged `model.decode_actions(model_actions, observation.state)`,
then use the native action preparation and simulator path. Native parameters
must be frozen. Collection runs without gradients and advances the recurrent
reader exactly once per observation chunk.

Each detached CPU record contains the selected final-layer native KV, mask,
public feedback and reset mask; all four behavior noisy-action states, native
velocities, suffix states, times, sampled residuals and behavior means; and the
old summed log probability, critic value and bridge features. Storage bytes and
maximum residual magnitudes are reported. Saved bridge features are diagnostic
only and must never replace current recurrent replay.

Call `replay_chunk(record, memory)` chronologically. It returns
`(log_prob, next_memory, diagnostics)` with differentiable `value`, `entropy`,
`features`, per-pass means and per-pass log probabilities. Move the returned
memory to the next chunk without detaching so gradients span complete trials.
The method transfers CPU records to the learner device and recomputes the
reader using current parameters. Saved native suffixes and noisy actions remain
conditioning context from the behavior trajectory. Recomputing native flow
states under changed residuals would score a different augmented action and
produce an incorrect PPO ratio. Frozen native KV and suffixes receive no
gradients. Deterministic evaluation records are rejected as PPO behavior.

## Qualification and practical limits

On allocated Linux run `tests/test_pi05_bridge_policy.py`. Synthetic checks cover
both CTM and Transformer-XL, exact direct Gaussian sums, old/replayed likelihood
parity, deterministic zero-head motor parity, active-channel support, native
gradient isolation, actor/critic updates and gradients through earlier chunks.
The live smoke must check likelihood parity before any optimizer step and
inspect finite gradients and ratios on the first two updates. With zero actor
and critic heads, update one trains their final projections; core gradients
appear after those heads become nonzero.

Summing 224 Gaussian terms makes likelihood ratios sensitive to small mean
changes. Record approximate KL, log-ratio ranges, clipping fraction and actual
residual magnitudes. Lower learning rate or shorter bounded qualification may
be necessary; averaging the log probability would change the policy objective.
CPU offloading bounds persistent rollout storage, but full chronological replay
still retains activation graphs and incurs repeated KV transfers. None of these
unit checks establish adaptation, motor competence under exploration or useful
learning on held-out scenes.
