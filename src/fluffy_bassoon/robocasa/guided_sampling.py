"""Steer a frozen native π0.5 toward a target action chunk during flow sampling.

No weight changes. Two mechanisms share one sampler:

* guide: at every denoising step, move the current clean-action estimate a
  fraction ``weight`` of the way to the target on the masked coordinates.
* select: draw several chunks without guidance and keep the nearest one.

``guided_sample_actions`` mirrors ``Pi0.sample_actions`` of the pinned OpenPI
checkout (robocasa-benchmark/openpi ``ca4c6d71``). It is a copy of that loop,
so ``Motor.parity`` must pass against ``policy.infer`` in the job before any
result from this module is used.
"""
import numpy as np


def guide_velocity(v_t, x_t, time, target, mask, weight):
    """Return the velocity after moving the clean estimate toward ``target``.

    OpenPI's convention is t=1 noise, t=0 action, v = noise - action, so the
    clean estimate is ``x_t - time * v_t``. The correction shifts that estimate
    by ``weight * (target - estimate)`` where ``mask`` is one. ``weight=0``
    leaves ``v_t`` unchanged; ``weight=1`` makes the final Euler step land
    exactly on the target for the masked coordinates.
    """
    estimate = x_t - time * v_t
    return v_t - weight * mask * (target - estimate) / time


def guided_sample_actions(model, rng, observation, *, num_steps, noise, target, mask, weight):
    import einops
    import jax
    import jax.numpy as jnp
    from openpi.models import model as _model
    from openpi.models.pi0 import make_attn_mask

    observation = _model.preprocess_observation(None, observation, train=False)
    dt = -1.0 / num_steps
    batch_size = observation.state.shape[0]
    if noise is None:
        noise = jax.random.normal(rng, (batch_size, model.action_horizon, model.action_dim))

    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

    def step(carry):
        x_t, time = carry
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
            observation, x_t, jnp.broadcast_to(time, batch_size))
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_to_suffix = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_to_suffix, suffix_attn_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (prefix_out, suffix_out), _ = model.PaliGemma.llm(
            [None, suffix_tokens], mask=full_attn_mask, positions=positions,
            kv_cache=kv_cache, adarms_cond=[None, adarms_cond])
        assert prefix_out is None
        v_t = model.action_out_proj(suffix_out[:, -model.action_horizon:])
        v_t = guide_velocity(v_t, x_t, time, target, mask, weight)
        return x_t + dt * v_t, time + dt

    def cond(carry):
        return carry[1] >= -dt / 2

    x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
    return x_0


class Motor:
    """A frozen OpenPI policy with guide and select sampling on chosen action coordinates."""

    def __init__(self, policy, norm_stats, *, dims=(0, 1, 2, 6), num_steps=10):
        import jax
        from flax import nnx

        self.policy = policy
        self.dims = tuple(dims)
        model = policy._model
        self.horizon, self.width = int(model.action_horizon), int(model.action_dim)
        actions = norm_stats["actions"]
        self.mean = np.asarray(actions.mean, np.float32)
        self.scale = np.asarray(actions.std, np.float32) + np.float32(1e-6)
        self.coordinate = np.zeros(self.width, np.float32)
        self.coordinate[list(self.dims)] = 1.
        graphdef, self._state = nnx.split(model)

        def run(state, observation, noise, target, mask, weight):
            return guided_sample_actions(
                nnx.merge(graphdef, state), jax.random.key(0), observation,
                num_steps=num_steps, noise=noise, target=target, mask=mask, weight=weight)

        self._run = jax.jit(run)
        self.affine_checked = False

    def encode(self, commands, valid):
        """Map executed [H,12] commands to a model-space target and its mask."""
        commands, valid = np.asarray(commands, np.float32), np.asarray(valid, bool)
        if commands.shape != (self.horizon, 12) or valid.shape != (self.horizon,):
            raise ValueError("Require one [H,12] command window and its [H] validity")
        target = np.zeros((self.horizon, self.width), np.float32)
        for d in self.dims:
            target[:, d] = (commands[:, d] - self.mean[d]) / self.scale[d]
        return target, valid[:, None] * self.coordinate[None]

    def _sample(self, inputs, noise, target, mask, weight):
        import jax
        import jax.numpy as jnp
        from openpi.models import model as _model

        noise = np.asarray(noise, np.float32)
        if noise.ndim == 2:
            noise = noise[None]
        batch = len(noise)
        if noise.shape != (batch, self.horizon, self.width):
            raise ValueError("Require explicit [B,H,D] noise")
        data = self.policy._input_transform(jax.tree.map(lambda x: x, inputs))
        state = np.asarray(data["state"])
        batched = jax.tree.map(lambda x: jnp.repeat(jnp.asarray(x)[None], batch, 0), data)
        x0 = self._run(self._state, _model.Observation.from_dict(batched), jnp.asarray(noise),
                       jnp.asarray(target)[None], jnp.asarray(mask)[None], jnp.float32(weight))
        return np.asarray(x0), state

    def _decode(self, x0, state):
        chunk = np.asarray(self.policy._output_transform(dict(state=state, actions=x0))["actions"])
        if chunk.shape != (self.horizon, 12) or not np.isfinite(chunk).all():
            raise ValueError("Require one finite decoded [H,12] chunk")
        if not self.affine_checked:
            # The target encoding assumes the output transform is this affine map
            # on the steered coordinates. Stop if the checkpoint's transform differs.
            for d in self.dims:
                np.testing.assert_allclose(chunk[:, d], x0[:, d] * self.scale[d] + self.mean[d],
                                           rtol=0, atol=1e-4)
            self.affine_checked = True
        return chunk

    def distance(self, x0, target, mask, steps):
        """RMS model-space gap to the target over the first ``steps`` controls."""
        m = mask[:steps]
        return float(np.sqrt((np.square(x0[:steps] - target[:steps]) * m).sum() / m.sum())) if m.any() else None

    def guide(self, inputs, noise, target, mask, weight, *, steps=5):
        x0, state = self._sample(inputs, noise, target, mask, weight)
        return self._decode(x0[0], state), self.distance(x0[0], target, mask, steps)

    def select(self, inputs, noises, target, mask, *, steps=5):
        """Sample one unguided chunk per noise and keep the nearest to the target."""
        x0, state = self._sample(inputs, noises, target, mask, 0.)
        gaps = [self.distance(x, target, mask, steps) for x in x0]
        best = 0 if gaps[0] is None else int(np.argmin(gaps))
        return self._decode(x0[best], state), gaps[best], gaps

    def parity(self, inputs, noise):
        """Largest decoded-action gap between this sampler at weight zero and the policy's own."""
        zero = np.zeros((self.horizon, self.width), np.float32)
        mine, _ = self.guide(inputs, noise, zero, zero, 0.)
        native = np.asarray(self.policy.infer(inputs, noise=np.asarray(noise, np.float32))["actions"])
        return float(np.abs(mine - native).max())
