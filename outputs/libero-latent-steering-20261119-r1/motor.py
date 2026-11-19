"""Frozen JAX OpenPI π0.5 motor with continuous-latent velocity steering and query embedding."""

from pathlib import Path
import sys

from flax import nnx, serialization
import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import model as openpi_model
from openpi.models.pi0 import make_attn_mask
from openpi.policies import policy_config
from openpi.training import config

from fluffy_bassoon.libero.latent_adapter import AdapterConfig, apply_latent_adapter

CAMPAIGN = Path(__file__).resolve().parents[1] / "libero-meta-learning-20261117-r1"
sys.path.insert(0, str(CAMPAIGN))
from reference import CHECKPOINT, observation


HORIZON = 10
ACTION_DIM = 7
FLOW_DIM = 32


def prefix_with_hidden(model, obs):
    """Run frozen VLM prefix forward and return preprocessed observation, hidden, mask, and KV cache."""
    obs = openpi_model.preprocess_observation(None, obs, train=False)
    tokens, mask, ar_mask = model.embed_prefix(obs)
    positions = jnp.cumsum(mask, axis=1) - 1
    (hidden, _), cache = model.PaliGemma.llm(
        [tokens, None], mask=make_attn_mask(mask, ar_mask), positions=positions
    )
    return obs, hidden, mask, cache


def velocity(model, obs, prefix_mask, cache, noisy, time):
    """Run one frozen suffix step and extract normalized action-expert conditioning features."""
    tokens, mask, ar_mask, cond = model.embed_suffix(obs, noisy, time)
    prefix_attn = jnp.broadcast_to(prefix_mask[:, None, :], (*mask.shape, prefix_mask.shape[-1]))
    attn = jnp.concatenate((prefix_attn, make_attn_mask(mask, ar_mask)), axis=-1)
    positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(mask, axis=-1) - 1
    (_, hidden), _ = model.PaliGemma.llm(
        [None, tokens], mask=attn, positions=positions, kv_cache=cache, adarms_cond=[None, cond]
    )
    hidden = hidden[:, -HORIZON:]
    original = model.action_out_proj(hidden)
    hidden = hidden.astype(jnp.float32)
    hidden = hidden * jax.lax.rsqrt(jnp.mean(jnp.square(hidden), axis=-1, keepdims=True) + 1e-6)
    state = jnp.broadcast_to(obs.state[:, None, :8], (*hidden.shape[:2], 8))
    time = jnp.broadcast_to(time[:, None, None], (*hidden.shape[:2], 1))
    position = jnp.broadcast_to(jnp.linspace(0.0, 1.0, HORIZON)[None, :, None], time.shape)
    features = jnp.concatenate(
        (hidden, state, noisy[..., :ACTION_DIM], original[..., :ACTION_DIM], time, position), axis=-1
    )
    return original, features


def query_features_from_prefix(obs, hidden, mask):
    """Build the 2056-d [state[:8], rms_norm(mean_pooled_prefix_hidden)] vector."""
    hidden_f32 = hidden.astype(jnp.float32)
    masked = jnp.where(mask[..., None], hidden_f32, 0.0)
    pooled = jnp.sum(masked, axis=1) / jnp.maximum(jnp.sum(mask, axis=1, keepdims=True), 1.0)
    pooled = pooled * jax.lax.rsqrt(jnp.mean(jnp.square(pooled), axis=-1, keepdims=True) + 1e-6)
    return jnp.concatenate((obs.state[:, :8].astype(jnp.float32), pooled), axis=-1)


def teacher_features(model, obs, actions, key):
    obs, _, mask, cache = prefix_with_hidden(model, obs)
    noise_key, time_key = jax.random.split(key)
    noise = jax.random.normal(noise_key, actions.shape)
    time = jax.random.beta(time_key, 1.5, 1.0, (len(actions),)) * 0.999 + 0.001
    weight = time[:, None, None]
    noisy = weight * noise + (1.0 - weight) * actions
    original, features = velocity(model, obs, mask, cache, noisy, time)
    target = (noise - actions - original)[..., :ACTION_DIM]
    return features, target


def sample_with_latent(model, obs, noise, params, latent, active):
    obs, _, mask, cache = prefix_with_hidden(model, obs)
    dt = -0.1

    def denoise(carry):
        noisy, time = carry
        original, features = velocity(model, obs, mask, cache, noisy, jnp.broadcast_to(time, (len(noisy),)))
        base_step = noisy + dt * original
        update = dt * apply_latent_adapter(params, features, latent, active=active)
        return base_step.at[..., :ACTION_DIM].add(update), time + dt

    actions, _ = jax.lax.while_loop(lambda carry: carry[1] >= -dt / 2, denoise, (noise, 1.0))
    return actions


class LatentMotor:
    """Frozen OpenPI π0.5 backbone with pluggable continuous-latent velocity adapter."""

    def __init__(self, checkpoint=None):
        if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
            raise RuntimeError("Require one allocated GPU for OpenPI")
        self.policy = policy_config.create_trained_policy(config.get_config("pi05_libero"), str(CHECKPOINT))
        graph, self.weights = nnx.split(self.policy._model)
        self._teacher = jax.jit(
            lambda weights, obs, actions, key: teacher_features(nnx.merge(graph, weights), obs, actions, key)
        )
        self._sample = jax.jit(
            lambda weights, obs, noise, params, latent, active: sample_with_latent(
                nnx.merge(graph, weights), obs, noise, params, latent, active
            )
        )
        self._embed = jax.jit(
            lambda weights, obs: query_features_from_prefix(*prefix_with_hidden(nnx.merge(graph, weights), obs)[:3])
        )
        self.params = None
        self.adapter_config = None
        self.cases = None
        if checkpoint is not None:
            saved = serialization.msgpack_restore(Path(checkpoint).read_bytes())
            self.params = jax.tree.map(jnp.asarray, saved["params"])
            self.adapter_config = AdapterConfig(**saved["config"])
            self.cases = tuple(saved["cases"])

    def inputs(self, raws, prompts, targets=None):
        items = []
        for row, (raw, prompt) in enumerate(zip(raws, prompts, strict=True)):
            item = observation(raw, prompt)
            if targets is not None:
                item["actions"] = targets[row]
            items.append(self.policy._input_transform(item))
        inputs = jax.tree.map(lambda *values: jnp.asarray(np.stack(values)), *items)
        return openpi_model.Observation.from_dict(inputs), inputs.get("actions")

    def training_features(self, batch, key):
        obs, actions = self.inputs(batch["raw"], batch["prompt"], batch["actions"])
        if actions.shape != (len(obs.state), HORIZON, FLOW_DIM):
            raise ValueError(f"Invalid OpenPI target shape {actions.shape}")
        return self._teacher(self.weights, obs, actions, key)

    def query_embedding(self, raw, prompt):
        obs, _ = self.inputs([raw], [prompt])
        emb = np.asarray(self._embed(self.weights, obs)[0], dtype=np.float32)
        if emb.shape != (2056,) or not np.isfinite(emb).all():
            raise ValueError(f"Invalid query embedding shape or nonfinite values: {emb.shape}")
        return emb

    def condition_latent(self, condition_index):
        return np.asarray(self.params["condition_table"][condition_index], dtype=np.float32)

    def proposal(self, raw, prompt, noise, latent=None, active=True):
        if latent is None:
            result = self.policy.infer(observation(raw, prompt), noise=noise)["actions"]
        else:
            if self.params is None:
                raise RuntimeError("Load or attach adapter params before requesting steered proposals")
            obs, _ = self.inputs([raw], [prompt])
            latent_arr = jnp.asarray(latent, dtype=jnp.float32).reshape(1, -1)
            active_arr = jnp.array([bool(active)], dtype=bool)
            actions = self._sample(self.weights, obs, jnp.asarray(noise)[None], self.params, latent_arr, active_arr)
            result = self.policy._output_transform(
                dict(state=np.asarray(obs.state[0]), actions=np.asarray(actions[0]))
            )["actions"]
        if result.shape != (HORIZON, ACTION_DIM) or not np.isfinite(result).all():
            raise ValueError("Invalid π0.5 action proposal")
        return result
