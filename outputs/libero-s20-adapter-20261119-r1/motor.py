"""Fit a velocity correction around the frozen native π0.5 action expert."""

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


CAMPAIGN = Path(__file__).resolve().parents[1] / "libero-meta-learning-20261117-r1"
sys.path.insert(0, str(CAMPAIGN))
from reference import CHECKPOINT, observation


HORIZON = 10
ACTION_DIM = 7
FLOW_DIM = 32
WIDTH = 192


def adapter_init(key, features):
    first, second = jax.random.split(key)
    init = jax.nn.initializers.lecun_normal()
    return dict(
        hidden1=init(first, (features, WIDTH)),
        bias1=jnp.zeros(WIDTH),
        hidden2=init(second, (WIDTH, WIDTH)),
        bias2=jnp.zeros(WIDTH),
        out=jnp.zeros((WIDTH, ACTION_DIM)),
        bias_out=jnp.zeros(ACTION_DIM),
    )


def correction(params, features, timing):
    label = jnp.broadcast_to((2 * timing - 1)[:, None, None], (*features.shape[:2], 1))
    features = jnp.concatenate((features, label), axis=-1)
    hidden = jax.nn.silu(features @ params["hidden1"] + params["bias1"])
    hidden = jax.nn.silu(hidden @ params["hidden2"] + params["bias2"])
    return hidden @ params["out"] + params["bias_out"]


def prefix(model, obs):
    obs = openpi_model.preprocess_observation(None, obs, train=False)
    tokens, mask, ar_mask = model.embed_prefix(obs)
    positions = jnp.cumsum(mask, axis=1) - 1
    _, cache = model.PaliGemma.llm([tokens, None], mask=make_attn_mask(mask, ar_mask), positions=positions)
    return obs, mask, cache


def velocity(model, obs, prefix_mask, cache, noisy, time):
    tokens, mask, ar_mask, cond = model.embed_suffix(obs, noisy, time)
    prefix_attn = jnp.broadcast_to(prefix_mask[:, None, :], (*mask.shape, prefix_mask.shape[-1]))
    attn = jnp.concatenate((prefix_attn, make_attn_mask(mask, ar_mask)), axis=-1)
    positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(mask, axis=-1) - 1
    (_, hidden), _ = model.PaliGemma.llm(
        [None, tokens], mask=attn, positions=positions, kv_cache=cache, adarms_cond=[None, cond],
    )
    hidden = hidden[:, -HORIZON:]
    original = model.action_out_proj(hidden)
    hidden = hidden.astype(jnp.float32)
    hidden = hidden * jax.lax.rsqrt(jnp.mean(jnp.square(hidden), axis=-1, keepdims=True) + 1e-6)
    state = jnp.broadcast_to(obs.state[:, None, :8], (*hidden.shape[:2], 8))
    time = jnp.broadcast_to(time[:, None, None], (*hidden.shape[:2], 1))
    position = jnp.broadcast_to(jnp.linspace(0., 1., HORIZON)[None, :, None], time.shape)
    features = jnp.concatenate((hidden, state, noisy[..., :ACTION_DIM], original[..., :ACTION_DIM], time, position), axis=-1)
    return original, features


def teacher_features(model, obs, actions, key):
    obs, mask, cache = prefix(model, obs)
    noise_key, time_key = jax.random.split(key)
    noise = jax.random.normal(noise_key, actions.shape)
    time = jax.random.beta(time_key, 1.5, 1., (len(actions),)) * 0.999 + 0.001
    weight = time[:, None, None]
    noisy = weight * noise + (1 - weight) * actions
    original, features = velocity(model, obs, mask, cache, noisy, time)
    target = (noise - actions - original)[..., :ACTION_DIM]
    return features, target


def sample(model, obs, noise, params, timing):
    obs, mask, cache = prefix(model, obs)
    dt = -0.1

    def denoise(carry):
        noisy, time = carry
        original, features = velocity(model, obs, mask, cache, noisy, jnp.broadcast_to(time, (len(noisy),)))
        updated = original.at[..., :ACTION_DIM].add(correction(params, features, timing))
        return noisy + dt * updated, time + dt

    actions, _ = jax.lax.while_loop(lambda carry: carry[1] >= -dt / 2, denoise, (noise, 1.))
    return actions


class Motor:
    def __init__(self, checkpoint=None):
        if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
            raise RuntimeError("Require the one allocated GPU for OpenPI")
        self.policy = policy_config.create_trained_policy(config.get_config("pi05_libero"), str(CHECKPOINT))
        graph, self.weights = nnx.split(self.policy._model)
        self._teacher = jax.jit(lambda weights, obs, actions, key: teacher_features(nnx.merge(graph, weights), obs, actions, key))
        self._sample = jax.jit(lambda weights, obs, noise, params, timing: sample(nnx.merge(graph, weights), obs, noise, params, timing))
        self.params = None
        if checkpoint is not None:
            saved = serialization.msgpack_restore(Path(checkpoint).read_bytes())
            self.params = jax.tree.map(jnp.asarray, saved["params"])

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

    def proposal(self, raw, prompt, noise, timing=None):
        if timing is None:
            result = self.policy.infer(observation(raw, prompt), noise=noise)["actions"]
        else:
            obs, _ = self.inputs([raw], [prompt])
            actions = self._sample(self.weights, obs, jnp.asarray(noise)[None], self.params, jnp.array([timing]))
            result = self.policy._output_transform(dict(state=np.asarray(obs.state[0]), actions=np.asarray(actions[0])))["actions"]
        if result.shape != (HORIZON, ACTION_DIM) or not np.isfinite(result).all():
            raise ValueError("Invalid π0.5 action proposal")
        return result
