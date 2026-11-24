"""Run-local interventions on the pinned JAX Pi0.action_out_proj."""
from types import MethodType

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from openpi.shared import nnx_utils

CONDITIONS = ("original", "policy_only", "adapter_only", "joint")
HORIZON, ACTION_DIM, HISTORY_DIM, RANK, BOTTLENECK = 10, 32, 52, 4, 16
# Four [command7, delta_xyz3, delta_width, width, valid] slots. Use centimetres.
UNITS = jnp.tile(jnp.array([1.] * 7 + [100.] * 5 + [1.], dtype=jnp.float32), 4)


def initial_params(width, condition, seed=20261124):
    if condition not in CONDITIONS:
        raise ValueError(condition)
    key_w, key_h = jax.random.split(jax.random.key(seed))
    params = {}
    if condition in ("policy_only", "joint"):
        params["left"] = jax.random.normal(key_w, (width, RANK)) / np.sqrt(width)
        params["right"] = jnp.zeros((RANK, ACTION_DIM), jnp.float32)
    if condition in ("adapter_only", "joint"):
        params["history_in"] = jax.random.normal(key_h, (HISTORY_DIM, BOTTLENECK)) / np.sqrt(HISTORY_DIM)
        params["history_out"] = jnp.zeros((BOTTLENECK, HORIZON * ACTION_DIM), jnp.float32)
    return params


def full_params(params, template):
    """Inactive blocks are identically zero; dynamic inputs allow one live executable."""
    return {k: params[k] if k in params else jnp.zeros_like(v) for k, v in template.items()}


def effective_kernel(base, params):
    if "left" not in params:
        return base
    update = jnp.matmul(params["left"], params["right"], precision=jax.lax.Precision.HIGHEST)
    return (base.astype(jnp.float32) + update).astype(base.dtype)


def install(layer, params, history):
    """Change the actual Linear kernel on the trace-local module, preserving native precision and bias."""
    layer.kernel.value = effective_kernel(layer.kernel.value, params)
    if "history_in" not in params:
        return layer
    z = jnp.tanh((history * UNITS) @ params["history_in"])
    offset = (z @ params["history_out"]).reshape(-1, HORIZON, ACTION_DIM)

    def project(hidden):
        native = layer(hidden)
        return (native.astype(jnp.float32) + offset).astype(native.dtype)

    return project


def cached_forward(layer):
    graph, state = nnx.split(layer)

    def predict(params, hidden, history):
        return install(nnx.merge(graph, state), params, history)(hidden)

    return predict


def bind(model, function, **jit_kwargs):
    return nnx_utils.module_jit(MethodType(function, model), **jit_kwargs)


def capture(model, rng, observation, actions, *, train):
    """Trace the stock compute_loss itself. Only observe its last layer input."""
    projection = model.action_out_proj
    features = []

    def capture_input(hidden):
        features.append(hidden)
        return projection(hidden)

    model.action_out_proj = capture_input
    native_losses = model.compute_loss(rng, observation, actions, train=train)
    hidden, = features
    # Pinned Pi0.compute_loss uses the second of three subkeys for epsilon.
    _, noise_key, _ = jax.random.split(rng, 3)
    target = jax.random.normal(noise_key, actions.shape) - actions
    return hidden, target, native_losses


def live_sample(model, rng, observation, noise, params, history):
    model.action_out_proj = install(model.action_out_proj, params, history)
    return model.sample_actions(rng, observation, noise=noise)


def live_loss(model, rng, observation, actions, params, history, *, train):
    model.action_out_proj = install(model.action_out_proj, params, history)
    return model.compute_loss(rng, observation, actions, train=train)


def token_loss(pred, target):
    return jnp.mean(jnp.square(pred - target), axis=-1)
