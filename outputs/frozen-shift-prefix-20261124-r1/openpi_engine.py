"""OpenPI 215abfb engine for fit.py and the prefixed sampler for adapter.py.

Not executed on the authoring VM (no JAX, no OpenPI). Every stock leaf enters each compiled
function as an input; only the prefix is differentiated.
"""
import contextlib

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from openpi.models.model import Observation

import prefix as px


def cast(value, like):
    return jnp.asarray(value).astype(like.dtype)


@contextlib.contextmanager
def injected(cls, prefix):
    """Trace-time replacement of `embed_prefix`. The stock method is restored before the trace returns."""
    stock = cls.embed_prefix

    def embed(self, obs):
        tokens, mask, ar = stock(self, obs)
        return px.inject(jnp.concatenate, cast, tokens, mask, ar, prefix, obs.tokenized_prompt.shape[1])

    cls.embed_prefix = embed
    try:
        yield
    finally:
        cls.embed_prefix = stock


def sampler(policy, prefix):
    """Replacement for `policy._sample_actions` that shares the policy's weight arrays."""
    graph, state = nnx.split(policy._model)
    arrays = {key: jnp.asarray(value) for key, value in prefix.items()}
    compiled = {}

    def call(rng, observation, *, noise=None, **static):
        key = tuple(sorted(static.items()))
        if key not in compiled:
            def run(state, arrays, rng, observation, noise):
                model = nnx.merge(graph, state)
                with injected(type(model), arrays):
                    return model.sample_actions(rng, observation, noise=noise, **static)
            compiled[key] = jax.jit(run)
        return compiled[key](state, arrays, rng, observation, noise)

    return call


class Engine:
    def __init__(self, policy, seed):
        self.policy = policy
        self.graph, self.state = nnx.split(policy._model)
        self.key = jax.random.PRNGKey(seed)
        graph = self.graph

        def loss(arrays, state, key, obs, actions, train):
            model = nnx.merge(graph, state)
            if arrays is None:
                return model.compute_loss(key, obs, actions, train=train).mean()
            with injected(type(model), arrays):
                return model.compute_loss(key, obs, actions, train=train).mean()

        def reference(state, obs):
            model = nnx.merge(graph, state)
            tokens, mask, _ = model.embed_prefix(obs)
            text = obs.tokenized_prompt.shape[1]
            gate = mask[:, -text:, None].astype(jnp.float32)
            return (tokens[:, -text:].astype(jnp.float32) * gate).sum((0, 1)) / gate.sum()

        self._grad = jax.jit(jax.value_and_grad(lambda a, s, k, o, x: loss(a, s, k, o, x, True)))
        self._loss = jax.jit(lambda a, s, k, o, x: loss(a, s, k, o, x, False))
        self._stock = jax.jit(lambda s, k, o, x: loss(None, s, k, o, x, False))
        self._reference = jax.jit(reference)

    def digest(self):
        """Every resident leaf of the policy the harness calls, reread from the device."""
        state = nnx.state(self.policy._model).flat_state()
        return {"/".join(map(str, path)): px.digest(np.asarray(variable.value)) for path, variable in sorted(state.items())}

    def pack(self, row):
        """Stock input transform, including action normalisation and padding."""
        if "packed" not in row:
            values = self.policy._input_transform(dict(row["item"], actions=np.asarray(row["actions"])))
            stacked = jax.tree.map(lambda x: jnp.asarray(x)[None], values)
            row["packed"] = (Observation.from_dict(stacked), stacked["actions"])
        return row["packed"]

    def draw(self, row, draw):
        key = self.key
        for word in (row["index"], *draw):
            key = jax.random.fold_in(key, word)
        return key

    def reference(self, row):
        return np.asarray(self._reference(self.state, self.pack(row)[0]))

    def value_and_grad(self, prefix, row, draw):
        value, grad = self._grad({k: jnp.asarray(v) for k, v in prefix.items()}, self.state,
                                 self.draw(row, draw), *self.pack(row))
        return float(value), {k: np.asarray(v, dtype=np.float32) for k, v in grad.items()}

    def loss(self, prefix, row, draw):
        if prefix is None:
            return float(self._stock(self.state, self.draw(row, draw), *self.pack(row)))
        return float(self._loss({k: jnp.asarray(v) for k, v in prefix.items()}, self.state,
                                self.draw(row, draw), *self.pack(row)))
