"""Opt-in, ephemeral B1 prefix reuse for the frozen pi05_libero motor."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time
import types

import numpy as np


def prefix_with_cache(model, observation):
    """Native prefix equations with both final hidden states and KV returned."""
    import jax.numpy as jnp
    from openpi.models import model as model_module
    from openpi.models.pi0 import make_attn_mask
    observation = model_module.preprocess_observation(None, observation, train=False)
    tokens, mask, ar_mask = model.embed_prefix(observation)
    attention = make_attn_mask(mask, ar_mask)
    positions = jnp.cumsum(mask, axis=1) - 1
    (hidden, _), cache = model.PaliGemma.llm([tokens, None], mask=attention, positions=positions)
    return observation, hidden, mask, cache


def actions_from_cache(model, observation, prefix_mask, cache, noise, *, num_steps=10):
    """Native Pi0.sample_actions suffix loop, using the already computed prefix."""
    import einops
    import jax
    import jax.numpy as jnp
    from openpi.models.pi0 import make_attn_mask
    dt = -1.0 / num_steps
    batch_size = observation.state.shape[0]

    def step(carry):
        x_t, current_time = carry
        tokens, mask, ar_mask, condition = model.embed_suffix(
            observation, x_t, jnp.broadcast_to(current_time, batch_size))
        suffix_attention = make_attn_mask(mask, ar_mask)
        prefix_attention = einops.repeat(prefix_mask, "b p -> b s p", s=tokens.shape[1])
        attention = jnp.concatenate([prefix_attention, suffix_attention], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(mask, axis=-1) - 1
        (prefix_out, suffix_out), _ = model.PaliGemma.llm(
            [None, tokens], mask=attention, positions=positions, kv_cache=cache,
            adarms_cond=[None, condition])
        assert prefix_out is None
        velocity = model.action_out_proj(suffix_out[:, -model.action_horizon:])
        return x_t + dt * velocity, current_time + dt

    def condition(carry):
        return carry[1] >= -dt / 2

    actions, _ = jax.lax.while_loop(condition, step, (noise, 1.0))
    return actions


def _content(value):
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("Motor cache requires string observation/metadata keys")
        return ("dict", tuple((key, _content(value[key])) for key in sorted(value)))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_content(item) for item in value))
    if value is None or isinstance(value, str):
        return (type(value).__name__, value)
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ValueError("Object-valued motor cache identity is unsupported")
    return ("array", array.dtype.str, array.shape,
            hashlib.sha256(np.ascontiguousarray(array).tobytes()).digest())


def _callable_identity(value):
    return (getattr(value, "__self__", None), getattr(value, "__func__", value))


@dataclass
class _Prepared:
    observation: object
    mask: object
    cache: object
    features: np.ndarray


class _NativeBackend:
    def __init__(self, native):
        from openpi.shared import nnx_utils
        self.native = native
        model = native.policy._model
        self.prefix = nnx_utils.module_jit(types.MethodType(prefix_with_cache, model))
        self.suffix = nnx_utils.module_jit(types.MethodType(actions_from_cache, model))

    def prepare(self, obs, instruction):
        from .vla_features import pool_prefix
        observation = self.native.feature_encoder.prepared(obs, instruction)
        prepared, hidden, mask, cache = self.prefix(observation)
        if prepared.state.shape[0] != 1:
            raise ValueError("Cached scalar motor requires B1 observations")
        features = pool_prefix(np.asarray(hidden), np.asarray(mask),
            prepared.tokenized_prompt.shape[-1], np.asarray(obs["state"])[None])[0]
        return _Prepared(prepared, mask, cache, features)

    def split_rng(self, key):
        import jax
        return jax.random.split(key)

    def decode(self, prepared, noise):
        import jax.numpy as jnp
        policy = self.native.policy
        actions = np.asarray(self.suffix(prepared.observation, prepared.mask, prepared.cache,
            jnp.asarray(noise)[None], num_steps=policy._sample_kwargs.get("num_steps", 10)))[0]
        return policy._output_transform(dict(state=np.asarray(prepared.observation.state)[0],
                                             actions=actions))["actions"]

    def attest_first_call(self):
        from .vla_libero import attest_process_gpu
        attest_process_gpu(self.native.output / "motor-binding.json", graphics=False)


class _Features:
    def __init__(self, owner, native):
        self._owner, self._native = owner, native

    def __getattr__(self, name):
        return getattr(self._native, name)

    def __call__(self, obs, instruction):
        return self._owner._features(obs, instruction)

    def qualify(self, obs, instruction):
        self._owner.clear_cache()
        try:
            return self._native.qualify(obs, instruction)
        finally:
            self._owner.clear_cache()

    def diagnose(self, obs, instruction):
        self._owner.clear_cache()
        try:
            return self._native.diagnose(obs, instruction)
        finally:
            self._owner.clear_cache()


class CachedScalarMotor:
    """Preserve native state/API while consuming one matching prepared prefix.

    ``backend`` is a model-free test seam: prepare returns an object with
    ``features``; decode uses no RNG; split_rng and attest_first_call retain
    the native attempt clock and first-inference binding. A present stale
    cache fails before RNG use. With no cache, native infer remains available.
    The caller retains frozen weights; full weight hashes belong to the
    allocated qualification, not a host transfer on every decision.
    """
    def __init__(self, native, *, backend=None):
        if native.horizon != 10 or native.feature_encoder is None:
            raise ValueError("Require the native H10 motor and prefix feature encoder")
        if native.feature_encoder.contract.get("observation_dim") != 4104:
            raise ValueError("Require the unchanged 4104-dimensional feature contract")
        self.native = native
        self._policy, self._model = native.policy, native.policy._model
        self._encoder = native.feature_encoder
        self._cache = None
        self._backend = _NativeBackend(native) if backend is None else backend
        if getattr(self._backend, "native", native) is not native:
            raise ValueError("Prefix backend belongs to another native motor")
        self.feature_encoder = _Features(self, self._encoder)
        self._transforms = tuple(_callable_identity(getattr(self.policy, name, None))
                                 for name in ("_input_transform", "_output_transform"))
        self._configuration = self._configuration_identity()
        self.cached_infer_calls = self.native_infer_calls = 0

    def __getattr__(self, name):
        return getattr(self.native, name)

    def __getstate__(self):
        raise TypeError("Motor KV is ephemeral; checkpoint the native RNG and clear_cache before restore")

    @property
    def policy(self):
        return self.native.policy

    @property
    def calls(self):
        return self.native.calls

    @calls.setter
    def calls(self, value):
        self.native.calls = value

    @property
    def metadata(self):
        return self.native.metadata

    @property
    def horizon(self):
        return self.native.horizon

    @property
    def cache_pending(self):
        return self._cache is not None

    @property
    def execution_metadata(self):
        return dict(kind="cached_native_prefix_scalar_b1_v1", native_contract_unchanged=True,
            prefix_reference_commit="c6f46e8cd80c7cca0877983b659c76ef26b1edbd", execution_microbatch=1,
            cache_capacity=1, cache_checkpointed=False, feature_prefix_reused_for_motor=True,
            immutable_weights_assumed=True, full_weight_hash_required_by_runtime_qualification=True,
            cached_infer_calls=self.cached_infer_calls, native_infer_calls=self.native_infer_calls,
            requires_external_runtime_qualification=True)

    def _configuration_identity(self):
        kwargs = getattr(self.policy, "_sample_kwargs", {})
        if set(kwargs) - {"num_steps"} or kwargs.get("num_steps", 10) != 10:
            raise ValueError("Cached scalar motor requires the original ten-step sampler")
        return _content((self.native.metadata, self._encoder.contract, kwargs))

    def _check_runtime(self):
        transforms = tuple(_callable_identity(getattr(self.policy, name, None))
                           for name in ("_input_transform", "_output_transform"))
        if (self.native.policy is not self._policy or self.policy._model is not self._model
                or self.native.feature_encoder is not self._encoder or self.native.horizon != 10
                or any(a is not b for old, new in zip(self._transforms, transforms)
                       for a, b in zip(old, new))
                or self._configuration_identity() != self._configuration):
            raise ValueError("Frozen motor/runtime identity changed; create a fresh adapter")

    def clear_cache(self):
        self._cache = None
        for name in ("_last_obs", "_last_instruction", "_last_features"):
            if hasattr(self._encoder, name):
                setattr(self._encoder, name, None)

    invalidate_cache = clear_cache

    def restore_motor_rng(self, key):
        self.clear_cache()
        self.policy._rng = key

    def _features(self, obs, instruction):
        started = time.monotonic()
        try:
            self._check_runtime()
            identity = (_content(obs), str(instruction))
            if (self._cache is not None and self._cache[0] == identity
                    and self._cache[2] is self.policy._rng):
                self._encoder.cache_hits += 1
                return self._cache[1].features.copy()
            self.clear_cache()
            prepared = self._backend.prepare(obs, instruction)
            features = np.asarray(prepared.features, np.float32)
            if features.shape != (4104,) or not np.isfinite(features).all():
                raise ValueError("Native pooled feature shape or values changed")
            self._check_runtime()
            if identity != (_content(obs), str(instruction)):
                raise ValueError("Motor observation/instruction changed during prefix preparation")
            prepared.features = features.copy()
            self._cache = (identity, prepared, self.policy._rng)
            self._encoder.calls += 1
            self._encoder.elapsed_seconds += time.monotonic() - started
            return features.copy()
        except BaseException:
            self.clear_cache()
            raise

    def infer(self, obs, instruction, noise):
        try:
            self._check_runtime()
            noise = np.asarray(noise, np.float32)
            if noise.shape != (10, 32) or not np.isfinite(noise).all():
                raise ValueError("Require exact finite native noise shaped [10,32]")
            cached = self._cache
            if cached is None:
                self.native_infer_calls += 1
                return self.native.infer(obs, instruction, noise)
            if cached[0] != (_content(obs), str(instruction)) or cached[2] is not self.policy._rng:
                raise ValueError("Prepared scalar prefix is stale or belongs to another observation/instruction")
            self.clear_cache()
            self.cached_infer_calls += 1
            self.native.calls += 1
            self.policy._rng, _ = self._backend.split_rng(self.policy._rng)
            actions = np.asarray(self._backend.decode(cached[1], noise))
            if self.native.calls == 1:
                self._backend.attest_first_call()
            if actions.shape != (10, 7) or not np.isfinite(actions).all():
                raise ValueError("Unexpected native cached motor output")
            return actions
        finally:
            self.clear_cache()
