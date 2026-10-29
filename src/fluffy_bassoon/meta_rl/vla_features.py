"""Common frozen OpenPI vision/language features for multi-task attachments.

This follows the native prefix path without modifying upstream code or retaining
its KV cache. The first implementation recomputes the prefix during decoding;
feature calls and time are counted separately so that cost remains visible.
"""
from __future__ import annotations

import time
import types

import numpy as np


FEATURE_KIND = "openpi_final_prefix_visual_mean_language_mean_public_state_v1"


def pool_prefix(hidden, mask, prompt_length, public_state, *, xp=np):
    """Pool visual and language tokens separately so language is not diluted."""
    visual_length = hidden.shape[1] - prompt_length
    if not 0 < visual_length < hidden.shape[1]:
        raise ValueError("Require both native visual tokens and instruction tokens")
    hidden = hidden.astype(xp.float32)
    mask = mask.astype(xp.float32)
    def mean(start, end):
        weights = mask[:, start:end]
        return xp.sum(hidden[:, start:end] * weights[..., None], axis=1) / xp.maximum(
            xp.sum(weights, axis=1, keepdims=True), 1.)
    return xp.concatenate((mean(0, visual_length), mean(visual_length, hidden.shape[1]),
                           public_state[:, :8].astype(xp.float32)), axis=-1)


def native_prefix_hidden(model, observation):
    """The exact frozen prefix computation used by Pi0.sample_actions."""
    import jax.numpy as jnp
    from openpi.models import model as model_module
    from openpi.models.pi0 import make_attn_mask
    observation = model_module.preprocess_observation(None, observation, train=False)
    tokens, mask, autoregressive_mask = model.embed_prefix(observation)
    attention = make_attn_mask(mask, autoregressive_mask)
    positions = jnp.cumsum(mask, axis=1) - 1
    (hidden, _), _ = model.PaliGemma.llm([tokens, None], mask=attention, positions=positions)
    return hidden, mask


def native_prefix_features(model, observation, public_state):
    import jax.numpy as jnp
    hidden, mask = native_prefix_hidden(model, observation)
    return pool_prefix(hidden, mask, observation.tokenized_prompt.shape[-1], public_state, xp=jnp)


def motor_element(obs, instruction):
    from openpi_client import image_tools
    return {"observation/image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["image"], 224, 224)),
            "observation/wrist_image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["wrist"], 224, 224)),
            "observation/state": obs["state"], "prompt": str(instruction)}


class FrozenOpenPIPrefixFeatures:
    """One immutable shared VLA feature space across all tasks and methods."""
    def __init__(self, policy, *, checkpoint_receipt_sha256, openpi_commit, model_config):
        from openpi.shared import nnx_utils
        self.policy = policy
        self.apply = nnx_utils.module_jit(types.MethodType(native_prefix_features, policy._model))
        self.calls, self.cache_hits, self.elapsed_seconds = 0, 0, 0.
        self._last_obs, self._last_instruction, self._last_features = None, None, None
        self.contract = dict(kind=FEATURE_KIND, checkpoint_receipt_sha256=checkpoint_receipt_sha256,
            openpi_commit=openpi_commit, model_config=model_config,
            observation_dim=4104, visual_dim=2048, instruction_dim=2048, public_state_dim=8,
            pooling="separate mask-weighted means of final VLM visual and instruction tokens",
            normalization="raw frozen hidden values; student fits normalization on training tasks only",
            trainable_feature_parameters=0, kv_cache_retained=False,
            execution="native prefix recomputed for features and motor; both costs recorded")

    def prepared(self, obs, instruction):
        import jax
        import jax.numpy as jnp
        from openpi.models.model import Observation
        element = motor_element(obs, instruction)
        transformed = self.policy._input_transform(jax.tree.map(lambda x: x, element))
        batched = jax.tree.map(lambda x: jnp.asarray(x)[None], transformed)
        observation = Observation.from_dict(batched)
        if observation.tokenized_prompt is None or not np.asarray(observation.tokenized_prompt_mask).any():
            raise ValueError("Multi-task features require a nonempty public instruction")
        return observation

    def __call__(self, obs, instruction):
        if self._last_obs is obs and self._last_instruction == instruction:
            self.cache_hits += 1
            return self._last_features.copy()
        import jax.numpy as jnp
        started = time.monotonic()
        observation = self.prepared(obs, instruction)
        result = np.asarray(self.apply(observation, jnp.asarray(obs["state"], dtype=jnp.float32)[None]))[0]
        if result.shape != (4104,) or not np.isfinite(result).all():
            raise ValueError("Native pooled VLA feature shape or values differ from the declared contract")
        self.calls += 1
        self.elapsed_seconds += time.monotonic() - started
        self._last_obs, self._last_instruction, self._last_features = obs, instruction, result.copy()
        return result.copy()

    def qualify(self, obs, instruction):
        """Check pooled output against the unpooled native final-prefix hidden."""
        from openpi.shared import nnx_utils
        reference = nnx_utils.module_jit(types.MethodType(native_prefix_hidden, self.policy._model))
        observation = self.prepared(obs, instruction)
        hidden, mask = reference(observation)
        expected = pool_prefix(np.asarray(hidden), np.asarray(mask),
            observation.tokenized_prompt.shape[-1], np.asarray(obs["state"])[None])[0]
        actual = self(obs, instruction)
        delta = float(np.max(np.abs(expected - actual)))
        if delta > 2e-4:
            raise RuntimeError("Pooled feature path differs from native final-prefix hidden states")
        return dict(passed=True, maximum_delta=delta, final_prefix_shape=list(hidden.shape),
                    public_feature_shape=list(actual.shape), feature_contract=self.contract)
