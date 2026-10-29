"""Common frozen OpenPI vision/language features for multi-task attachments.

This follows the native prefix path without modifying upstream code or retaining
its KV cache. The first implementation recomputes the prefix during decoding;
feature calls and time are counted separately so that cost remains visible.
"""
from __future__ import annotations

import time
import types

import numpy as np


FEATURE_KIND = "openpi_final_prefix_visual_mean_language_mean_public_state_v2"
QUALIFICATION_MAX_DELTA = 2e-4


def difference_metrics(reference, candidate):
    """Exact shape/finite checks and numerical errors, without choosing a tolerance."""
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    if reference.shape != candidate.shape or not reference.size:
        raise ValueError("Require matching nonempty numerical arrays")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("Non-finite feature diagnostic arrays")
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    norm = float(np.linalg.norm(reference.astype(np.float64).reshape(-1)))
    return dict(maximum_absolute_error=float(np.max(np.abs(difference))),
                rms_error=float(np.sqrt(np.mean(difference ** 2))),
                relative_l2_error=float(np.linalg.norm(difference.reshape(-1)) / max(norm, 1e-12)),
                exactly_equal=bool(np.array_equal(reference, candidate)),
                reference_dtype=str(reference.dtype), candidate_dtype=str(candidate.dtype))


def partition_differences(reference, candidate, hidden_width):
    if np.asarray(reference).shape != (hidden_width * 2 + 8,):
        raise ValueError("Feature diagnostic requires visual, language and eight state coordinates")
    return {name: difference_metrics(reference[start:end], candidate[start:end])
            for name, start, end in (("visual", 0, hidden_width),
                                     ("language", hidden_width, hidden_width * 2),
                                     ("state", hidden_width * 2, hidden_width * 2 + 8))}


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
    """Legacy fused model/pooling path, retained only for numerical diagnostics."""
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
        self.hidden_apply = nnx_utils.module_jit(types.MethodType(native_prefix_hidden, policy._model))
        self.fused_apply = nnx_utils.module_jit(types.MethodType(native_prefix_features, policy._model))
        self.calls, self.cache_hits, self.elapsed_seconds = 0, 0, 0.
        self._last_obs, self._last_instruction, self._last_features = None, None, None
        self.contract = dict(kind=FEATURE_KIND, checkpoint_receipt_sha256=checkpoint_receipt_sha256,
            openpi_commit=openpi_commit, model_config=model_config,
            observation_dim=4104, visual_dim=2048, instruction_dim=2048, public_state_dim=8,
            pooling="separate NumPy float32 mask-weighted means after native hidden materialization",
            normalization="raw frozen hidden values; student fits normalization on training tasks only",
            trainable_feature_parameters=0, kv_cache_retained=False,
            execution="standalone native prefix hidden materialized on host before pooling; native prefix recomputed for motor; both costs recorded",
            hidden_materialization="standalone module_jit output transferred to host before reduction",
            pooling_dtype="float32")

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
        started = time.monotonic()
        observation = self.prepared(obs, instruction)
        # Keep the model output boundary explicit. Fusing pooling into the model
        # changed BF16 results on the qualified runtime; host transfer prevents
        # that reduction from being compiled into the native prefix function.
        hidden, mask = self.hidden_apply(observation)
        result = pool_prefix(np.asarray(hidden), np.asarray(mask),
            observation.tokenized_prompt.shape[-1], np.asarray(obs["state"])[None])[0]
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
        delta = difference_metrics(expected, actual)["maximum_absolute_error"]
        if delta > QUALIFICATION_MAX_DELTA:
            raise RuntimeError("Pooled feature path differs from native final-prefix hidden states")
        return dict(passed=True, maximum_delta=delta, final_prefix_shape=list(hidden.shape),
                    public_feature_shape=list(actual.shape), feature_contract=self.contract)

    def diagnose(self, obs, instruction):
        """Isolate model-fusion differences from reduction differences; never relax qualification."""
        import jax
        import jax.numpy as jnp
        from openpi.shared import nnx_utils
        started = time.monotonic()
        observation = self.prepared(obs, instruction)
        state = jnp.asarray(obs["state"], dtype=jnp.float32)[None]
        prompt_length = observation.tokenized_prompt.shape[-1]
        reference = nnx_utils.module_jit(types.MethodType(native_prefix_hidden, self.policy._model))
        hidden, mask = reference(observation)
        hidden_host, mask_host = np.asarray(hidden), np.asarray(mask)
        host = pool_prefix(hidden_host, mask_host, prompt_length, np.asarray(obs["state"])[None])[0]
        # Both reductions consume the SAME materialized standalone hidden tensor.
        jax_pool = jax.jit(lambda h, m, s: pool_prefix(h, m, prompt_length, s, xp=jnp))
        same_hidden_jax = np.asarray(jax_pool(hidden, mask, state))[0]
        # Direct calls deliberately bypass __call__'s observation-identity cache.
        fused_first = np.asarray(self.fused_apply(observation, state))[0]
        fused_repeat = np.asarray(self.fused_apply(observation, state))[0]
        canonical_first = self(obs, instruction)
        # A new mapping bypasses the observation-identity cache while retaining
        # exactly the same public input arrays and instruction.
        canonical_repeat = self(dict(obs), instruction)
        width = hidden_host.shape[-1]
        comparisons = {
            "same_hidden_numpy_vs_jax_pool": partition_differences(host, same_hidden_jax, width),
            "standalone_hidden_numpy_pool_vs_fused_model_pool": partition_differences(host, fused_first, width),
            "standalone_hidden_jax_pool_vs_fused_model_pool": partition_differences(same_hidden_jax, fused_first, width),
            "identical_fused_feature_repeat": partition_differences(fused_first, fused_repeat, width),
            "standalone_hidden_numpy_pool_vs_canonical_features": partition_differences(host, canonical_first, width),
            "identical_canonical_feature_repeat": partition_differences(canonical_first, canonical_repeat, width),
        }
        delta = difference_metrics(host, canonical_first)["maximum_absolute_error"]
        fused_delta = difference_metrics(host, fused_first)["maximum_absolute_error"]
        deterministic = bool(np.array_equal(canonical_first, canonical_repeat))
        return dict(completed=True, original_qualification_passed=delta <= QUALIFICATION_MAX_DELTA,
            original_qualification_max_delta=QUALIFICATION_MAX_DELTA, maximum_delta=delta,
            identical_feature_function_deterministic=deterministic,
            identical_fused_feature_function_deterministic=bool(np.array_equal(fused_first, fused_repeat)),
            legacy_fused_maximum_delta=fused_delta,
            legacy_fused_original_qualification_passed=fused_delta <= QUALIFICATION_MAX_DELTA,
            canonical_repeat_bypassed_observation_cache=True,
            comparisons=comparisons, hidden_dtype=str(hidden_host.dtype), hidden_shape=list(hidden_host.shape),
            mask_dtype=str(mask_host.dtype), token_mask=mask_host.astype(bool).tolist(),
            prompt_length=prompt_length, visual_token_count=int(mask_host[:, :-prompt_length].sum()),
            language_token_count=int(mask_host[:, -prompt_length:].sum()),
            prompt_token_mask=np.asarray(observation.tokenized_prompt_mask).astype(bool).tolist(),
            feature_contract=self.contract, elapsed_seconds=time.monotonic() - started,
            interpretation="Diagnostics do not relax the original qualification threshold or establish feature correctness.")
