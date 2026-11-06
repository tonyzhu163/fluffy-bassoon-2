"""Shared batched frozen RoboTwin motor for action-residual train and eval.

This is the native evaluation ODE motor, with explicit initial noise. RLinf's
Flow-SDE PPO/GRPO teachers retain their native train sampler and likelihoods.
The input transform and preprocessor run once, then the same prefix is consumed
by denoising. Residual heads run outside this module's no-grad boundary.
"""
from dataclasses import dataclass
import hashlib

import torch


PUBLIC_KEYS = ("main_images", "wrist_images", "states", "task_descriptions")


def public_batch(observation):
    """Keep only the three cameras, proprioception and instruction, never truth."""
    result = {key: observation[key] for key in PUBLIC_KEYS}
    state, main, wrists = result["states"], result["main_images"], result["wrist_images"]
    if (not torch.is_tensor(state) or state.ndim != 2 or state.shape[1] != 14
            or not len(state) or not state.is_floating_point() or not torch.isfinite(state).all()):
        raise ValueError("Require finite native public states [B,14]")
    batch = len(state)
    if (not torch.is_tensor(main) or not torch.is_tensor(wrists)
            or main.ndim != 4 or wrists.ndim != 5
            or main.shape[0] != batch or wrists.shape[:2] != (batch, 2)
            or main.shape[-1] != 3 or wrists.shape[-1] != 3
            or min(*main.shape[1:3], *wrists.shape[2:4]) < 1
            or main.dtype != torch.uint8 or wrists.dtype != torch.uint8):
        raise ValueError("Require native uint8 main [B,H,W,3] and wrist [B,2,H,W,3] images")
    prompts = result["task_descriptions"]
    if (not isinstance(prompts, (list, tuple)) or len(prompts) != batch
            or any(not isinstance(text, str) or not text.strip() for text in prompts)):
        raise ValueError("Require one actual instruction per batch row")
    result["task_descriptions"] = list(prompts)
    return result


def stack_public_observations(observations):
    """Batch current singleton lanes without copying monitoring-only fields."""
    rows = [public_batch(row) for row in observations]
    if not rows or any(len(row["states"]) != 1 for row in rows):
        raise ValueError("Require a nonempty sequence of singleton native observations")
    return public_batch({key: ([row[key][0] for row in rows] if key == "task_descriptions"
                              else torch.cat([row[key] for row in rows], dim=0))
                         for key in PUBLIC_KEYS})


def case_noise(decision_ids, *, seed, horizon, action_dim, device):
    """One original-shaped draw per decision, independent of batching/order.

    Addresses must include trial, episode and chunk identity. This specifies
    only the frozen ODE motor's initial noise, not a Flow-SDE teacher schedule.
    CPU and CUDA generators are separate numerical conditions.
    """
    identities = tuple(decision_ids)
    if (not identities or len(set(identities)) != len(identities)
            or any(not isinstance(key, str) or not key for key in identities)
            or type(seed) is not int or not 0 <= seed < 2**63
            or any(type(n) is not int or n < 1 for n in (horizon, action_dim))):
        raise ValueError("Require unique decision addresses, a seed and positive native noise dimensions")
    draws = []
    for identity in identities:
        digest = hashlib.sha256(f"robotwin-ode-noise-v1\0{seed}\0{identity}".encode()).digest()
        address_seed = int.from_bytes(digest[:8], "big") % (2**63)
        generator = torch.Generator(device=device).manual_seed(address_seed)
        draws.append(torch.randn((1, horizon, action_dim), generator=generator,
                                 device=device, dtype=torch.float32))
    return torch.cat(draws, dim=0)


class _PrefixReady(Exception):
    pass


@dataclass(frozen=True)
class _Prepared:
    observation: object
    prefix_mask: torch.Tensor
    cache: object
    batch_size: int


class FrozenRoboTwinMotor:
    """One frozen motor per inference worker, shared by residual train and eval.

    A preparation token belongs to exactly one current observation batch and
    is consumed even when sampling fails. Never share this instance across
    simultaneous callers. Features retain the existing native order: masked
    mean VLM prefix, followed by transformed/normalized proprioception14.
    """

    def __init__(self, model, *, prefix_width=2048):
        if (type(prefix_width) is not int or prefix_width < 1
                or any(type(getattr(model, key, None)) is not int or getattr(model, key) < 1
                       for key in ("action_horizon", "action_dim"))
                or getattr(model, "action_chunk", model.action_horizon) != model.action_horizon
                or getattr(model, "action_env_dim", None) != 14):
            raise ValueError("Bind a full native horizon, noise dimension and decoded ALOHA14 motor")
        self.model = model.eval().requires_grad_(False)
        self.prefix_width, self.feature_dim = prefix_width, prefix_width + 14
        self.noise_shape = (model.action_horizon, model.action_dim)
        self._prepared, self._preparing = None, False
        self._signature = self._weights()
        self.prefix_calls = self.sample_calls = self.prepared_rows = self.sampled_rows = 0

    def _weights(self):
        return {name: (id(tensor), tensor._version, tensor.dtype, tensor.device)
                for name, tensor in (*self.model.named_parameters(), *self.model.named_buffers())}

    def check_frozen(self):
        if (any(module.training for module in self.model.modules())
                or any(p.requires_grad for p in self.model.parameters())
                or self._weights() != self._signature):
            raise RuntimeError("Frozen RoboTwin motor weights, buffers or eval mode changed")

    @torch.no_grad()
    def prepare_batch(self, observation):
        if self._prepared is not None or self._preparing:
            raise RuntimeError("Consume the current prefix before preparing another observation")
        self.check_frozen()
        public = public_batch(observation)
        batch, captured = len(public["states"]), []
        original = self.model.build_prefix_cache

        def capture(prepared):
            prefix, mask, cache = original(prepared)
            captured.append((prepared, prefix, mask, cache))
            raise _PrefixReady()

        self._preparing = True
        self.model.build_prefix_cache = capture
        try:
            self.model.predict_action_batch(public, mode="eval", noise=torch.zeros(
                (batch, *self.noise_shape), dtype=torch.float32, device=self.model.device))
        except _PrefixReady:
            pass
        finally:
            self.model.build_prefix_cache = original
            self._preparing = False
        if len(captured) != 1:
            raise RuntimeError("Require exactly one native prefix before denoising")
        prepared, prefix, mask, cache = captured[0]
        if (mask.ndim != 2 or len(mask) != batch or mask.dtype != torch.bool
                or prefix.shape != (batch, mask.shape[1], self.prefix_width)
                or not mask.any(1).all() or not torch.isfinite(prefix[mask]).all()
                or prepared.state.ndim != 2 or len(prepared.state) != batch
                or prepared.state.shape[1] < 14):
            raise ValueError("Require valid native prefixes and transformed state for every row")
        mean = prefix.float().masked_fill(~mask[..., None], 0).sum(1) / mask.sum(1, keepdim=True)
        features = torch.cat((mean, prepared.state[:, :14].float()), dim=-1).detach()
        if not torch.isfinite(features).all():
            raise ValueError("Nonfinite native pre-action features")
        self.check_frozen()
        self._prepared = _Prepared(prepared, mask, cache, batch)
        self.prefix_calls += 1
        self.prepared_rows += batch
        return features, self._prepared

    @torch.no_grad()
    def propose_batch(self, prepared, *, noise):
        """Decode exactly the supplied noise through the already-built prefix."""
        if self._prepared is None or prepared is not self._prepared:
            raise ValueError("Require the current unconsumed observation token")
        try:
            self.check_frozen()
            if (not torch.is_tensor(noise) or noise.shape != (prepared.batch_size, *self.noise_shape)
                    or noise.device != torch.device(self.model.device) or noise.dtype != torch.float32
                    or not torch.isfinite(noise).all()):
                raise ValueError("Require finite explicit native float32 noise [B,H,D] on the motor device")
            targets = self.model._sample_actions_from_prefix_cache(
                prepared.observation, prepared.prefix_mask, prepared.cache, noise=noise.detach())
            actions = self.model.decode_actions(targets, prepared.observation.state)
            if (not torch.is_tensor(actions)
                    or actions.shape != (prepared.batch_size, self.noise_shape[0], 14)
                    or not torch.isfinite(actions).all()):
                raise ValueError("Require finite decoded native action chunks [B,H,14]")
            self.check_frozen()
            self.sample_calls += 1
            self.sampled_rows += prepared.batch_size
            return actions.detach()
        finally:
            self._prepared = None

    def discard(self, prepared):
        if self._prepared is None or prepared is not self._prepared:
            raise ValueError("Discard only the current observation token")
        self._prepared = None

    def infer_batch(self, observation, *, noise):
        features, prepared = self.prepare_batch(observation)
        return self.propose_batch(prepared, noise=noise), features
