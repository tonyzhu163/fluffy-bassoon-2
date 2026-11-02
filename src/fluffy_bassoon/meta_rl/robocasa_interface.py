"""Pre-action public features and deterministic steering for native frozen pi0.5.

prepare intercepts the existing prefix API before denoising, using explicit
dummy noise. execute reuses that exact prefix and supplies the policy's entire
noise tensor. No second motor sample or seed-based hidden exploration enters.
GPU/native qualification is required before a new scientific run.
"""
import math

import numpy as np
import torch

from .robocasa_residual import public_observation
from .robocasa_residual import RoboCasaResidualEnvironment


class _PrefixReady(Exception):
    pass


def _steer_value(entry, code, fraction, seed, batch):
    key, value = entry
    if value.ndim < 2 or value.shape[0] != batch:
        raise ValueError("Require a batch-aligned native value cache")
    rng = np.random.default_rng(seed)
    projection = torch.as_tensor(rng.choice((-1., 1.), (32, value.shape[-1])) / math.sqrt(32),
                                 dtype=torch.float32, device=value.device)
    delta = (torch.tanh(code) @ projection).to(value.dtype)
    delta = delta.reshape(batch, *([1] * (value.ndim - 2)), value.shape[-1])
    rms = value.float().square().mean(tuple(range(1, value.ndim)), keepdim=True).sqrt()
    return key, value + fraction * rms.to(value.dtype) * delta


class FrozenRoboCasaInterface:
    def __init__(self, native, *, bound=None):
        self.native, self.bound = native, bound
        self.model = native.model.eval().requires_grad_(False)
        self._prepared = None

    @torch.no_grad()
    def prepare(self, observation):
        """Serial compatibility port; preparation itself is batched."""
        if np.shape(observation["states"]) != (1, 25):
            raise ValueError("Require one native public state25")
        features, prepared = self.prepare_batch(observation)
        return features[0], prepared

    @torch.no_grad()
    def prepare_batch(self, observation):
        """Return public pre-action features [B, 16 + prefix_width] and one prefix.

        The native input adapter runs with explicit dummy noise, then stops at
        its first prefix build, before denoising. Execution must consume this
        token before a second observation can be prepared.
        """
        if self._prepared is not None:
            raise RuntimeError("Execute the prepared observation before preparing another")
        state = observation["states"]
        if torch.is_tensor(state):
            state = state.detach().cpu().numpy()
        state = np.asarray(state, dtype=np.float32)
        if state.ndim != 2 or state.shape[1] != 25 or not len(state):
            raise ValueError("Require a nonempty batch of native public state25")
        batch = len(state)
        public = public_observation(observation)
        original = self.model.build_prefix_cache
        captured = []

        def capture(*args, **kwargs):
            result = original(*args, **kwargs)
            captured.append(result)
            raise _PrefixReady()

        self.model.build_prefix_cache = capture
        try:
            self.model.predict_action_batch(public, mode="eval",
                noise=torch.zeros((batch, 50, 32), device=self.model.device))
        except _PrefixReady:
            pass
        finally:
            self.model.build_prefix_cache = original
        if len(captured) != 1:
            raise ValueError("Native prefix API must run exactly once before action generation")
        hidden, mask, _ = captured[0]
        if (hidden.ndim != 3 or mask.ndim != 2 or hidden.shape[:2] != mask.shape
                or len(mask) != batch or mask.dtype != torch.bool or not mask.any(-1).all()):
            raise ValueError("Require one nonempty native prefix per batch row")
        pooled = hidden.float().masked_fill(~mask[..., None], 0).sum(1) / mask.sum(1, keepdim=True)
        pooled = pooled / pooled.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        state = state[:, [*range(11, 25), 7, 8]]
        stats = self.native.stats["state"]
        state = (state - np.asarray(stats["mean"])[:16]) / (np.asarray(stats["std"])[:16] + 1e-6)
        features = np.concatenate((state, pooled.cpu().numpy()), -1).astype(np.float32)
        if not np.isfinite(features).all():
            raise ValueError("Nonfinite pre-action features")
        self._prepared = dict(observation=public, prefix=captured[0], batch_size=batch)
        return features, self._prepared

    @torch.no_grad()
    def execute(self, prepared, *, noise, steering, spec):
        """Preserve the serial candidate axis used by InterfaceSpec.decode."""
        if self._prepared is None or prepared is not self._prepared or prepared["batch_size"] != 1:
            raise ValueError("Execute only the current single pre-action prefix")
        if spec.kind in ("select", "direct"):
            noise = torch.as_tensor(noise)[None]
        return self.execute_batch(prepared, noise=noise,
            steering=torch.as_tensor(steering)[None], spec=spec)[0]

    @torch.no_grad()
    def execute_batch(self, prepared, *, noise, steering, spec):
        """Execute [B,50,32] explicit noise and [B,D] steering in one native call.

        Legacy selection accepts [B,candidates,50,32] and performs one batched
        native call per candidate. Direct actions accept an empty candidate
        axis. Neither path samples noise or rebuilds the captured VLM prefix.
        """
        if self._prepared is None or prepared is not self._prepared:
            raise ValueError("Execute only the current pre-action prefix")
        prefix, velocity = self.model.build_prefix_cache, self.model.velocity_from_suffix
        batch = prepared["batch_size"]
        try:
            code = torch.as_tensor(steering, dtype=torch.float32, device=self.model.device)
            draws = torch.as_tensor(noise, device=self.model.device)
            if code.shape != (batch, spec.steering_dim) or not torch.isfinite(code).all():
                raise ValueError("Require finite steering [B,steering_dim]")
            expected = ((batch, spec.candidates, 50, 32) if spec.kind == "select" else
                        (batch, 0, 50, 32) if spec.kind == "direct" else (batch, 50, 32))
            if draws.shape != expected or not torch.isfinite(draws).all():
                raise ValueError("Require finite explicit native noise with the configured batch/candidate shape")
            native_prefix = prepared["prefix"]
            if spec.kind in ("conditioning", "conditioning_all"):
                hidden, mask, cache = native_prefix
                # Final layer only, or every layer with its own fixed projection.
                # Depth-from-top seeding keeps the final layer's map identical.
                layers = range(len(cache) - 1, len(cache)) if spec.kind == "conditioning" else range(len(cache))
                cache = list(cache)
                for layer in layers:
                    cache[layer] = _steer_value(cache[layer], code, spec.fraction,
                                                921003 + len(cache) - 1 - layer, batch)
                native_prefix = (hidden, mask, tuple(cache))
            self.model.build_prefix_cache = lambda *args, **kwargs: native_prefix
            if spec.kind == "velocity":
                def altered_velocity(suffix):
                    value = velocity(suffix)
                    return value + spec.fraction * torch.tanh(code)[:, None, :].to(value.dtype)
                self.model.velocity_from_suffix = altered_velocity
            if spec.kind == "direct":
                chunks = code.cpu().numpy().reshape(batch, 50, 12)
                stats = self.native.stats["actions"]
                chunks = chunks * (np.asarray(stats["std"])[:12] + 1e-6) + np.asarray(stats["mean"])[:12]
            else:
                candidates = draws.unbind(1) if spec.kind == "select" else (draws,)
                chunks = []
                for draw in candidates:
                    chunk, _ = self.model.predict_action_batch(prepared["observation"], mode="eval", noise=draw)
                    chunks.append(np.asarray(chunk.detach().cpu() if torch.is_tensor(chunk) else chunk,
                                             dtype=np.float32))
                chunks = np.stack(chunks, axis=1)
                selected = code.argmax(-1).cpu().numpy() if spec.kind == "select" else np.zeros(batch, dtype=int)
                chunks = chunks[np.arange(batch), selected]
            if spec.kind == "action":
                if self.bound is None or self.bound.residual_dim != 6:
                    raise ValueError("Action steering requires the qualified six-dimensional residual bound")
                chunks = np.stack([self.bound.apply(chunk, row)
                    for chunk, row in zip(chunks, code.cpu().numpy())])
            if chunks.shape != (batch, 50, 12) or not np.isfinite(chunks).all():
                raise ValueError("Require finite native action chunks [B,50,12]")
            return np.asarray(chunks, dtype=np.float32)
        finally:
            self.model.build_prefix_cache, self.model.velocity_from_suffix = prefix, velocity
            self._prepared = None


class RoboCasaInterfaceEnvironment(RoboCasaResidualEnvironment):
    """Native video/accounting port with an explicit first-success contract.

    The factory must set its native wrapper to this horizon with physical
    success termination disabled. PPO stops on the selected scorer's success
    at a chunk boundary; the exact native success control is retained for S.
    For F, prefix_observer consumes synchronized RGB from reset onwards and
    scores causal prefixes. Native success never enters that observer/payload.
    Final truth is written only into the isolated monitoring sidecar.
    """
    def __init__(self, *args, interface_horizon, reward_arm, prefix_observer=None, **kwargs):
        if (type(interface_horizon) is not int or interface_horizon < 50 or interface_horizon % 50
                or reward_arm not in ("S", "F") or reward_arm == "F" and prefix_observer is None):
            raise ValueError("Bind a full-chunk horizon and an explicit success scorer")
        super().__init__(*args, horizon=None, **kwargs)
        self.interface_horizon, self.reward_arm = interface_horizon, reward_arm
        self.prefix_observer = prefix_observer
        self.bound_horizon = interface_horizon

    def _frame(self, observation):
        super()._frame(observation)
        if self.prefix_observer is not None:
            from .robocasa_video import observation_frames
            self.prefix_observer.append(observation_frames(observation))

    def reset(self):
        if self.prefix_observer is not None:
            self.prefix_observer.reset()
        observation = super().reset()
        self.horizon = self.interface_horizon
        return observation

    def step_chunk(self, chunk):
        begin = self.steps
        observation, done, episode = super().step_chunk(chunk)
        payload = dict(executed_controls=self.steps - begin, end_step=self.steps)
        if self.reward_arm == "S":
            first = self.success_steps[0] if self.success_steps else None
            payload["outcome"] = dict(first_success_step=first, reward=float(first is not None))
        else:
            payload["outcome"] = self.prefix_observer.score(self.steps)
        return observation, done, payload

    def end_episode(self):
        if not self.active:
            return  # The inherited full-horizon path already closed its evidence.
        from .rlinf_flora_ppo import digest, write_new
        from .robocasa_video import REWARD_CAMERAS
        self.writer.close()
        counts = self._counts()
        if counts["controls"] - counts["reset_controls"] != self.steps:
            raise ValueError("Physical controls differ from first-success accounting")
        videos = self.writer.refs()
        write_new(self.output / (self.key + ".json"), dict(id=self.key, completed=True,
            task=self.native.task_descriptions[0], horizon=self.steps,
            configured_horizon=self.interface_horizon, videos=videos, camera_order=list(REWARD_CAMERAS),
            protocol="frozen_motor_first_success_chunk_v1", reward_arm=self.reward_arm,
            raw_rgb_only=True, info_on_video=False, physical_controls=self.steps))
        first = self.success_steps[0] if self.success_steps else None
        write_new(self.monitoring_root / (self.key + ".json"), dict(id=self.key, case=self.case,
            simulator_final_success=bool(self.success_steps and self.success_steps[-1] == self.steps),
            simulator_success_once=first is not None, first_success_step=first,
            success_steps=self.success_steps, native_counts=counts, video_refs_sha256=digest(videos)))
        self.active = False
