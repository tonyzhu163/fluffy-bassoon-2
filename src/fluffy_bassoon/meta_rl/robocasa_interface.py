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


class FrozenRoboCasaInterface:
    def __init__(self, native, *, bound=None):
        self.native, self.bound = native, bound
        self.model = native.model.eval().requires_grad_(False)
        self._prepared = None

    @torch.no_grad()
    def prepare(self, observation):
        if self._prepared is not None:
            raise RuntimeError("Execute the prepared observation before preparing another")
        original = self.model.build_prefix_cache
        captured = []

        def capture(*args, **kwargs):
            result = original(*args, **kwargs)
            captured.append(result)
            raise _PrefixReady()

        self.model.build_prefix_cache = capture
        try:
            self.model.predict_action_batch(public_observation(observation), mode="eval",
                noise=torch.zeros((1, 50, 32), device=self.model.device))
        except _PrefixReady:
            pass
        finally:
            self.model.build_prefix_cache = original
        if len(captured) != 1:
            raise ValueError("Native prefix API must run exactly once before action generation")
        hidden, mask, _ = captured[0]
        if mask.ndim != 2 or len(mask) != 1 or not mask.any():
            raise ValueError("Require one nonempty native prefix")
        pooled = hidden.float().masked_fill(~mask[..., None], 0).sum(1) / mask.sum(1, keepdim=True)
        pooled = pooled / pooled.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        state = np.asarray(observation["states"], dtype=np.float32)
        if state.shape != (1, 25):
            raise ValueError("Require native public state25")
        state = state[:, [*range(11, 25), 7, 8]]
        stats = self.native.stats["state"]
        state = (state - np.asarray(stats["mean"])[:16]) / (np.asarray(stats["std"])[:16] + 1e-6)
        features = np.concatenate((state, pooled.cpu().numpy()), -1)[0].astype(np.float32)
        if not np.isfinite(features).all():
            raise ValueError("Nonfinite pre-action features")
        self._prepared = dict(observation=public_observation(observation), prefix=captured[0])
        return features, self._prepared

    @torch.no_grad()
    def execute(self, prepared, *, noise, steering, spec):
        if prepared is not self._prepared:
            raise ValueError("Execute only the current pre-action prefix")
        prefix, velocity = self.model.build_prefix_cache, self.model.velocity_from_suffix
        native_prefix = prepared["prefix"]
        code = torch.as_tensor(steering[None], dtype=torch.float32, device=self.model.device)
        if spec.kind == "conditioning":
            hidden, mask, cache = native_prefix
            key, value = cache[-1]
            rng = np.random.default_rng(921003)
            projection = torch.as_tensor(rng.choice((-1., 1.), (32, value.shape[-1])) / math.sqrt(32),
                                         dtype=torch.float32, device=value.device)
            delta = (torch.tanh(code) @ projection).to(value.dtype)
            delta = delta.reshape(1, *([1] * (value.ndim - 2)), value.shape[-1])
            rms = value.float().square().mean(tuple(range(1, value.ndim)), keepdim=True).sqrt()
            native_prefix = (hidden, mask, (*cache[:-1], (key, value + spec.fraction * rms.to(value.dtype) * delta)))
        self.model.build_prefix_cache = lambda *args, **kwargs: native_prefix
        if spec.kind == "velocity":
            def altered_velocity(suffix):
                value = velocity(suffix)
                return value + spec.fraction * torch.tanh(code)[:, None, :].to(value.dtype)
            self.model.velocity_from_suffix = altered_velocity
        try:
            if spec.kind == "direct":
                chunks = np.asarray(steering, dtype=np.float32).reshape(1, 50, 12)
                stats = self.native.stats["actions"]
                chunks = chunks * (np.asarray(stats["std"])[:12] + 1e-6) + np.asarray(stats["mean"])[:12]
            else:
                chunks = []
                for draw in noise:
                    chunk, _ = self.model.predict_action_batch(prepared["observation"], mode="eval",
                        noise=torch.as_tensor(draw[None], device=self.model.device))
                    chunks.append(np.asarray(chunk.detach().cpu() if torch.is_tensor(chunk) else chunk,
                                             dtype=np.float32)[0])
                chunks = np.asarray(chunks)
            selected = int(np.argmax(steering)) if spec.kind == "select" else 0
            chunk = chunks[selected]
            if spec.kind == "action":
                if self.bound is None or self.bound.residual_dim != 6:
                    raise ValueError("Action steering requires the qualified six-dimensional residual bound")
                chunk = self.bound.apply(chunk, steering)
            if chunk.shape != (50, 12) or not np.isfinite(chunk).all():
                raise ValueError("Require a finite native action chunk")
            return np.asarray(chunk, dtype=np.float32)
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
