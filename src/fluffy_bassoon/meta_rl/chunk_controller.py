"""Frozen decoded-chunk controller shared by physical training/evaluation.

The frozen motor owns numerical B/row and supplies current public features and
base proposals. This controller never changes motor batching or takes physics
steps. Its caller records the actual native batch and dispatches full commands.
"""
from copy import deepcopy

import numpy as np
import torch

from .chunk_history import ChunkHistoryEncoder, MODES


class FrozenChunkController:
    def __init__(self, policy, *, native_batch, mode='full'):
        if type(native_batch) is not int or native_batch < 1 or mode not in MODES:
            raise ValueError('Require a fixed native batch and explicit history mode')
        self.policy = policy.eval().requires_grad_(False)
        self.native_batch, self.mode = native_batch, mode
        self.layout = policy.config.layout
        self.encoders = [ChunkHistoryEncoder(self.layout, mode=mode) for _ in range(native_batch)]
        self.state, self.pending, self.failed = None, None, False
        self._signature = self._weights()

    def _weights(self):
        return {name:(id(value), value._version, value.dtype, value.device)
                for name,value in (*self.policy.named_parameters(), *self.policy.named_buffers())}

    def _check(self):
        if self.failed:
            raise RuntimeError('Controller failed; preserve the physical record without automatic replay')
        if (any(model.training for model in self.policy.modules())
                or any(p.requires_grad for p in self.policy.parameters()) or self._weights() != self._signature):
            raise RuntimeError('Frozen decoded-controller weights, normalization or mode changed')

    def _flags(self, values, name):
        values = np.asarray(values)
        if values.dtype != np.bool_ or values.shape != (self.native_batch,):
            raise ValueError('Require explicit boolean per-lane '+name)
        return values

    def reset_trials(self, lanes=None):
        self._check()
        if self.pending is not None:
            raise RuntimeError('Observe pending physical commands before resetting trials')
        lanes = list(range(self.native_batch)) if lanes is None else list(lanes)
        if (not lanes or len(set(lanes)) != len(lanes)
                or any(type(i) is not int or not 0 <= i < self.native_batch for i in lanes)):
            raise ValueError('Require distinct valid trial lanes')
        for lane in lanes:
            self.encoders[lane].reset_trial()

    @torch.no_grad()
    def propose(self, features, bases, *, episode_starts, active=None, deterministic=True, generator=None):
        self._check()
        if self.pending is not None:
            raise RuntimeError('Observe the pending physical decision before another query')
        parameter = next(self.policy.parameters())
        for value, shape in ((features, (self.native_batch, self.layout.feature_dim)),
                             (bases, (self.native_batch, *self.layout.command_shape))):
            if (not torch.is_tensor(value) or value.shape != shape or value.dtype != torch.float32
                    or value.device != parameter.device or not torch.isfinite(value).all()):
                raise ValueError('Require current finite float32 public features/base at the fixed native batch on device')
        starts = self._flags(episode_starts, 'episode starts')
        active = np.ones(self.native_batch, np.bool_) if active is None else self._flags(active, 'physical activity')
        if (starts & ~active).any():
            raise ValueError('An inactive tail call cannot start a physical episode')
        encoders = deepcopy(self.encoders)
        feedback = np.zeros((self.native_batch, self.layout.feedback_dim), np.float32)
        reset = np.zeros(self.native_batch, np.bool_)
        for lane in np.flatnonzero(active):
            feedback[lane], reset[lane] = encoders[lane].query_parts(episode_start=starts[lane])
        try:
            if active.any():
                tokens = torch.cat((features.detach(), bases.detach().flatten(1),
                    torch.as_tensor(feedback, device=parameter.device),
                    torch.as_tensor(starts[:, None], device=parameter.device, dtype=torch.float32)), -1).to(parameter.dtype)
                before = self.state if self.state is not None else self.policy.core.initial_state(self.native_batch)
                residual, _, after = self.policy.step(tokens, before,
                    reset=torch.as_tensor(reset, device=parameter.device),
                    deterministic=deterministic, generator=generator)
                mask = torch.as_tensor(active, device=parameter.device)
                state = {key:torch.where(mask.reshape(-1, *([1]*(value.ndim-1))), value, before[key])
                         for key,value in after.items()}
                residual = torch.where(mask[:, None, None], residual.float(), 0.)
            else:
                residual, state = torch.zeros_like(bases), self.state
            commands = bases.detach()+residual
            if not torch.isfinite(commands).all():
                raise ValueError('Nonfinite decoded policy commands')
            self._check()
            self.encoders, self.state = encoders, state
            self.pending = dict(active=active.copy(), commands=commands.cpu().numpy().copy())
            return commands
        except BaseException:
            self.failed = True
            raise

    def observe(self, issued, delivered_masks, rewards, *, terminated, truncated, physics_seconds):
        self._check()
        if self.pending is None:
            raise RuntimeError('Physical feedback requires a pending policy query')
        try:
            issued, masks = np.asarray(issued), np.asarray(delivered_masks)
            rewards, seconds = np.asarray(rewards), np.asarray(physics_seconds)
            term, trunc = self._flags(terminated, 'termination'), self._flags(truncated, 'timeout')
            if (issued.dtype != np.float32 or not np.array_equal(issued, self.pending['commands'])
                    or masks.shape != (self.native_batch, self.layout.chunk_length) or masks.dtype != np.bool_
                    or rewards.shape != seconds.shape or seconds.shape != (self.native_batch,)
                    or not np.isfinite(rewards).all() or not np.isfinite(seconds).all()):
                raise ValueError('Require measured feedback for the exact full issued commands')
            active = self.pending['active']
            if ((seconds[active] <= 0).any() or (seconds[~active] != 0).any()
                    or masks[~active].any()):
                raise ValueError('Only positive-duration decisions advance history; audit terminal no-ops separately')
            encoders = deepcopy(self.encoders)
            for lane in np.flatnonzero(active):
                encoders[lane].observe(issued[lane], masks[lane], float(rewards[lane]),
                    terminated=term[lane], truncated=trunc[lane], physics_seconds=float(seconds[lane]))
            self.encoders, self.pending = encoders, None
        except BaseException:
            self.failed = True
            raise
