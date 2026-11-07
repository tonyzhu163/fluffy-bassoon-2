"""AD and value-weighted fitting for public decoded-action histories.

The physical corpus/export gate remains separate. These losses consume complete
positive-duration episodes, retain whole requested residual chunks, and share
the online encoder's full/episode/static reset semantics.
"""
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .chunk_history import MODES, trial_tokens
from .chunk_policy import ChunkHistoryCritic, ChunkResidualPolicy
from .history_value import (ValueConfig, discounted_returns, expectile_loss,
                            importance_weights, learning_step, next_values)


@dataclass(frozen=True)
class ChunkObjectiveConfig:
    objective: str
    discount: float = .99
    expectile: float = .7
    temperature: float | None = None
    maximum_weight: float | None = None
    target_tau: float = .005
    bellman_weight: float = .1

    def __post_init__(self):
        if self.objective not in ('ad', 'dit', 'ic_iql'):
            raise ValueError('Choose AD, DIT-style or offline IQL; DPT requires qualified query labels')
        if self.temperature is None:
            object.__setattr__(self, 'temperature', 3. if self.objective == 'ic_iql' else 1.)
        if self.maximum_weight is None:
            object.__setattr__(self, 'maximum_weight', 100. if self.objective == 'ic_iql' else 20.)
        ValueConfig('ic_iql' if self.objective == 'ic_iql' else 'dit',
            self.discount, self.expectile, self.temperature, self.maximum_weight,
            self.target_tau, self.bellman_weight)


def chunk_batch(trials, policy, *, mode='full', discount=.99):
    """Collate already selected complete trials/windows, without hidden labels.

    No implicit data splitting, window sampling, normalization or label clipping
    occurs here. No-op calls stay in the export audit rather than this physical
    decision stream. Construct tokens after selecting the admitted window.
    """
    if mode not in MODES or not trials:
        raise ValueError('Require complete decoded-action windows and an explicit history mode')
    layout = policy.config.layout
    windows = []
    fields = ('features', 'bases', 'issued', 'delivered_masks', 'rewards',
              'terminated', 'truncated', 'physics_seconds', 'episode_starts')
    for trial in trials:
        token, _ = trial_tokens(layout, mode=mode, **{key: trial[key] for key in fields})
        actions = np.stack([layout.residual(a, b) for a, b in zip(trial['issued'], trial['bases'], strict=True)])
        done = np.asarray(trial['terminated']) | np.asarray(trial['truncated'])
        windows.append(dict(tokens=token, actions=actions, rewards=trial['rewards'],
            terminated=trial['terminated'], truncated=trial['truncated'],
            starts=trial['episode_starts'], physics_seconds=trial['physics_seconds'],
            returns=discounted_returns(trial['rewards'], done, discount)))
    length = max(len(row['tokens']) for row in windows)
    if length > policy.config.context_length:
        raise ValueError('Complete physical windows exceed the declared context capacity')
    parameter = next(policy.parameters())
    shapes = dict(tokens=(layout.token_dim,), actions=layout.command_shape,
        rewards=(), terminated=(), truncated=(), starts=(), physics_seconds=(), returns=(), valid=())
    boolean = ('terminated', 'truncated', 'starts', 'valid')
    batch = {key: torch.zeros((length, len(windows), *shape), device=parameter.device,
        dtype=torch.bool if key in boolean else parameter.dtype) for key, shape in shapes.items()}
    for lane, window in enumerate(windows):
        count = len(window['tokens'])
        batch['valid'][:count, lane] = True
        for key, value in window.items():
            batch[key][:count, lane] = torch.as_tensor(value, device=parameter.device)
    return batch


class ChunkHistoryLearner(nn.Module):
    """Shared actor architecture, separate history-conditioned Q/V estimators.

    Values fit first, followed by actor extraction with detached value weights.
    AD uses unit weights and the same transformed decoded-action density. This
    module does not admit a one-version corpus as chronological AD training.
    """
    def __init__(self, model_config, residual_radius, objective_config):
        super().__init__()
        if not isinstance(objective_config, ChunkObjectiveConfig):
            raise ValueError('Require an explicit decoded-chunk objective configuration')
        self.config = objective_config
        self.actor = ChunkResidualPolicy(model_config, residual_radius)
        self.q = ChunkHistoryCritic(model_config, action_value=True)
        self.v = ChunkHistoryCritic(model_config, action_value=False)
        self.target_q = deepcopy(self.q).requires_grad_(False)

    @torch.no_grad()
    def fit_feature_normalization(self, training_features):
        self.actor.fit_feature_normalization(training_features)
        for critic in (self.q, self.v, self.target_q):
            critic.feature_mean.copy_(self.actor.feature_mean)
            critic.feature_std.copy_(self.actor.feature_std)

    def _validate(self, batch, mode):
        if mode not in MODES:
            raise ValueError('Require full, episode or static history mode')
        valid = batch['valid']
        if (valid.ndim != 2 or min(valid.shape) < 1 or valid.dtype != torch.bool or not valid[0].all()
                or (valid[1:] & ~valid[:-1]).any()):
            raise ValueError('Require a nonempty contiguous physical prefix in every lane')
        for key in ('terminated', 'truncated', 'starts'):
            if batch[key].dtype != torch.bool or batch[key].shape != valid.shape:
                raise ValueError('Keep explicit separate physical boundary flags')
        done = batch['terminated'] | batch['truncated']
        has_next = torch.cat((valid[1:], torch.zeros_like(valid[:1])), 0)
        if ((valid & ~has_next) & ~done).any() or not batch['starts'][0].all():
            raise ValueError('Require complete physical episodes, with both timeout and termination terminal')
        if ((batch['starts'][1:] != done[:-1]) & valid[1:]).any():
            raise ValueError('Physical starts disagree with the preceding terminal feedback')
        if batch['tokens'].shape != (*valid.shape, self.actor.config.layout.token_dim):
            raise ValueError('Require the public decoded-chunk token layout')
        if batch['actions'].shape != (*valid.shape, *self.actor.config.layout.command_shape):
            raise ValueError('Require whole decoded residual action labels')
        if (batch['tokens'][..., -1][valid] != batch['starts'][valid]).any():
            raise ValueError('Token episode starts disagree with Bellman boundaries')
        for key in ('rewards', 'physics_seconds', 'returns'):
            if batch[key].shape != valid.shape or not torch.isfinite(batch[key][valid]).all():
                raise ValueError('Require finite physical feedback and value labels')
        if not (batch['physics_seconds'][valid] > 0).all():
            raise ValueError('Audit zero-duration no-ops outside physical decision fitting')
        following = torch.zeros_like(batch['rewards'][0])
        returns = []
        for time in range(len(valid)-1, -1, -1):
            following = torch.where(valid[time], batch['rewards'][time] +
                self.config.discount*following*(~done[time]), 0.)
            returns.append(following)
        expected = torch.stack(returns[::-1])
        if not torch.allclose(batch['returns'][valid], expected[valid], rtol=1e-5, atol=1e-6):
            raise ValueError('Return labels disagree with the declared finite-episode Bellman clock')
        return valid, done

    @staticmethod
    def _reset(batch, time, mode):
        if time == 0 or mode == 'static':
            return torch.ones_like(batch['valid'][time])
        return batch['starts'][time] if mode == 'episode' else None

    def _critic_sequence(self, model, batch, mode):
        kwargs = dict(action=batch['actions']) if model.action_value else {}
        return model.sequence(batch['tokens'], reset=self._sequence_resets(batch, mode),
                              valid=batch['valid'], **kwargs)

    @staticmethod
    def _sequence_resets(batch, mode):
        if mode == 'static':
            return torch.ones_like(batch['valid'])
        reset = batch['starts'].clone() if mode == 'episode' else torch.zeros_like(batch['valid'])
        reset[0] = True
        return reset

    def value_loss(self, batch, *, mode='full'):
        if self.config.objective == 'ad':
            raise ValueError('AD has no value-fitting phase')
        valid, done = self._validate(batch, mode)
        q = self._critic_sequence(self.q, batch, mode)
        value = self._critic_sequence(self.v, batch, mode)
        rewards = torch.where(valid, batch['rewards'], 0.)
        with torch.no_grad():
            frozen_q = self._critic_sequence(self.target_q, batch, mode)
            td = rewards + self.config.discount*next_values(value.detach(), done, valid)
        if self.config.objective == 'ic_iql':
            q_loss = (q-td).square()[valid].mean()
            v_loss = expectile_loss(frozen_q-value, self.config.expectile)[valid].mean()
        else:
            returns = torch.where(valid, batch['returns'], 0.)
            next_q = next_values(frozen_q, done, valid)
            q_loss = ((q-returns).square() + self.config.bellman_weight*
                (q-rewards-self.config.discount*next_q).square())[valid].mean()
            v_loss = ((value-returns).square() + self.config.bellman_weight*(value-td).square())[valid].mean()
        return q_loss+v_loss, dict(q_loss=q_loss.detach(), v_loss=v_loss.detach())

    def actor_loss(self, batch, *, mode='full'):
        valid, _ = self._validate(batch, mode)
        with torch.no_grad():
            weights = (torch.ones_like(batch['rewards']) if self.config.objective == 'ad' else
                importance_weights(self._critic_sequence(self.q, batch, mode)-
                                   self._critic_sequence(self.v, batch, mode), self.config))
        nll = -self.actor.log_prob_sequence(batch['tokens'], batch['actions'],
            reset=self._sequence_resets(batch, mode), valid=valid)
        return (weights*nll)[valid].mean(), dict(nll=nll[valid].mean().detach(),
            weight_mean=weights[valid].mean(), weight_max=weights[valid].max())

    @torch.no_grad()
    def update_target(self):
        for target, parameter in zip(self.target_q.parameters(), self.q.parameters(), strict=True):
            target.lerp_(parameter, self.config.target_tau)


def chunk_learning_step(learner, optimizer, batch, *, phase, mode='full', execution_batch_size=None):
    """Reuse the qualified global valid-decision weighting/Adam execution path."""
    return learning_step(learner, optimizer, batch, phase=phase, mode=mode,
                         execution_batch_size=execution_batch_size)
