"""Parallel sequence training must match incremental deployment numerically."""
from copy import deepcopy
from types import MethodType

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.causal_decoder import CausalDecoderCore
from fluffy_bassoon.meta_rl.chunk_history import ChunkHistoryLayout
from fluffy_bassoon.meta_rl.chunk_policy import ChunkModelConfig
from fluffy_bassoon.meta_rl.chunk_value import (
    ChunkHistoryLearner, ChunkObjectiveConfig, chunk_batch, chunk_learning_step)
from fluffy_bassoon.meta_rl.history_value import importance_weights


@pytest.mark.parametrize('mode', ['full', 'episode', 'static'])
def test_sequence_values_and_gradients_match_cached_sliding_attention(mode):
    torch.manual_seed(304)
    parallel = CausalDecoderCore(5, width=12, memory_length=2, layers=2, heads=3).double()
    reference = deepcopy(parallel)
    tokens = torch.randn(7, 2, 5, dtype=torch.double)
    valid = torch.tensor([[True, t < 4] for t in range(7)])
    starts = torch.zeros(7, 2, dtype=torch.bool)
    starts[0] = True;starts[3, 0] = True;starts[2, 1] = True
    reset = torch.zeros_like(starts) if mode == 'full' else starts.clone()
    if mode == 'static':
        reset[:] = True
    reset[0] = True
    tokens[~valid] = torch.nan
    left = tokens.clone().requires_grad_()
    right = tokens.clone().requires_grad_()
    actual = parallel.sequence(left, reset=reset, valid=valid)
    state, outputs = None, []
    for t in range(len(tokens)):
        if state is not None:
            empty = reference.initial_state(2)
            state = {key:torch.where(reset[t].reshape(-1, *([1]*(value.ndim-1))), empty[key], value)
                     for key,value in state.items()}
        value, state = reference(torch.where(valid[t, :, None], right[t], 0.), state)
        outputs.append(value)
    expected = torch.stack(outputs)
    torch.testing.assert_close(actual[valid], expected[valid], rtol=2e-12, atol=2e-12)
    weights = torch.randn_like(actual)
    (actual[valid]*weights[valid]).sum().backward()
    (expected[valid]*weights[valid]).sum().backward()
    torch.testing.assert_close(left.grad, right.grad, rtol=2e-11, atol=2e-12)
    for a,b in zip(parallel.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, rtol=2e-11, atol=2e-12)


def test_parallel_attention_is_causal_and_clears_episode_history():
    torch.manual_seed(17)
    core = CausalDecoderCore(3, width=8, layers=2, memory_length=2048, heads=2).double()
    tokens = torch.randn(8, 2, 3, dtype=torch.double)
    reset = torch.zeros(8, 2, dtype=torch.bool);reset[0] = True;reset[4, 0] = True
    valid = torch.ones_like(reset)
    expected = core.sequence(tokens, reset=reset, valid=valid)
    changed = tokens.clone();changed[6:] += 100
    actual = core.sequence(changed, reset=reset, valid=valid)
    torch.testing.assert_close(actual[:6], expected[:6], rtol=0, atol=0)
    changed = tokens.clone();changed[:4, 0] -= 100
    actual = core.sequence(changed, reset=reset, valid=valid)
    torch.testing.assert_close(actual[4:, 0], expected[4:, 0], rtol=0, atol=0)


def test_sequence_masks_require_complete_lane_prefixes_and_valid_dtype():
    core = CausalDecoderCore(3, width=8, layers=1, memory_length=4, heads=2)
    tokens = torch.zeros(3, 2, 3)
    valid = torch.ones(3, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match='boolean'):
        core.sequence(tokens, reset=valid.float(), valid=valid)
    broken = valid.clone();broken[1, 0] = False
    with pytest.raises(ValueError, match='contiguous'):
        core.sequence(tokens, reset=valid, valid=broken)
    tokens[1, 0] = torch.nan
    with pytest.raises(ValueError, match='finite valid'):
        core.sequence(tokens, reset=valid, valid=valid)


def _trial(short=False):
    count = 2 if short else 4
    return dict(features=np.arange(count*3, dtype=np.float32).reshape(count, 3)/10,
        bases=np.zeros((count, 2, 2), np.float32), issued=np.full((count, 2, 2), .15, np.float32),
        delivered_masks=np.tile(np.array([True, False]), (count, 1)),
        rewards=np.array([0., 1., 0., 0.][:count], np.float32),
        terminated=np.array([False, True, False, False][:count]),
        truncated=np.array([False, False, False, True][:count]),
        physics_seconds=np.full(count, .1, np.float32),
        episode_starts=np.array([True, False, True, False][:count]))


def _reset(batch, time, mode):
    if time == 0 or mode == 'static':
        return torch.ones_like(batch['valid'][time])
    return batch['starts'][time] if mode == 'episode' else None


def _stepped_critic(self, model, batch, mode):
    state, outputs = None, []
    for time, tokens in enumerate(batch['tokens']):
        valid = batch['valid'][time]
        kwargs = dict(action=torch.where(valid[:, None, None], batch['actions'][time], 0.)) if model.action_value else {}
        value, state = model.step(torch.where(valid[:, None], tokens, 0.), state,
                                  reset=_reset(batch, time, mode), **kwargs)
        outputs.append(value)
    return torch.stack(outputs)


def _stepped_actor_loss(self, batch, *, mode):
    valid, _ = self._validate(batch, mode)
    with torch.no_grad():
        weights = importance_weights(self._critic_sequence(self.q, batch, mode)-
                                     self._critic_sequence(self.v, batch, mode), self.config)
    state, losses = None, []
    for time, token in enumerate(batch['tokens']):
        active = valid[time]
        logp, state = self.actor.log_prob(torch.where(active[:, None], token, 0.),
            torch.where(active[:, None, None], batch['actions'][time], 0.), state,
            reset=_reset(batch, time, mode))
        losses.append(-logp)
    nll = torch.stack(losses)
    return (weights*nll)[valid].mean(), dict(nll=nll[valid].mean().detach(),
        weight_mean=weights[valid].mean(), weight_max=weights[valid].max())


@pytest.mark.parametrize('mode', ['full', 'episode', 'static'])
@pytest.mark.parametrize('phase', ['actor', 'value'])
def test_decoded_sequence_loss_and_adam_update_match_stepped_reference(mode, phase):
    torch.manual_seed(804)
    cfg = ChunkModelConfig(ChunkHistoryLayout(3, 2, 2), width=8, layers=2, heads=2, context_length=8)
    parallel = ChunkHistoryLearner(cfg, torch.full((2, 2), .5), ChunkObjectiveConfig('ic_iql')).double()
    reference = deepcopy(parallel)
    reference._critic_sequence = MethodType(_stepped_critic, reference)
    reference.actor_loss = MethodType(_stepped_actor_loss, reference)
    batch = chunk_batch([_trial(), _trial(short=True), _trial()], parallel.actor, mode=mode)
    # Nontrivial normalization is shared by actor, Q, V and target Q.
    for learner in (parallel, reference):
        with torch.no_grad():
            for model in (learner.actor,learner.q,learner.v,learner.target_q):
                model.feature_mean.copy_(torch.tensor([.1, .2, .3]))
                model.feature_std.copy_(torch.tensor([.2, .3, .4]))
    optimizers = [torch.optim.AdamW(m.actor.parameters() if phase == 'actor' else
        [*m.q.parameters(), *m.v.parameters()], lr=3e-4) for m in (parallel, reference)]
    a = chunk_learning_step(parallel, optimizers[0], batch, phase=phase, mode=mode)
    b = chunk_learning_step(reference, optimizers[1], batch, phase=phase, mode=mode)
    assert a['loss'] == pytest.approx(b['loss'], rel=1e-12, abs=1e-12)
    for left,right in zip(parallel.parameters(),reference.parameters(),strict=True):
        torch.testing.assert_close(left,right,rtol=2e-10,atol=2e-11)
        if left.grad is not None:
            torch.testing.assert_close(left.grad,right.grad,rtol=2e-10,atol=2e-11)
