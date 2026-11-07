"""Physical Bellman boundaries and decoded-chunk learner numerical controls."""
from copy import deepcopy
import math

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.chunk_history import ChunkHistoryLayout
from fluffy_bassoon.meta_rl.chunk_policy import ChunkModelConfig
from fluffy_bassoon.meta_rl.chunk_value import (
    ChunkObjectiveConfig, ChunkHistoryLearner, chunk_batch, chunk_learning_step)


def learner(objective='ic_iql'):
    cfg = ChunkModelConfig(ChunkHistoryLayout(2, 2, 1), width=8, layers=1,
        heads=2, context_length=8)
    return ChunkHistoryLearner(cfg, torch.ones(2, 1), ChunkObjectiveConfig(objective)).double()


def trial(*, short=False):
    count = 2 if short else 4
    return dict(features=np.arange(count*2, dtype=np.float32).reshape(count, 2)/10,
        bases=np.zeros((count, 2, 1), np.float32),
        issued=np.full((count, 2, 1), .2, np.float32),
        delivered_masks=np.tile(np.array([True, False]), (count, 1)),
        rewards=np.array([1., 2., 7., 11.][:count], np.float32),
        terminated=np.array([False, True, False, False][:count]),
        truncated=np.array([False, False, False, True][:count]),
        physics_seconds=np.full(count, .004, np.float32),
        episode_starts=np.array([True, False, True, False][:count]))


def constant(critic, value):
    with torch.no_grad():
        for parameter in critic.parameters():
            parameter.zero_()
        critic.head[-1].bias.fill_(value)


def test_physical_batch_keeps_full_tail_and_stops_at_both_boundaries():
    model = learner()
    batch = chunk_batch([trial(), trial(short=True)], model.actor)
    assert batch['valid'].sum().item() == 6
    torch.testing.assert_close(batch['returns'][:, 0], torch.tensor([2.98, 2., 17.89, 11.], dtype=torch.double))
    # Undelivered tails remain action targets and shifted public feedback.
    assert (batch['actions'][batch['valid']][:, -1] != 0).all()
    begin = model.actor.config.layout.current_dim
    assert batch['tokens'][2, 0, begin] == pytest.approx(.2)
    assert batch['tokens'][2, 0, begin+1] == pytest.approx(.2)
    model._validate(batch, 'full')
    bad = deepcopy(batch); bad['truncated'][-1, 0] = False
    with pytest.raises(ValueError, match='complete physical episodes'):
        model.value_loss(bad)
    bad = deepcopy(batch); bad['physics_seconds'][0, 0] = 0
    with pytest.raises(ValueError, match='zero-duration'):
        model.value_loss(bad)
    with pytest.raises(ValueError, match='Bellman clock'):
        model.value_loss(chunk_batch([trial()], model.actor, discount=.5))


def test_iql_losses_match_reference_without_cross_episode_bootstrap():
    model = learner()
    batch = chunk_batch([trial(), trial(short=True)], model.actor)
    constant(model.q, 2.); constant(model.v, 3.); constant(model.target_q, 4.)
    loss, metrics = model.value_loss(batch)
    # The next episode's V=3 never bootstraps the terminated row at t1 or timeout t3.
    targets = torch.tensor([1+.99*3, 2., 7+.99*3, 11., 1+.99*3, 2.], dtype=torch.double)
    expected_q = (2-targets).square().mean()
    torch.testing.assert_close(metrics['q_loss'], expected_q, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(metrics['v_loss'], torch.tensor(.7, dtype=torch.double))
    torch.testing.assert_close(loss, expected_q+.7)
    loss.backward()
    assert model.q.head[-1].bias.grad is not None and model.v.head[-1].bias.grad is not None
    assert all(p.grad is None for p in model.actor.parameters())
    assert all(p.grad is None for p in model.target_q.parameters())
    model.update_target()
    torch.testing.assert_close(model.target_q.head[-1].bias, torch.tensor([3.99], dtype=torch.double))


def test_actor_extraction_weights_density_and_gradients_are_separate():
    for objective, weight in [('ad', 1.), ('dit', 20.), ('ic_iql', math.exp(1.))]:
        model = learner(objective)
        constant(model.q, 4.); constant(model.v, 1.)
        batch = chunk_batch([trial()], model.actor, mode='static')
        loss, metrics = model.actor_loss(batch, mode='static')
        logp = []
        for time in range(4):
            value, _ = model.actor.log_prob(batch['tokens'][time], batch['actions'][time],
                reset=torch.ones(1, dtype=torch.bool))
            logp.append(value)
        expected = -torch.stack(logp).mean()*weight
        torch.testing.assert_close(loss, expected, rtol=1e-12, atol=1e-12)
        assert metrics['weight_mean'] == pytest.approx(weight)
        loss.backward()
        assert model.actor.actor.weight.grad.isfinite().all()
        assert all(p.grad is None for p in (*model.q.parameters(), *model.v.parameters(), *model.target_q.parameters()))
    with pytest.raises(ValueError, match='qualified query labels'):
        ChunkObjectiveConfig('dpt')


def test_execution_batch_partition_preserves_global_valid_decision_update():
    torch.manual_seed(23)
    full, split = learner(), None
    split = deepcopy(full)
    batch = chunk_batch([trial(), trial(short=True), trial()], full.actor, mode='episode')
    optimizers = [torch.optim.SGD(list(m.q.parameters())+list(m.v.parameters()), lr=.001)
                  for m in (full, split)]
    a = chunk_learning_step(full, optimizers[0], batch, phase='value', mode='episode')
    b = chunk_learning_step(split, optimizers[1], batch, phase='value', mode='episode', execution_batch_size=2)
    assert a['loss'] == pytest.approx(b['loss'], rel=1e-12, abs=1e-12)
    for left, right in zip(full.parameters(), split.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=1e-11, atol=1e-11)
