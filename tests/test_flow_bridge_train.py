from copy import deepcopy

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.flow_history import FlowBridgeConfig, HistoryFlowSteeringPolicy
from fluffy_bassoon.meta_rl.flow_bridge_train import (
    bridge_update, flow_batch_loss, query_context, query_weights, restore_resume,
    save_resume, steering_effect, steering_gate, validate_batches, episode_reset_batch)


class Motor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.project = torch.nn.Linear(4, 32, bias=False)
        self.eval().requires_grad_(False)

    def run_suffix(self, observation, actions, time, cache, mask):
        key, value = cache[0]
        return actions * .3 + self.project((key + value).flatten(2).mean(1))[:, None]

    def velocity_from_suffix(self, suffix):
        return suffix


def fixture(interface):
    torch.manual_seed(31)
    policy = HistoryFlowSteeringPolicy(FlowBridgeConfig(
        interface=interface, observation_dim=16, history_action_dim=12, width=8,
        layers=1, heads=2, context_length=8, kv_heads=1, head_dim=4,
        kv_rank=2, kv_layers=(0,), action_horizon=4))
    motor = Motor()
    batch = dict(tokens=torch.randn(5, 4, 31), actions=torch.randn(5, 4, 12),
        rewards=torch.zeros(5, 4), dones=torch.zeros(5, 4, dtype=torch.bool),
        starts=torch.zeros(5, 4, dtype=torch.bool), returns=torch.zeros(5, 4),
        valid=torch.ones(5, 4, dtype=torch.bool), query_rel=torch.tensor([1, 2, 3, 4]),
        target_actions=torch.randn(4, 4, 32), noise=torch.randn(4, 4, 32),
        timesteps=torch.tensor([.2, .4, .6, .8]), weights=torch.tensor([.5, 1., 2., 3.]),
        query_indices=[(i, i + 1) for i in range(4)])
    batch['starts'][0] = True
    batch['dones'][-1] = True
    cache = torch.randn(4, 3, 1, 4), torch.randn(4, 3, 1, 4)

    def prepare(indices):
        lanes = [index[0] for index in indices]
        return {}, ((cache[0][lanes], cache[1][lanes]),), torch.ones(len(lanes), 3, dtype=torch.bool)

    return policy, motor, batch, prepare


@pytest.mark.parametrize('interface', ['kv_bridge', 'velocity_bridge'])
def test_native_bridge_logical_batch_matches_accumulation_and_excludes_expert(interface):
    policy, motor, batch, prepare = fixture(interface)
    other = deepcopy(policy)
    before = deepcopy(motor.state_dict())
    first = torch.optim.AdamW(policy.parameters(), lr=3e-4, weight_decay=0.)
    second = torch.optim.AdamW(other.parameters(), lr=3e-4, weight_decay=0.)
    for _ in range(2):
        a = bridge_update(policy, motor, first, batch, prepare, execution_batch_size=4)
        b = bridge_update(other, motor, second, batch, prepare, execution_batch_size=1)
        assert a['loss'] == pytest.approx(b['loss'], abs=2e-6)
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value, other.state_dict()[key], atol=2e-6, rtol=2e-5)
    assert all(torch.equal(value, motor.state_dict()[key]) for key, value in before.items())
    assert all(p.grad is None for p in motor.parameters())
    motor.project.weight.requires_grad_(True)
    with pytest.raises(ValueError, match='Freeze'):
        bridge_update(policy, motor, first, batch, prepare, execution_batch_size=4)


@pytest.mark.parametrize('interface', ['kv_bridge', 'velocity_bridge'])
def test_flow_checkpoint_restores_exact_next_update_and_both_rngs(tmp_path, interface):
    policy, motor, batch, prepare = fixture(interface)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-4, weight_decay=0.)
    sampler = np.random.default_rng(123)
    generator = torch.Generator().manual_seed(456)
    bridge_update(policy, motor, optimizer, batch, prepare, execution_batch_size=2)
    checkpoint = tmp_path / 'resume.pt'
    contract = dict(interface=interface, frozen_motor='test', corpus='same')
    save_resume(checkpoint, contract=contract, policy=policy, optimizer=optimizer,
        sampler=sampler, generator=generator, progress=dict(update=1, plateau={'stale_checks': 3}))
    expected_sample = sampler.integers(1000, size=10)
    batch['noise'] = torch.randn(batch['noise'].shape, generator=generator)
    expected_noise = batch['noise'].clone()
    bridge_update(policy, motor, optimizer, batch, prepare, execution_batch_size=2)
    expected = deepcopy(policy.state_dict())
    progress = restore_resume(checkpoint, contract=contract, policy=policy, optimizer=optimizer,
        sampler=sampler, generator=generator)
    assert progress == dict(update=1, plateau={'stale_checks': 3})
    np.testing.assert_array_equal(sampler.integers(1000, size=10), expected_sample)
    batch['noise'] = torch.randn(batch['noise'].shape, generator=generator)
    assert torch.equal(batch['noise'], expected_noise)
    bridge_update(policy, motor, optimizer, batch, prepare, execution_batch_size=2)
    assert all(torch.equal(value, policy.state_dict()[key]) for key, value in expected.items())
    with pytest.raises(ValueError, match='exact'):
        restore_resume(checkpoint, contract=dict(contract, corpus='other'), policy=policy,
            optimizer=optimizer, sampler=sampler, generator=generator)


@pytest.mark.parametrize('interface', ['kv_bridge', 'velocity_bridge'])
def test_context_never_uses_future_query_labels_or_feedback(interface):
    policy, _, batch, _ = fixture(interface)
    before = query_context(policy, batch).detach()
    changed = deepcopy(batch)
    changed['target_actions'].fill_(10000.)
    changed['actions'].fill_(-10000.)
    changed['returns'].fill_(99999.)
    for lane, query in enumerate(batch['query_rel']):
        changed['tokens'][int(query) + 1:, lane] = 10000.
    assert torch.equal(before, query_context(policy, changed).detach())
    assert torch.equal(query_weights(batch, 'ad'), torch.ones(4))
    with pytest.raises(ValueError, match='fitted frozen value'):
        query_weights(batch, 'dit')


@pytest.mark.parametrize('interface', ['kv_bridge', 'velocity_bridge'])
def test_native_validation_keeps_required_graph_without_backward_or_updates(interface):
    policy, motor, batch, prepare = fixture(interface)
    before = deepcopy(policy.state_dict())
    expected = float(flow_batch_loss(policy, motor, batch, prepare).detach())
    with torch.no_grad():
        actual = validate_batches(policy, motor, [batch, batch], prepare)
    assert actual == expected
    assert all(torch.equal(v, policy.state_dict()[k]) for k, v in before.items())
    assert all(p.grad is None for p in policy.parameters())
    assert all(p.grad is None for p in motor.parameters())


@pytest.mark.parametrize('interface', ['kv_bridge', 'velocity_bridge'])
def test_steering_gate_rejects_native_parity_and_admits_fitted_history_steering(interface):
    policy, motor, batch, prepare = fixture(interface)
    batch['starts'][2] = True
    gate = dict(minimum_loss_reduction=1e-3, minimum_relative_steering=1e-3, minimum_history_fraction=1e-3)
    initial = steering_effect(policy, motor, [batch], prepare)
    assert initial['queries'] == 4
    assert initial['steering_rms'] == 0. and initial['loss_reduction'] == pytest.approx(0., abs=1e-7)
    assert initial['native_loss'] == pytest.approx(initial['steered_loss'])
    assert not steering_gate(initial, **gate)['passed']
    optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-2, weight_decay=0.)
    batch['weights'] = torch.ones(4)
    for _ in range(30):
        bridge_update(policy, motor, optimizer, batch, prepare, execution_batch_size=4)
    fitted = steering_effect(policy, motor, [batch], prepare)
    assert fitted['steered_loss'] < fitted['native_loss'] and fitted['relative_steering'] > 0
    assert steering_gate(fitted, **dict(gate, minimum_history_fraction=0.))['passed']
    history = steering_gate(fitted, **gate)
    if interface == 'velocity_bridge':
        assert history['passed']
    else:
        # This toy KV fit learns a context-independent edit; the gate must say so.
        assert not history['passed'] and history['failures'][0].startswith('history_fraction')
    assert all(p.grad is None for p in motor.parameters())
    with pytest.raises(ValueError, match='threshold'):
        steering_gate(fitted, **dict(gate, minimum_history_fraction=float('nan')))


@pytest.mark.parametrize('interface', ['kv_bridge', 'velocity_bridge'])
def test_episode_reset_input_drops_previous_episode_feedback(interface):
    policy, _, batch, _ = fixture(interface)
    batch['starts'][2] = True
    reset = episode_reset_batch(policy, batch)
    feedback = slice(16, 16 + 12 + 2)
    assert torch.equal(reset['tokens'][2, :, feedback], torch.zeros(4, 14))
    assert torch.equal(reset['tokens'][3:], batch['tokens'][3:])
    assert torch.equal(reset['tokens'][2, :, :16], batch['tokens'][2, :, :16])
    assert torch.equal(reset['tokens'][2, :, -1], batch['tokens'][2, :, -1])
    assert not torch.equal(reset['tokens'], batch['tokens'])
    # Only previous-episode feedback at the start differs: reset contexts must ignore it.
    changed = deepcopy(batch)
    changed['tokens'][2, :, feedback] = 1000.
    torch.testing.assert_close(
        query_context(policy, episode_reset_batch(policy, changed), mode='episode'),
        query_context(policy, reset, mode='episode'))
