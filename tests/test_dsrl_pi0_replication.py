"""Allocated-Linux checks of original iterator chronology and physical receipts."""
from collections import deque
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from fluffy_bassoon.meta_rl.dsrl_pi0_replication import (
    CaptureEnv, Prefetch, compact_replay, restore_replay, save_complete, restore_complete,
    save_evaluation,
    frozen_motor_model,
)


def upstream_iterator(sample):
    # Exact yield/enqueue order in the pinned ReplayBuffer.get_iterator.
    queue = deque(sample() for _ in range(2))
    while queue:
        yield queue.popleft()
        queue.append(sample())


def test_prefetch_matches_upstream_sample_and_global_rng_order():
    def sample():
        return np.random.randint(0, 501, size=256)
    np.random.seed(71)
    reference = upstream_iterator(sample)
    expected = [next(reference) for _ in range(11)]
    expected_rng = np.random.randint(2**30)
    np.random.seed(71)
    actual = Prefetch(sample)
    for item in expected:
        np.testing.assert_array_equal(next(actual), item)
    assert np.random.randint(2**30) == expected_rng


@pytest.mark.parametrize('consumed', [0, 1, 7])
def test_prefetch_recovery_retains_pending_batch_and_suspended_refill(consumed):
    def sample():
        return np.random.randint(0, 9000, size=256)
    np.random.seed(19)
    original = Prefetch(sample)
    for _ in range(consumed):
        next(original)
    saved, rng = original.state(), np.random.get_state()
    expected = [next(original) for _ in range(5)]
    expected_next = np.random.randint(2**30)
    resumed = Prefetch(sample)
    resumed.restore(saved)
    np.random.set_state(rng)
    for item in expected:
        np.testing.assert_array_equal(next(resumed), item)
    assert np.random.randint(2**30) == expected_next


class Env:
    def __init__(self, *, broken_reset=False):
        self.calls, self.broken_reset = 0, broken_reset

    def reset(self):
        if self.broken_reset:
            raise RuntimeError('reset failed')
        self.calls = 0
        return {'state': np.zeros(8)}

    def get_sim_state(self):
        return np.arange(12, dtype=np.float64)

    def step(self, action):
        self.calls += 1
        return {'state': np.zeros(8)}, int(self.calls == 3), self.calls == 3, {}


def test_full_episode_reserved_before_reset_and_exact_actions_settled(tmp_path):
    capture = CaptureEnv(Env(), tmp_path, action_limit=405)
    capture.reset()
    assert json.loads((tmp_path/'physical-cost.json').read_text())['uncertain_actions'] == 400
    for _ in range(3):
        capture.step(np.zeros(7))
    capture.finish()
    assert capture.charged == 3
    assert capture.ledger['episodes'][0]['success'] is True
    assert capture.ledger['episodes'][0]['actions'] == 3
    assert capture.ledger['uncertain_actions'] == 0
    capture.reset()
    capture.step(np.zeros(7))
    capture.finish()
    assert capture.charged == 4
    assert len(capture.ledger['episodes']) == 2


def test_reset_failure_keeps_conservative_physical_charge(tmp_path):
    capture = CaptureEnv(Env(broken_reset=True), tmp_path, action_limit=400)
    with pytest.raises(RuntimeError, match='reset failed'):
        capture.reset()
    saved = json.loads((tmp_path/'physical-cost.json').read_text())
    assert saved['uncertain_actions'] == 400
    assert saved['pending']['maximum_actions'] == 400
    assert saved['known_actions'] == 0


def test_cannot_start_episode_with_insufficient_physical_budget(tmp_path):
    capture = CaptureEnv(Env(), tmp_path, action_limit=399)
    with pytest.raises(RuntimeError, match='No complete400'):
        capture.reset()
    assert capture.charged == 0


def test_replay_recovery_retains_filled_rows_capacity_and_trajectory_cursor():
    original = SimpleNamespace(data={'observations': {'state': np.arange(80).reshape(10, 8)},
        'actions': np.arange(20).reshape(10, 2)}, capacity=10, size=3,
        _traj_counter=2, _start=2, traj_bounds={0: (0, 2), 1: (2, 3)}, streaming_buffer_size=None)
    saved = compact_replay(original)
    assert saved['data']['observations']['state'].shape == (3, 8)
    assert saved['data']['actions'].shape == (3, 2)
    resumed = SimpleNamespace()
    restore_replay(resumed, saved)
    assert resumed.capacity == 10 and resumed.size == 3
    assert resumed.traj_bounds == original.traj_bounds and resumed._start == 2
    np.testing.assert_array_equal(resumed.data['observations']['state'][:3],
                                  original.data['observations']['state'][:3])
    np.testing.assert_array_equal(resumed.data['actions'][:3], original.data['actions'][:3])


def test_populated_sac_optimizer_replay_and_prefetch_next_update_recover(tmp_path):
    # Mandatory in the allocated qualification command; no VLA or simulator.
    root = os.environ.get('DSRL_OFFICIAL_ROOT')
    if root is None:
        pytest.skip('Set DSRL_OFFICIAL_ROOT in the allocated original runtime')
    import jax
    from flax import serialization
    from flax.core import freeze
    from fluffy_bassoon.meta_rl.vla_libero import OfficialSAC
    observations = dict(pixels=np.zeros((2,64,64,3,1), np.uint8),
                        state=np.zeros((2,8,1), np.float32))
    batch = freeze(dict(observations=observations, next_observations=observations,
        actions=np.zeros((2,1,32), np.float32), next_actions=np.zeros((2,1,32), np.float32),
        rewards=np.array([-1,0], np.float32), masks=np.array([1,0], np.float32),
        discount=np.full(2, .999**20, np.float32)))
    buffer = SimpleNamespace(data=dict(batch), capacity=2, size=2, _traj_counter=1,
        _start=2, traj_bounds={0:(0,2)}, streaming_buffer_size=None)
    # compact_replay accepts ordinary nested mappings, like upstream buffers.
    buffer.data = {key: dict(value) if hasattr(value, 'keys') else value
                   for key, value in buffer.data.items()}
    first = OfficialSAC(root, 0).agent
    first.update(batch)
    iterator = Prefetch(lambda: batch, jax.device_put)
    next(iterator)
    motor = SimpleNamespace(_rng=jax.random.key(17))
    contract = {'qualification':'original-pi0-complete-scalar-recovery'}
    progress = dict(episodes=1, updates=1, training_primitives=20)
    original_state = jax.tree_util.tree_map(lambda value: np.asarray(value).copy(),
                                           serialization.to_state_dict(first._save_dict))
    original_rng = np.asarray(first._rng).copy()
    save_evaluation(tmp_path, first, motor, progress, contract)
    snapshot = json.loads((tmp_path/'evaluation-checkpoints/000001.json').read_text())
    assert (tmp_path/'evaluation-checkpoints'/snapshot['path']).is_file()
    np.testing.assert_array_equal(first._rng, original_rng)
    save_complete(tmp_path, first, buffer, iterator, motor, progress, contract)
    receipt = json.loads((tmp_path/'recovery.json').read_text())
    save_complete(tmp_path, first, buffer, iterator, motor, progress, contract)
    successor = json.loads((tmp_path/'recovery.json').read_text())
    assert successor['path'] != receipt['path']
    assert (tmp_path/receipt['path']).is_file()
    assert successor['sha256'] == receipt['sha256']
    receipt = successor
    first.update(next(iterator))
    expected = jax.tree_util.tree_leaves(first._save_dict)
    second = OfficialSAC(root, 1).agent
    resumed_iterator = Prefetch(lambda: batch, jax.device_put)
    resumed_buffer = SimpleNamespace()
    resumed_motor = SimpleNamespace(_rng=jax.random.key(99))
    restored = restore_complete(tmp_path/receipt['path'], contract, second,
                                resumed_buffer, resumed_iterator, resumed_motor)
    assert restored == progress
    # Flax restoration can reorder mapping keys; compare every state leaf,
    # including optimizer moments, exactly rather than MessagePack byte order.
    restored_state = serialization.to_state_dict(second._save_dict)
    assert jax.tree_util.tree_structure(restored_state) == jax.tree_util.tree_structure(original_state)
    for left, right in zip(jax.tree_util.tree_leaves(original_state),
                           jax.tree_util.tree_leaves(restored_state), strict=True):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    assert resumed_buffer.size == 2 and resumed_buffer._traj_counter == 1
    np.testing.assert_array_equal(jax.random.key_data(resumed_motor._rng), jax.random.key_data(motor._rng))
    second.update(next(resumed_iterator))
    for left, right in zip(expected, jax.tree_util.tree_leaves(second._save_dict), strict=True):
        np.testing.assert_allclose(np.asarray(left), np.asarray(right), rtol=1e-5, atol=1e-6)


def test_pinned_policy_exposes_original_frozen_model_through_bound_method():
    if os.environ.get('DSRL_OFFICIAL_ROOT') is None:
        pytest.skip('Run in the pinned original OpenPI runtime')
    import jax.numpy as jnp
    from flax import nnx
    from openpi.policies.policy import Policy
    from fluffy_bassoon.meta_rl.vla_motor_replay import parameter_signature

    class TinyMotor(nnx.Module):
        action_dim, action_horizon = 32, 50

        def __init__(self):
            self.scale = nnx.Param(jnp.asarray(2., dtype=jnp.float32))

        def sample_actions(self, observation, noise=None):
            del observation
            return noise*self.scale.value

        def get_prefix_rep(self, observation):
            return observation

    model = TinyMotor()
    policy = Policy(model)
    assert not hasattr(policy, '_model')
    assert frozen_motor_model(policy) is model
    before = parameter_signature(model)
    np.testing.assert_array_equal(policy._sample_actions(None, noise=jnp.ones(3)),
                                  np.full(3, 2.))
    assert parameter_signature(frozen_motor_model(policy)) == before
