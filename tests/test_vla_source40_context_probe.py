"""Offline fixed-action context replay; execute only on allocated Linux."""
from copy import deepcopy
import gzip
import pickle

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.vla_history import FrozenHistoryController, HistoryConfig, HistoryNoisePolicy, file_sha256
from fluffy_bassoon.meta_rl.vla_source40_context_probe import (
    VARIANTS, load_episode, policy_signature, probe_context, restore, snapshot, validate_rows)


def recorded():
    torch.manual_seed(31)
    policy = HistoryNoisePolicy(HistoryConfig(3, 2, width=8, layers=1, heads=2, context_length=8))
    controller = FrozenHistoryController(policy)
    controller.generator = torch.Generator().manual_seed(100)
    rows = []
    for index in range(3):
        feature = np.asarray([index, 1., -2.], np.float32)
        proposed = controller.act(feature)
        executed = np.clip(proposed, -1., 1.).astype(np.float32)
        done, reward = index == 2, float(index == 2) - 1.
        controller.observe(executed_noise=executed, reward=reward, done=done)
        rows.append(dict(public_features=feature, proposed_noise=proposed, executed_noise=executed,
            reward=reward, done=done, primitive_steps=10))
    controller.generator = torch.Generator().manual_seed(200)
    feature = np.asarray([0., 1., -2.], np.float32)
    proposed = controller.act(feature)
    query = dict(public_features=feature, proposed_noise=proposed,
        executed_noise=np.clip(proposed, -1., 1.).astype(np.float32), reward=0., done=True, primitive_steps=10)
    return policy, rows, query


def test_genuine_replay_and_fixed_weights_with_paired_sampling():
    policy, support, query = recorded()
    before = policy_signature(policy)
    result, snapshots = probe_context(policy, support, query, support_seed=100, query_seed=200)
    assert tuple(result['variants']) == VARIANTS
    assert result['genuine_first_query_proposal_max_error'] == 0.
    assert result['genuine_first_query_clipped_max_error'] == 0.
    assert result['weights_unchanged'] and result['new_simulator_actions'] == result['optimizer_updates'] == 0
    assert before == policy_signature(policy)
    for variant in result['variants'].values():
        assert variant['paired_standard_normal_max_error'] < 1e-5
    for saved in snapshots.values():
        np.testing.assert_array_equal(saved['support_end']['previous_noise'], support[-1]['executed_noise'])
        assert saved['support_end']['previous_done'] and not saved['support_end']['pending']
        assert saved['query_before']['generator'].device.type == 'cpu'
        assert saved['query_after']['pending']


def test_reset_matches_empty_history_and_only_declared_reward_inputs_change():
    policy, support, query = recorded()
    untouched = deepcopy(support)
    result, saved = probe_context(policy, support, query, support_seed=100, query_seed=200)
    empty = FrozenHistoryController(policy, generator=torch.Generator().manual_seed(200))
    expected = empty.act(query['public_features'])
    np.testing.assert_array_equal(expected, result['variants']['episode_reset']['proposed_noise'])
    assert result['variants']['episode_reset']['effective_prior_memory_l2'] == 0.
    assert result['variants']['genuine']['effective_prior_memory_l2'] > 0.
    assert result['variants']['all_rewards_masked']['support_reward_inputs'] == [0., 0., 0.]
    assert result['variants']['terminal_reward_flipped']['support_reward_inputs'] == [-1., -1., -1.]
    assert saved['terminal_reward_flipped']['query_before']['previous_reward'] == -1.
    for original, after in zip(untouched, support, strict=True):
        for name in original:
            np.testing.assert_array_equal(original[name], after[name])


def test_query_outcome_and_later_observations_are_not_probe_inputs():
    policy, support, query = recorded()
    original, _ = probe_context(policy, support, query, support_seed=100, query_seed=200)
    changed = dict(query, reward=1e20, done=False, motor_next_observation='unseen future')
    other, _ = probe_context(policy, support, changed, support_seed=100, query_seed=200)
    assert original == other


def test_genuine_proposal_or_clip_mismatch_is_rejected():
    policy, support, query = recorded()
    changed = deepcopy(query)
    changed['proposed_noise'][0] += .01
    with pytest.raises(RuntimeError, match='replay differs'):
        probe_context(policy, support, changed, support_seed=100, query_seed=200)
    changed = deepcopy(query)
    changed['executed_noise'][0] += .01
    with pytest.raises(RuntimeError, match='replay differs'):
        probe_context(policy, support, changed, support_seed=100, query_seed=200)
    with pytest.raises(RuntimeError, match='elapsed-time'):
        probe_context(policy, support, query, support_seed=100, query_seed=200, deadline=0.)


def test_snapshots_restore_context_and_generator_exactly():
    policy, support, query = recorded()
    _, saved = probe_context(policy, support, query, support_seed=100, query_seed=200)
    restored = FrozenHistoryController(policy)
    restore(restored, saved['genuine']['query_before'])
    np.testing.assert_array_equal(restored.act(query['public_features']), query['proposed_noise'])
    assert snapshot(restored)['pending']


def test_trace_hash_boundaries_and_actual_clipping_are_checked(tmp_path):
    _, support, _ = recorded()
    receipt = dict(completed=True, chunks=3, primitive_steps=30, trace='trace.pkl.gz')
    trace = tmp_path / receipt['trace']
    with gzip.open(trace, 'wb') as stream:
        pickle.dump(support, stream)
    receipt['trace_sha256'] = file_sha256(trace)
    assert len(load_episode(tmp_path, receipt, 3, 2)) == 3
    changed = deepcopy(support)
    changed[0]['done'] = True
    with pytest.raises(ValueError, match='boundary'):
        validate_rows(changed, receipt, 3, 2)
    changed = deepcopy(support)
    changed[0]['executed_noise'][0] += .1
    with pytest.raises(ValueError, match='clipping'):
        validate_rows(changed, receipt, 3, 2)
    trace.write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        load_episode(tmp_path, receipt, 3, 2)
