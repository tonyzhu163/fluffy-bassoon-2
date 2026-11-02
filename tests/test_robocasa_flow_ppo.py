import numpy as np
import pytest

from fluffy_bassoon.meta_rl.rlinf_robocasa_flow_ppo import first_success_reward


def test_transient_success_pays_once_without_later_chunk_cancellation():
    successes = np.array([[False, False], [True, False], [False, True],
                          [True, True], [False, False]])
    seen = np.zeros(2, dtype=bool)
    rewards = []
    for success in successes:
        rewards.append(first_success_reward(success, seen))
        seen |= success
    np.testing.assert_array_equal(rewards, [[0, 0], [1, 0], [0, 1], [0, 0], [0, 0]])
    np.testing.assert_array_equal(np.sum(rewards, axis=0), [1, 1])


def test_first_success_rejects_a_broadcast_latch():
    with pytest.raises(ValueError, match='latch shape'):
        first_success_reward([True, False], False)
