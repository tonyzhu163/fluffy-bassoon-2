import copy

import numpy as np
import pytest

from fluffy_bassoon.meta_rl.vla_shifted_poc import (
    SearchController, replay_rows, validate_development_case,
)


def test_search_uses_only_own_past_success_and_keeps_the_successful_arm():
    policy = SearchController(0, 4)
    np.testing.assert_array_equal(policy.action(None), np.zeros(32))
    policy.observe_episode(False)
    second = policy.action(None)
    assert not np.array_equal(second, np.zeros(32))
    policy.observe_episode(True)
    np.testing.assert_array_equal(policy.action(None), second)
    policy.observe_episode(False)
    np.testing.assert_array_equal(policy.action(None), second)


def test_timeout_bootstraps_critic_while_success_is_terminal():
    row = dict(sac_observation={"state": np.zeros((8, 1), np.float32)},
        sac_next_observation={"state": np.ones((8, 1), np.float32)},
        executed_noise=np.zeros(32, np.float32), reward=-1., success=False, done=True,
        primitive_steps=10)
    timeout, success = replay_rows([row, dict(row, reward=0., success=True)])
    assert timeout["masks"] == 1. and success["masks"] == 0.
    assert timeout["actions"].shape == (1, 32)
    assert float(timeout["discount"]) == pytest.approx(.999 ** 10)


def test_development_gate_requires_a_source_training_family_and_declared_suite():
    tasks = {"libero_90:47": dict(split="train", teacher_collection_allowed=True)}
    metadata = dict(benchmark_kind="plus", development=True, plus_case={"id": 668},
        origin_base={"train_task_key": "libero_90:47"}, horizon=280,
        initial_states=50, instruction="put the cream cheese in the basket", bddl_sha256="actual")
    validate_development_case(metadata, "plus", tasks, 280)
    for changed in (dict(metadata, development=False), dict(metadata, benchmark_kind="pro")):
        with pytest.raises(ValueError, match="development"):
            validate_development_case(changed, "plus", tasks, 280)
    heldout = copy.deepcopy(tasks)
    heldout["libero_90:47"]["split"] = "test"
    with pytest.raises(ValueError, match="development"):
        validate_development_case(metadata, "plus", heldout, 280)
