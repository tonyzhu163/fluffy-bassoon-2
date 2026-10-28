"""Cached rotation composition, split isolation and reward-only PPO plumbing."""

import json

import numpy as np
import pytest
from gymnasium.spaces.utils import flatdim
from gymnasium.utils.env_checker import check_env
import torch

from fluffy_bassoon.meta_rl.maniskill_rotation import (
    ANGLES, NET_ANGLES, CachedRotationTrial, OutcomeCache, baseline_trials, evaluate_policy,
)
from fluffy_bassoon.meta_rl.ppo import GRUCore, RecurrentActorCritic, collect_trials, update_ppo


@pytest.fixture
def cache_file(tmp_path):
    rows = []
    for scene in range(32):
        for angle in NET_ANGLES:
            rows.append(dict(object=scene // 4, overlay=scene % 4, position=scene,
                             orientation=0, episode_id=1000 + scene, scene_index=scene,
                             net_angle_degrees=angle, repeat=0, success_once=angle == 0,
                             grasped_once=True, object_plate_distance_end_m=12345))
    path = tmp_path / "outcomes.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_cache_validates_exact_angle_composition_and_scene_split(cache_file):
    cache = OutcomeCache(cache_file)
    assert cache.describe()["splits"] == dict(train=24, validation=8)
    assert not set(cache.split_scenes["train"]) & set(cache.split_scenes["validation"])
    assert {a + b for a in ANGLES for b in ANGLES} == set(NET_ANGLES)
    assert all(isinstance(value, float) for outcomes in cache.outcomes.values() for value in outcomes)
    with pytest.raises(ValueError, match="checksum"):
        OutcomeCache(cache_file, "0" * 64)
    rows = cache_file.read_text().splitlines()
    cache_file.write_text("\n".join(rows[:-1]) + "\n")
    with pytest.raises(ValueError, match="exact composed"):
        OutcomeCache(cache_file)


def test_duplicate_rows_and_episode_leakage_are_rejected(cache_file):
    original = cache_file.read_text()
    cache_file.write_text(original + original.splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        OutcomeCache(cache_file)
    rows = [json.loads(line) for line in original.splitlines()]
    for row in rows:
        if row["scene_index"] == 3:
            row["episode_id"] = 1000
    cache_file.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="Episode identity"):
        OutcomeCache(cache_file)


def test_rotation_trial_contract_and_privileged_canceling_controller(cache_file):
    cache = OutcomeCache(cache_file)
    check_env(CachedRotationTrial(cache), skip_render_check=True)
    env = CachedRotationTrial(cache, split="validation")
    initial, info = env.reset(seed=7)
    assert info == {} and len(set(env._trial_scenes)) == 8
    assert set(env._trial_scenes) <= set(cache.split_scenes["validation"])
    theta = env._theta
    for attempt in range(8):
        action = ANGLES.index(-theta)
        observation, reward, terminated, truncated, info = env.step(action)
        assert reward == 1 and terminated == (attempt == 7) and not truncated
        assert info == dict(attempt_end=True, attempt_index=attempt, is_success=True)
        assert observation["previous_reward"].item() == 1
        assert env._theta == theta and env.observation_space.contains(observation)
        assert set(observation) == set(initial)
        np.testing.assert_array_equal(observation["state"], [0])


def test_reward_blinding_changes_only_feedback_and_preserves_seeded_rewards(cache_file):
    cache = OutcomeCache(cache_file)
    normal, blind = (CachedRotationTrial(cache, reward_blind=value) for value in (False, True))
    normal.reset(seed=27)
    blind.reset(seed=27)
    for _ in range(8):
        action = ANGLES.index(-normal._theta)
        a, b = normal.step(action), blind.step(action)
        assert a[1:] == b[1:] and a[1] == 1
        assert a[0]["previous_reward"].item() == 1
        assert b[0]["previous_reward"].item() == 0
        for key in set(a[0]) - {"previous_reward"}:
            np.testing.assert_array_equal(a[0][key], b[0][key])


def test_baselines_and_ppo_evaluation_cover_all_lesions_without_changing_rng(cache_file):
    cache = OutcomeCache(cache_file)
    baselines, raw = baseline_trials(cache, list(range(12)))
    assert baselines["oracle"]["mean_reward"] == 1
    assert raw["thompson"].shape == (12, 8)
    envs = [CachedRotationTrial(cache) for _ in range(2)]
    policy = RecurrentActorCritic(GRUCore(int(flatdim(envs[0].observation_space)), width=8),
                                   envs[0].action_space)
    rollout = collect_trials(policy, envs, [0, 1])
    metrics = update_ppo(policy, torch.optim.AdamW(policy.parameters(), lr=.001), rollout,
                         epochs=1, minibatch_trials=2)
    assert metrics["optimizer_steps"] == 1
    before_rng = torch.get_rng_state().clone()
    before_model = {key: value.clone() for key, value in policy.state_dict().items()}
    results, arrays = evaluate_policy(policy, cache, [0, 1])
    assert len(results) == 6 and arrays["sampled_reward_blind"].shape == (2, 8)
    torch.testing.assert_close(torch.get_rng_state(), before_rng, rtol=0, atol=0)
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value, before_model[key], rtol=0, atol=0)
