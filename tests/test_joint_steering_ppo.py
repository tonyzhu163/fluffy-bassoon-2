"""Episode causality and exact optimizer recovery for frozen steering PPO."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.joint_steering_ppo import (
    boundary_state, chronological_records, make_policy, normalize_first_bank, restore_boundary,
    summarize_episodes,
)
from fluffy_bassoon.meta_rl.residual_ppo import (
    ResidualActorCritic, ResidualPPOConfig, stack_records, update,
)


def actor():
    torch.manual_seed(31)
    return ResidualActorCritic(ResidualPPOConfig(feature_dim=3, residual_dim=2,
        width=8, layers=1, epochs=2, minibatch_size=2, gamma=.99, target_kl=.02))


def bank(policy):
    records = []
    for episode in range(2):
        for chunk in range(3):
            features = torch.tensor([[episode, chunk, 1.]], dtype=torch.float32)
            control, log_prob, value = policy.act(features,
                action_noise=torch.tensor([[.3 * (chunk - 1), -.2]]))
            records.append(dict(features=features[0].numpy(), control=control[0].numpy(),
                log_prob=float(log_prob[0]), value=float(value[0]), reward=float(chunk == 2 and episode == 1),
                episode_start=chunk == 0, done=chunk == 2,
                case=dict(task="OpenDrawer", native_seed=episode), chunk_index=chunk))
    return records


def test_parallel_arrival_order_restores_episode_causality_before_gae():
    policy = actor()
    records = bank(policy)
    results = [dict(records=records[1::2]), dict(records=records[::2])]
    cases = [dict(task="OpenDrawer", native_seed=index) for index in range(2)]
    ordered = chronological_records(results, cases)
    assert [(row["case"]["native_seed"], row["chunk_index"]) for row in ordered] == [
        (episode, chunk) for episode in range(2) for chunk in range(3)]
    for broken in (ordered[:-1], ordered + [ordered[0]], [dict(ordered[0], done=True), *ordered[1:]]):
        with pytest.raises(ValueError, match="complete contiguous"):
            chronological_records([dict(records=broken)], cases)
    from fluffy_bassoon.meta_rl.ppo import generalized_advantages
    rollout = stack_records(ordered)
    first = generalized_advantages(rollout["rewards"], rollout["values"], rollout["ends"],
                                  gamma=.99, gae_lambda=.95)
    # A later task episode may not change the first episode's targets.
    rollout["rewards"][3:] += 100
    second = generalized_advantages(rollout["rewards"], rollout["values"], rollout["ends"],
                                   gamma=.99, gae_lambda=.95)
    for left, right in zip(first, second):
        torch.testing.assert_close(left[:3], right[:3], rtol=0, atol=0)


def test_complete_boundary_restores_exact_next_adam_update_and_rng():
    policy = actor()
    optimizer = torch.optim.Adam(policy.parameters(), lr=policy.config.learning_rate)
    generator = torch.Generator().manual_seed(371)
    update(policy, optimizer, stack_records(bank(policy)), generator=generator)
    saved = deepcopy(boundary_state(policy, optimizer, generator, arm="velocity", iteration=1,
        sequence=4, worker_states={0: dict(sequence=4)}, escrow=dict(train=1)))
    next_rollout = stack_records(bank(policy))
    expected = update(policy, optimizer, next_rollout, generator=generator)
    expected_policy = deepcopy(policy.state_dict())
    expected_rng = torch.rand(5)
    restored = actor()
    recovered_optimizer = torch.optim.Adam(restored.parameters(), lr=restored.config.learning_rate)
    recovered_generator = torch.Generator().manual_seed(999)
    assert restore_boundary(saved, restored, recovered_optimizer, recovered_generator,
                            arm="velocity") == (1, 4, {0: dict(sequence=4)})
    actual = update(restored, recovered_optimizer, next_rollout, generator=recovered_generator)
    torch.testing.assert_close(torch.rand(5), expected_rng, rtol=0, atol=0)
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, expected_policy[key], rtol=0, atol=0)
    assert actual["optimizer_steps"] == expected["optimizer_steps"]
    assert actual["final_metrics"] == expected["final_metrics"]
    for index, state in optimizer.state_dict()["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(value, recovered_optimizer.state_dict()["state"][index][key], rtol=0, atol=0)
    changed = dict(saved, spec=dict(saved["spec"], fraction=.2))
    with pytest.raises(ValueError, match="contract"):
        restore_boundary(changed, restored, recovered_optimizer, recovered_generator, arm="velocity")


def test_first_rollout_feature_fit_preserves_actor_law_and_original_value_labels():
    policy = actor()
    rows = bank(policy)
    result = normalize_first_bank(policy, rows)
    torch.testing.assert_close(policy.feature_mean, torch.tensor([.5, 1., 1.]), rtol=0, atol=0)
    for original, corrected in zip(rows, result):
        assert corrected["log_prob"] == original["log_prob"]
        assert corrected["pre_normalization_value"] == original["value"]
    with torch.no_grad():
        policy.actor[-1].bias.fill_(.01)
    with pytest.raises(ValueError, match="before actor learning"):
        normalize_first_bank(policy, rows)


def test_report_weights_tasks_equally_and_rejects_duplicate_episodes():
    episodes = [dict(case=dict(task="OpenDrawer", native_seed=index),
                     success_once=index < 8, success_at_end=index < 6) for index in range(10)]
    episodes.append(dict(case=dict(task="TurnOffStove", native_seed=91),
                         success_once=False, success_at_end=False))
    report = summarize_episodes(episodes)
    assert report["macro_ID_success_once"] == .4
    assert report["macro_ID_success_at_end"] == .3
    assert report["macro_OOD_success_once"] is None
    assert report["episodes"] == 11
    with pytest.raises(ValueError, match="twice"):
        summarize_episodes([*episodes, episodes[0]])


def test_actual_1152_coefficient_actor_update_and_next_update_restore():
    policy = make_policy("kv_all_layers", seed=91)
    optimizer = torch.optim.Adam(policy.parameters(), lr=policy.config.learning_rate)
    generator = torch.Generator().manual_seed(92)
    features = torch.randn((24, 2064), generator=torch.Generator().manual_seed(93))
    noise = torch.randn((24, 1152), generator=torch.Generator().manual_seed(94))

    def rollout():
        control, log_prob, value = policy.act(features, action_noise=noise)
        return dict(features=features[:, None], controls=control[:, None],
                    log_probs=log_prob[:, None], values=value[:, None],
                    rewards=torch.tensor([0., 0., 0., 0., 0., 1.] * 4)[:, None],
                    ends=torch.tensor([False] * 5 + [True])[:, None].repeat(4, 1))

    first = update(policy, optimizer, rollout(), generator=generator)
    assert first["optimizer_steps"] >= 1 and np.isfinite(first["final_metrics"]["approximate_kl"])
    saved = deepcopy(boundary_state(policy, optimizer, generator, arm="kv_all_layers",
        iteration=1, sequence=4, worker_states={}, escrow={}))
    second_rollout = rollout()
    expected = update(policy, optimizer, second_rollout, generator=generator)
    expected_weights = deepcopy(policy.state_dict())
    restore_boundary(saved, policy, optimizer, generator, arm="kv_all_layers")
    actual = update(policy, optimizer, second_rollout, generator=generator)
    assert actual["final_metrics"] == expected["final_metrics"]
    assert actual["optimizer_steps"] == expected["optimizer_steps"]
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value, expected_weights[key], rtol=0, atol=0)
