"""RL² full-trial credit, actual-feedback replay and complete recovery.

Run model computation only on allocated Linux resources.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.ppo import generalized_advantages, update_ppo
from fluffy_bassoon.meta_rl.vla_rl2 import (FrozenRL2Controller, RL2Config,
    collect_rl2_trial, export_rl2_policy, load_rl2_policy, make_rl2_policy, train_rl2)


def small_policy(seed=7):
    return make_rl2_policy(RL2Config(observation_dim=3, noise_dim=2, width=8,
                                    layers=1, heads=2, context_length=8), seed=seed)


SIM = SimpleNamespace(metadata=dict(settle_steps=10))


def features(observation):
    return observation["public"]


def fake_episode(sim, motor, choose, *, initial_state, scene_seed, chunk, bounded, feedback):
    rows = []
    for index in range(2):
        observation = dict(public=np.asarray([initial_state / 10, .2, index / 2], np.float32))
        actual = choose(observation)
        done = index == 1
        reward = 0. if done else -1.
        feedback(actual, reward, done)
        rows.append(dict(done=done, primitive_steps=1))
    return rows, dict(initial_state=initial_state, scene_seed=scene_seed, success=True,
        chunks=2, primitive_steps=2, initial_observation_sha256=f"{initial_state}:{scene_seed}")


def collect(policy, **kwargs):
    return collect_rl2_trial(policy, SIM, None, features, initial_state=1, scene_seed=3,
        attempts=2, chunk=1, episode_runner=fake_episode, **kwargs)


def test_collector_replays_exact_likelihood_and_only_previous_actual_feedback():
    policy = small_policy()
    rollout, receipt = collect(policy)
    token = rollout["observations"][:, 0]
    assert rollout["observations"].shape == (4, 1, 8)
    assert np.array_equal(token[0, 3:].numpy(), [0., 0., 0., 0., 1.])
    assert torch.equal(token[1, 3:5], rollout["raw_actions"][0, 0].tanh())
    assert torch.equal(token[2, 3:5], rollout["raw_actions"][1, 0].tanh())
    assert np.array_equal(token[2, -3:].numpy(), [0., 1., 1.])
    assert rollout["trial_ends"][:, 0].tolist() == [False, False, False, True]
    assert rollout["attempt_ends"][:, 0].tolist() == [False, True, False, True]
    assert not rollout["resets"].any()
    assert receipt["primitive_steps"] == 4 and receipt["settling_steps"] == 20
    with torch.no_grad():
        replay = policy.evaluate_sequence(rollout["observations"], rollout["raw_actions"], rollout["resets"])[0]
    torch.testing.assert_close(replay, rollout["log_probs"], rtol=0, atol=0)


def test_episode_reset_erases_boundary_feedback_and_memory():
    policy = small_policy()
    rollout, _ = collect(policy, mode="episode", deterministic=True)
    assert rollout["resets"][:, 0].tolist() == [True, False, True, False]
    assert not rollout["observations"][2, 0, 3:-1].any()
    # Same scene and current observation reproduce the first action after reset.
    torch.testing.assert_close(rollout["raw_actions"][0], rollout["raw_actions"][2], rtol=0, atol=0)


def test_full_trial_return_and_graph_cross_physical_episode_boundary():
    policy = small_policy()
    rollout, _ = collect(policy)
    rewards = torch.tensor([[0.], [0.], [0.], [1.]])
    zero = torch.zeros_like(rewards)
    advantages, _ = generalized_advantages(rewards, zero, rollout["trial_ends"])
    assert advantages[:, 0].tolist() == [1., 1., 1., 1.]
    episode_advantages, _ = generalized_advantages(rewards, zero, rollout["attempt_ends"])
    assert episode_advantages[:, 0].tolist() == [0., 0., 1., 1.]
    tokens = rollout["observations"].clone().requires_grad_()
    values = policy.evaluate_sequence(tokens, rollout["raw_actions"], rollout["resets"])[2]
    values[-1].sum().backward()
    assert tokens.grad[0].abs().sum() > 0
    policy.zero_grad(set_to_none=True)
    tokens = rollout["observations"].clone().requires_grad_()
    resets = rollout["resets"].clone()
    resets[2] = True
    values = policy.evaluate_sequence(tokens, rollout["raw_actions"], resets)[2]
    values[-1].sum().backward()
    assert tokens.grad[:2].abs().sum() == 0


def test_actual_noise_remapping_is_rejected_before_false_ppo_likelihood():
    def remapped(sim, motor, choose, **kwargs):
        proposal = choose(dict(public=np.zeros(3, np.float32)))
        kwargs["feedback"](proposal + .1, 0., True)
        raise AssertionError("must reject the remapped coordinate")
    with pytest.raises(ValueError, match="actual unchanged"):
        collect_rl2_trial(small_policy(), SIM, None, features, initial_state=0, scene_seed=0,
            attempts=1, chunk=1, episode_runner=remapped)


def test_ppo_update_is_finite_and_exported_deployment_weights_stay_frozen(tmp_path):
    policy = small_policy()
    rollout, _ = collect(policy)
    before = {key: value.clone() for key, value in policy.state_dict().items()}
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    metrics = update_ppo(policy, optimizer, rollout, epochs=1, minibatch_trials=1,
                         bounded_entropy=True, episode_returns=False)
    assert metrics["gradient_norm"] > 0 and np.isfinite(metrics["loss"])
    assert any(not torch.equal(value, before[key]) for key, value in policy.state_dict().items())
    digest = export_rl2_policy(tmp_path / "policy.pt", policy, manifest={"status": "candidate"})
    loaded, _ = load_rl2_policy(tmp_path / "policy.pt", expected_sha256=digest)
    expected = {key: value.clone() for key, value in loaded.state_dict().items()}
    controller = FrozenRL2Controller(loaded)
    obs = np.zeros(3, np.float32)
    action = controller.act(obs)
    assert (np.abs(action) <= 1).all()
    controller.observe(action, -1., True)
    controller.act(obs)
    for key, value in loaded.state_dict().items():
        assert torch.equal(value, expected[key])
    assert all(not value.requires_grad for value in loaded.parameters())


def train_arguments(tmp_path, output_name):
    return dict(train_states=[0, 1], attempts=2, chunk=1, episode_horizon=2,
        max_primitive_steps=8, updates=2, output_dir=tmp_path / output_name,
        heldout_states=[2, 3], source_contract=dict(source_commit="b" * 40, feature_contract="fixed public features",
                             shared_prior_primitive_steps=100), epochs=1, seed=19,
        episode_runner=fake_episode)


def test_interrupted_training_recovers_next_sampler_action_and_optimizer_update(tmp_path):
    uninterrupted = small_policy(seed=19)
    expected, summary = train_rl2(uninterrupted, SIM, None, features, **train_arguments(tmp_path, "complete"))
    kwargs = train_arguments(tmp_path, "interrupted")
    calls = 0
    def interrupted(*args, **kw):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("simulated interruption between complete trials")
        return fake_episode(*args, **kw)
    kwargs["episode_runner"] = interrupted
    with pytest.raises(RuntimeError, match="simulated interruption"):
        train_rl2(small_policy(seed=19), SIM, None, features, **kwargs)
    kwargs["episode_runner"] = fake_episode
    kwargs["resume"] = kwargs["output_dir"] / "latest.pt"
    actual, resumed_summary = train_rl2(small_policy(seed=999), SIM, None, features, **kwargs)
    assert resumed_summary["updates"] == summary["updates"] == 2
    assert resumed_summary["counters"] == summary["counters"]
    assert summary["counters"]["environment_steps"] == 8
    assert summary["counters"]["optimizer_steps"] == 2
    for key, value in expected.state_dict().items():
        torch.testing.assert_close(value, actual.state_dict()[key], rtol=0, atol=0)
    expected_trace = torch.load(tmp_path / "complete" / "traces" / "trial-000002.pt", weights_only=True)
    actual_trace = torch.load(tmp_path / "interrupted" / "traces" / "trial-000002.pt", weights_only=True)
    assert expected_trace["receipt"] == actual_trace["receipt"]
    for key, value in expected_trace["rollout"].items():
        assert torch.equal(value, actual_trace["rollout"][key])


def test_invalid_budget_is_rejected_before_simulator_work(tmp_path):
    kwargs = train_arguments(tmp_path, "invalid")
    kwargs["max_primitive_steps"] = 3
    with pytest.raises(ValueError, match="complete-trial resource budget"):
        train_rl2(small_policy(), SIM, None, features, **kwargs)
    assert not kwargs["output_dir"].exists()
