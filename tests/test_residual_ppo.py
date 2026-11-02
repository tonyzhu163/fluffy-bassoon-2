"""Bounded action residual and memoryless residual PPO; run on allocated Linux resources."""

import json
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.residual_ppo import (HistoryResidualController, ResidualActorCritic,
    ResidualPPOConfig, collect_episodes, history_trial, ppo_config_record, stack_records, update)
from fluffy_bassoon.meta_rl.vla_history import (FrozenHistoryController, HistoryConfig,
    HistoryNoisePolicy, validate_trial)
from fluffy_bassoon.rl.action_residual import ResidualBound, action_std_from_norm_stats


ARM = (0, 1, 2, 3, 4, 5)


def bound(fraction=.25):
    return ResidualBound.from_std(np.arange(1, 13, dtype=float), indices=ARM, fraction=fraction, action_dim=12)


def test_zero_control_is_identity_and_only_arm_dims_move():
    rng = np.random.default_rng(0)
    chunk = rng.normal(size=(50, 12))
    b = bound()
    assert np.array_equal(b.apply(chunk, np.zeros(6)), chunk)
    moved = b.apply(chunk, np.full(6, 100.))
    np.testing.assert_allclose(moved[:, :6] - chunk[:, :6], np.broadcast_to(b.scale, (50, 6)), rtol=1e-12)
    assert np.array_equal(moved[:, 6:], chunk[:, 6:])


def test_residual_never_exceeds_bound_and_matches_torch():
    b = bound(.5)
    control = np.random.default_rng(1).normal(scale=50, size=(7, 6))
    delta = b.delta(control)
    assert (np.abs(delta) <= np.asarray(b.scale) + 1e-12).all()
    torch.testing.assert_close(b.delta(torch.as_tensor(control)), torch.as_tensor(delta))
    batched = b.apply(torch.zeros(7, 50, 12, dtype=torch.float64), torch.as_tensor(control))
    torch.testing.assert_close(batched[:, 0, :6], torch.as_tensor(delta))


def test_bound_validation_and_norm_stats(tmp_path):
    with pytest.raises(ValueError):
        ResidualBound(12, (0, 0), (1., 1.))
    with pytest.raises(ValueError):
        ResidualBound.from_std(np.zeros(12), indices=ARM, fraction=.25, action_dim=12)
    with pytest.raises(ValueError):
        bound().apply(np.zeros((50, 12)), np.zeros(5))
    path = tmp_path / "norm_stats.json"
    std = list(np.linspace(.1, 3.2, 32))
    path.write_text(json.dumps({"norm_stats": {"actions": {"mean": [0.] * 32, "std": std}}}))
    b = ResidualBound.from_std(action_std_from_norm_stats(path), indices=ARM, fraction=.25, action_dim=12)
    np.testing.assert_allclose(b.scale, .25 * np.asarray(std[:6]))


class LineMotor:
    """Frozen 'VLA' that always proposes zero motion; features are the position."""

    def __init__(self, environment):
        self.environment = environment

    def infer(self, observation):
        return np.zeros((5, 12)), np.asarray([observation, 1.], np.float32), 0


class LineEnvironment:
    """Reach a target at +1 within four chunks; terminal reward is success."""

    def reset(self):
        self.position, self.steps = 0., 0
        return self.position

    def step_chunk(self, chunk):
        self.position += float(chunk[:, 0].sum())
        self.steps += 1
        done = self.steps == 4
        return self.position, done, dict(success=abs(self.position - 1.) < .25)


def make_policy():
    torch.manual_seed(0)
    return ResidualActorCritic(ResidualPPOConfig(feature_dim=2, residual_dim=6, width=32, minibatch_size=16))


def test_initial_deterministic_policy_reproduces_frozen_motor():
    policy, environment = make_policy(), LineEnvironment()
    records = collect_episodes(policy, bound(), LineMotor(environment), environment,
                               lambda info: float(info["success"]), episodes=2, deterministic=True)
    assert all(np.array_equal(record["control"], np.zeros(6)) for record in records)
    assert [record["episode_start"] for record in records[:4]] == [True, False, False, False]
    assert [record["reward"] for record in records] == [0.] * 8


def test_ppo_learns_to_reach_target_and_exports_history():
    policy, environment = make_policy(), LineEnvironment()
    # Five steps per chunk at scale 0.25 * std(=1) reach about +1 in four chunks with tanh(u) near 0.2.
    b = bound()
    motor, reward = LineMotor(environment), lambda info: float(info["success"])
    policy.fit_feature_normalization(np.asarray([[0., 1.], [1., 1.]], np.float32))
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-3)
    generator = torch.Generator().manual_seed(0)
    history, before = [], None
    for iteration in range(60):
        records = collect_episodes(policy, b, motor, environment, reward, episodes=16, generator=generator)
        success = np.mean([record["reward"] for record in records if record["done"]])
        before = success if before is None else before
        history.append((iteration, records))
        update(policy, optimizer, stack_records(records), generator=generator)
    final = collect_episodes(policy, b, motor, environment, reward, episodes=16, deterministic=True)
    assert np.mean([record["reward"] for record in final if record["done"]]) > max(before, .5)
    trial = history_trial(history)
    validate_trial(trial, objective="ad")


def test_history_controller_adapter_applies_bound():
    b = bound()
    policy = HistoryNoisePolicy(HistoryConfig(2, 6, width=8, heads=2, layers=1, context_length=8))
    adapter = HistoryResidualController(FrozenHistoryController(policy), b)
    chunk = np.zeros((5, 12))
    executed, control = adapter.act(chunk, np.asarray([0., 1.], np.float32), deterministic=True)
    np.testing.assert_allclose(executed[0, :6], b.delta(control))
    adapter.observe(control, 1., True)
    with pytest.raises(ValueError):
        HistoryResidualController(FrozenHistoryController(
            HistoryNoisePolicy(HistoryConfig(2, 5, width=8, heads=2, layers=1, context_length=8))), b)


@pytest.mark.parametrize("target", [0., -1., float("nan"), float("inf"), True, "0.02"])
def test_kl_target_requires_finite_positive_number(target):
    with pytest.raises(ValueError, match="target_kl"):
        ResidualPPOConfig(2, 6, target_kl=target)


def diagnostic_batch(policy, size=7):
    """A complete rollout with an unequal final minibatch and varied returns."""
    features = torch.stack((torch.linspace(-1, 1, size), torch.ones(size)), -1)
    controls = torch.zeros(size, policy.config.residual_dim)
    with torch.no_grad():
        log_probs, _, values = policy.evaluate(features, controls)
    return dict(features=features[:, None], controls=controls[:, None],
        log_probs=log_probs[:, None], values=values[:, None],
        rewards=torch.linspace(0, 1, size)[:, None], ends=torch.ones(size, 1, dtype=torch.bool))


def test_epoch_kl_stop_records_current_full_batch_and_keeps_completed_epoch():
    config = ResidualPPOConfig(2, 6, width=8, epochs=10, minibatch_size=3, target_kl=.02)
    torch.manual_seed(12)
    policy = ResidualActorCritic(config)
    batch = diagnostic_batch(policy)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.)
    # Make each optimizer step shift the Gaussian mean by a known amount.
    # The resulting full-rollout sampled KL has an independent closed form.
    def shift_mean():
        with torch.no_grad():
            policy.actor[-1].bias.add_(.1)
    optimizer.step = shift_mean
    callbacks = []
    metrics = update(policy, optimizer, batch, generator=torch.Generator().manual_seed(8),
                     epoch_callback=callbacks.append)
    assert metrics["early_stopped"] and metrics["stop_reason"] == "target_kl"
    assert metrics["epochs_completed"] == 1 and metrics["optimizer_steps"] == 3
    assert metrics["samples"] == metrics["samples_processed"] == 7
    assert len(callbacks) == 1 and callbacks == metrics["epoch_metrics"]
    # Three steps: mean=.3, std=.5, six dimensions, actions fixed at zero.
    log_ratio = -6 * .3**2 / (2 * .5**2)
    expected = np.expm1(log_ratio) - log_ratio
    assert metrics["final_metrics"]["approximate_kl"] == pytest.approx(expected, rel=1e-5)
    assert callbacks[0]["approximate_kl"] == metrics["final_metrics"]["approximate_kl"]
    assert callbacks[0]["kl_threshold_exceeded"]
    torch.testing.assert_close(policy.actor[-1].bias, torch.full((6,), .3))
    assert not policy.training
    json.dumps(metrics, allow_nan=False)


def test_disabled_and_unreached_kl_target_preserve_adam_and_rng():
    torch.manual_seed(4)
    policy = ResidualActorCritic(ResidualPPOConfig(2, 6, width=8, epochs=3, minibatch_size=3))
    second = deepcopy(policy)
    second.config = replace(second.config, target_kl=1e6)
    batch = diagnostic_batch(policy)
    optimizers = [torch.optim.Adam(model.parameters(), lr=3e-4) for model in (policy, second)]
    generators = [torch.Generator().manual_seed(9) for _ in range(2)]
    rng_before = torch.get_rng_state()
    results = [update(model, optimizer, batch, generator=generator)
               for model, optimizer, generator in zip((policy, second), optimizers, generators)]
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)
    for key, state in optimizers[0].state_dict()["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, optimizers[1].state_dict()["state"][key][name], rtol=0, atol=0)
    torch.testing.assert_close(generators[0].get_state(), generators[1].get_state(), rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)
    for result in results:
        assert not result["early_stopped"] and result["stop_reason"] == "epochs_completed"
        assert result["epochs_completed"] == 3 and result["optimizer_steps"] == 9
        assert result["samples_processed"] == 21
        with torch.no_grad():
            log_probs, _, values = policy.evaluate(batch["features"][:, 0], batch["controls"][:, 0])
            log_ratio = log_probs - batch["log_probs"][:, 0]
            expected = (torch.expm1(log_ratio) - log_ratio).mean()
            returns = batch["rewards"][:, 0]  # Every transition terminates.
            explained = 1 - (returns - values).var(unbiased=False) / returns.var(unbiased=False)
        assert result["final_metrics"]["approximate_kl"] == pytest.approx(float(expected), abs=1e-8)
        assert result["final_metrics"]["explained_variance"] == pytest.approx(float(explained), abs=1e-7)
        json.dumps(result, allow_nan=False)
    assert "target_kl" not in ppo_config_record(policy.config)
    assert ppo_config_record(second.config)["target_kl"] == 1e6


def test_already_exceeded_kl_does_not_step_optimizer_or_sampler():
    torch.manual_seed(4)
    policy = ResidualActorCritic(ResidualPPOConfig(2, 6, width=8, epochs=10, target_kl=.02))
    batch = diagnostic_batch(policy)
    with torch.no_grad():
        policy.actor[-1].bias.add_(.3)
    expected = deepcopy(policy.state_dict())
    optimizer = torch.optim.Adam(policy.parameters())
    generator = torch.Generator().manual_seed(9)
    rng_before = generator.get_state()
    result = update(policy, optimizer, batch, generator=generator)
    assert result["early_stopped"] and result["epochs_completed"] == result["optimizer_steps"] == 0
    assert not result["epoch_metrics"] and not optimizer.state
    for key, value in expected.items():
        torch.testing.assert_close(policy.state_dict()[key], value, rtol=0, atol=0)
    torch.testing.assert_close(generator.get_state(), rng_before, rtol=0, atol=0)


def test_constant_returns_have_json_null_explained_variance():
    policy = make_policy()
    batch = diagnostic_batch(policy)
    batch["rewards"].zero_()
    result = update(policy, torch.optim.SGD(policy.parameters(), lr=0.), batch)
    assert result["final_metrics"]["explained_variance"] is None
    json.dumps(result, allow_nan=False)
