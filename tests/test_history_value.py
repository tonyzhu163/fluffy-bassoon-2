"""Numerical value objectives, causal histories and exact learner recovery."""
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.history_value import (
    HistoryValueLearner, ValueConfig, discounted_returns, expectile_loss,
    importance_weights, learning_step, next_values, sample_value_batch,
)
from fluffy_bassoon.meta_rl.vla_history import (
    HistoryConfig, FrozenHistoryController, export_policy, file_sha256, load_policy,
    save_dataset, validate_trial,
)


def trial(width=2, offset=0.):
    actions = np.arange(4 * width, dtype=np.float32).reshape(4, width) / (4 * width) + offset
    return dict(observations=np.arange(12, dtype=np.float32).reshape(4, 3) / 10 + offset,
        executed_noise=actions, target_noise=actions.copy(), rewards=np.array([0, 1, 0, 0], np.float32),
        dones=np.array([False, True, False, True]), episode_starts=np.array([True, False, True, False]),
        teacher_steps=np.array([0, 0, 1, 1], np.int64))


def learner(objective="dit", width=2):
    return HistoryValueLearner(HistoryConfig(3, width, width=8, layers=1, heads=2, context_length=8),
                               ValueConfig(objective))


def batch(model, *, count=3):
    return sample_value_batch([trial(model.actor.config.noise_dim)], model.actor,
        np.random.default_rng(3), batch_size=count, max_episodes=2, discount=.99)


def test_returns_and_bootstrap_never_cross_physical_boundaries():
    np.testing.assert_allclose(discounted_returns(np.array([0, 1, 3, 2]),
        np.array([False, True, False, True]), .5), [.5, 1, 4, 2])
    values = torch.tensor([[1.], [7.], [100.], [200.]])
    dones = torch.tensor([[False], [True], [False], [True]])
    valid = torch.ones_like(dones)
    torch.testing.assert_close(next_values(values, dones, valid), torch.tensor([[7.], [0.], [200.], [0.]]))
    with pytest.raises(ValueError, match="no next query"):
        next_values(values[:1], dones[:1], valid[:1])


def test_expectile_and_capped_weights_match_scalar_references():
    torch.testing.assert_close(expectile_loss(torch.tensor([-2., 3.]), .7), torch.tensor([1.2, 6.3]))
    weights = importance_weights(torch.tensor([-1., 0., 1., 1e20]), ValueConfig("ic_iql", maximum_weight=10))
    torch.testing.assert_close(weights, torch.tensor([np.exp(-1), 1, np.e, 10], dtype=torch.float32))
    assert not weights.requires_grad


@pytest.mark.parametrize("objective", ["dit", "ic_iql"])
def test_value_actions_are_executed_actions_not_dpt_labels(objective):
    data = trial()
    validate_trial(data, objective=objective)
    data["target_noise"] += 1
    with pytest.raises(ValueError, match="actually executed"):
        validate_trial(data, objective=objective)


def test_sampled_windows_finish_at_real_boundaries_and_returns_are_not_actor_inputs():
    model = learner()
    first = batch(model)
    for lane in range(first["valid"].shape[1]):
        end = int(first["valid"][:, lane].sum())
        assert first["dones"][end - 1, lane]
        assert not first["tokens"][0, lane, 3:-1].any()
    changed = {key: value.clone() for key, value in first.items()}
    changed["returns"][:] = 1e6
    left = model.actor_loss(first)[0]
    right = model.actor_loss(changed)[0]
    torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_actor_loss_matches_weighted_behavior_action_likelihood():
    model = learner()
    for critic, value in ((model.q, .3), (model.v, .1)):
        for parameter in critic.parameters():
            parameter.data.zero_()
        critic.head[-1].bias.data.fill_(value)
    data = batch(model)
    reference = model.actor.sequence_nll(data["tokens"], data["actions"], data["valid"], data["starts"])
    torch.testing.assert_close(model.actor_loss(data)[0], reference * np.exp(.2))


@pytest.mark.parametrize("objective", ["dit", "ic_iql"])
def test_iql_td_and_dit_mc_match_constant_critic_reference(objective):
    model = learner(objective)
    for critic, value in ((model.q, .2), (model.v, .1), (model.target_q, .3)):
        for parameter in critic.parameters():
            parameter.data.zero_()
        critic.head[-1].bias.data.fill_(value)
    data = batch(model)
    q, v, tq = torch.full_like(data["returns"], .2), torch.full_like(data["returns"], .1), torch.full_like(data["returns"], .3)
    td = data["rewards"] + .99 * next_values(v, data["dones"], data["valid"])
    if objective == "ic_iql":
        reference = ((q - td).square() + expectile_loss(tq - v, .7))[data["valid"]].mean()
    else:
        target_q = data["rewards"] + .99 * next_values(tq, data["dones"], data["valid"])
        reference = ((q - data["returns"]).square() + .1 * (q - target_q).square() +
            (v - data["returns"]).square() + .1 * (v - td).square())[data["valid"]].mean()
    torch.testing.assert_close(model.value_loss(data)[0], reference)


@pytest.mark.parametrize("objective", ["dit", "ic_iql"])
@pytest.mark.parametrize("phase", ["value", "actor"])
def test_execution_batches_preserve_logical_loss_and_update(objective, phase):
    torch.manual_seed(17)
    left = learner(objective); right = deepcopy(left)
    parameters = lambda model: [*model.q.parameters(), *model.v.parameters()] if phase == "value" else model.actor.parameters()
    a = torch.optim.AdamW(parameters(left), lr=1e-3, weight_decay=0.)
    b = torch.optim.AdamW(parameters(right), lr=1e-3, weight_decay=0.)
    data = batch(left)
    full = learning_step(left, a, data, phase=phase)
    split = learning_step(right, b, data, phase=phase, execution_batch_size=1)
    assert full["loss"] == pytest.approx(split["loss"], rel=1e-5)
    for x, y in zip(left.parameters(), right.parameters(), strict=True):
        torch.testing.assert_close(x, y, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("objective", ["dit", "ic_iql"])
def test_masked_nan_padding_cannot_change_value_loss_or_gradients(objective):
    model = learner(objective); copied = deepcopy(model)
    data = batch(model)
    padding = {}
    for key, values in data.items():
        extra = torch.zeros_like(values) if values.dtype == torch.bool else torch.full_like(values, float("nan"))
        padding[key] = torch.cat((values, extra), 0)
    loss = model.value_loss(data)[0]; loss.backward()
    other = copied.value_loss(padding)[0]; other.backward()
    torch.testing.assert_close(loss, other, atol=0, rtol=0)
    for x, y in zip(model.parameters(), copied.parameters(), strict=True):
        if x.grad is not None:
            torch.testing.assert_close(x.grad, y.grad, atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize("objective", ["dit", "ic_iql"])
@pytest.mark.parametrize("width", [32, 1152])
def test_both_interfaces_next_adam_recovery_and_actor_export(tmp_path, objective, width):
    torch.manual_seed(19)
    model = learner(objective, width)
    model.fit_normalization([trial(width)])
    optimizer = torch.optim.AdamW([*model.q.parameters(), *model.v.parameters()], lr=1e-3)
    data = batch(model, count=1)
    learning_step(model, optimizer, data, phase="value")
    path = tmp_path / "recovery.pt"
    rng = np.random.default_rng(91)
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), rng=rng.bit_generator.state), path)
    copied = learner(objective, width)
    saved = torch.load(path, weights_only=False)
    copied.load_state_dict(saved["model"])
    restored = torch.optim.AdamW([*copied.q.parameters(), *copied.v.parameters()], lr=1e-3)
    restored.load_state_dict(saved["optimizer"])
    rng2 = np.random.default_rng(); rng2.bit_generator.state = saved["rng"]
    a = sample_value_batch([trial(width)], model.actor, rng, batch_size=1, max_episodes=2, discount=.99)
    b = sample_value_batch([trial(width)], copied.actor, rng2, batch_size=1, max_episodes=2, discount=.99)
    learning_step(model, optimizer, a, phase="value")
    learning_step(copied, restored, b, phase="value")
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, copied.state_dict()[key], rtol=0, atol=0)
    candidate = tmp_path / "candidate.pt"
    export_policy(candidate, model.actor, manifest=dict(objective=objective, value_config=asdict(model.config)))
    inference, _ = load_policy(candidate, expected_sha256=file_sha256(candidate))
    one = FrozenHistoryController(model.actor); two = FrozenHistoryController(inference)
    np.testing.assert_array_equal(one.act(np.ones(3, np.float32), deterministic=True),
                                  two.act(np.ones(3, np.float32), deterministic=True))
