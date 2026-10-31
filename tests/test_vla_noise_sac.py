"""SAC math, exact recovery and deployment isolation; run on allocated Linux."""
from copy import deepcopy
import math

import numpy as np
import pytest
import torch
from torch.distributions import Normal, TanhTransform, TransformedDistribution

from fluffy_bassoon.meta_rl.vla_noise_sac import NoiseSAC, SACConfig, sac_target, squashed_log_prob


def learner(**changes):
    options = dict(hidden_dims=(8, 8), batch_size=4, warmup_chunks=4, utd=2,
                   replay_capacity=12, layer_norm=False, tau=.25)
    options.update(changes)
    return NoiseSAC(3, 2, device="cpu", seed=7, config=SACConfig(**options))


def transition(agent, index):
    obs = np.asarray([index / 10, -.1, .2], np.float32)
    action = agent.sample_warmup(obs)
    agent.observe(obs, action, float(index % 2), obs + .01,
                  index % 3 == 0, index % 3 == 1, primitive_steps=7)


def equal_state(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            if key not in {"update_seconds"}:
                equal_state(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            equal_state(x, y)
    else:
        assert a == b


def test_sac_bellman_target_matches_independent_scalar_values():
    reward = torch.tensor([[1.], [-2.], [0.]])
    terminal = torch.tensor([[0.], [1.], [0.]])
    soft_value = torch.tensor([[3.], [100.], [-4.]])
    expected = torch.tensor([[1. + .9 * 3.], [-2.], [.9 * -4.]])
    torch.testing.assert_close(sac_target(reward, terminal, soft_value, .9), expected)


def test_initialization_and_actions_preserve_callers_random_streams():
    cpu_before = torch.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    numpy_before = deepcopy(np.random.get_state())
    agent = learner()
    agent.act(np.zeros(3, np.float32))
    agent.sample_warmup()
    torch.testing.assert_close(torch.get_rng_state(), cpu_before, rtol=0, atol=0)
    for before, after in zip(cuda_before, torch.cuda.get_rng_state_all() if cuda_before else [], strict=True):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    equal_state(np.random.get_state(), numpy_before)


def test_squashed_density_matches_independent_transformed_distribution():
    mean = torch.tensor([[.2, -.3]], dtype=torch.float64)
    log_std = torch.tensor([[.1, -.2]], dtype=torch.float64)
    raw = torch.tensor([[-2., .4]], dtype=torch.float64)
    reference = TransformedDistribution(Normal(mean, log_std.exp()), [TanhTransform()])
    expected = reference.log_prob(raw.tanh()).sum(-1, keepdim=True)
    torch.testing.assert_close(squashed_log_prob(mean, log_std, raw), expected)
    assert torch.isfinite(squashed_log_prob(mean, log_std, torch.full_like(raw, 50.))).all()


def test_warmup_threshold_counts_all_interactions_before_updates():
    agent = learner()
    for i in range(4):
        transition(agent, i)
        assert agent.update() == {}
    assert agent.environment_steps == 4 and agent.primitive_steps == 28
    assert not agent.ready_to_update
    transition(agent, 4)
    metrics = agent.update()
    assert metrics["gradient_steps"] == 2
    assert metrics["optimizer_steps"] == 6
    assert agent.updates == 2


def test_twin_critic_target_is_frozen_and_soft_updated():
    agent = learner(warmup_chunks=0, batch_size=2)
    for i in range(2):
        transition(agent, i)
    old_target = [p.detach().clone() for p in agent.target_critic.parameters()]
    old_actor = [p.detach().clone() for p in agent.actor.parameters()]
    metrics = agent.update(gradient_steps=1)
    assert all(math.isfinite(value) for value in metrics.values())
    assert any(not torch.equal(old, current) for old, current in zip(old_actor, agent.actor.parameters()))
    assert all(not p.requires_grad and p.grad is None for p in agent.target_critic.parameters())
    for before, target, current in zip(old_target, agent.target_critic.parameters(),
                                       agent.critic.parameters(), strict=True):
        torch.testing.assert_close(target, .75 * before + .25 * current)


def test_complete_recovery_reproduces_next_action_sample_update_and_replay(tmp_path):
    source = learner(warmup_chunks=0)
    for i in range(7):
        transition(source, i)
    source.update(gradient_steps=2)
    source.save(tmp_path / "learner.pt")
    restored = learner(warmup_chunks=0)
    restored.load(tmp_path / "learner.pt")
    obs = np.asarray([.1, .2, -.3], np.float32)
    np.testing.assert_array_equal(source.act(obs), restored.act(obs))
    np.testing.assert_array_equal(source.sample_warmup(obs), restored.sample_warmup(obs))
    source.update(gradient_steps=1)
    restored.update(gradient_steps=1)
    equal_state(source.state_dict(), restored.state_dict())


def test_independent_target_restore_cannot_change_source_or_other_trial():
    source = learner(warmup_chunks=0)
    for i in range(5):
        transition(source, i)
    source.update(gradient_steps=1)
    checkpoint = source.state_dict()
    snapshot = deepcopy(checkpoint)
    target = learner(warmup_chunks=0)
    target.load_state_dict(checkpoint)
    target.begin_target_adaptation()
    transition(target, 8)
    target.update(gradient_steps=1)
    equal_state(checkpoint, snapshot)
    second = learner(warmup_chunks=0)
    second.load_state_dict(checkpoint)
    equal_state(source.state_dict(), second.state_dict())


def test_fresh_target_replay_updates_immediately_without_weakening_source_warmup():
    source = learner(warmup_chunks=0)
    for i in range(5):
        transition(source, i)
    source.update(gradient_steps=1)
    snapshot = source.state_dict()
    target = learner(warmup_chunks=0)
    target.load_state_dict(snapshot)
    convention = target.begin_target_adaptation()
    assert convention["source_replay_rows"] == 5
    assert convention["source_replay_excluded"] and len(target.replay) == 0
    assert not target.ready_to_update and not target.warming_up
    equal_state(target.actor.state_dict(), source.actor.state_dict())
    equal_state(target.actor_optimizer.state_dict(), source.actor_optimizer.state_dict())
    assert target.environment_steps == source.environment_steps
    transition(target, 9)
    assert len(target.replay) == 1 < target.config.batch_size
    assert target.ready_to_update
    metrics = target.update(gradient_steps=2)
    assert metrics["gradient_steps"] == 2
    assert target.target_counters()["primitive_steps"] == 7
    assert target.target_counters()["updates"] == 2
    assert target.target_counters()["optimizer_steps"] == 6
    assert target.target_counters()["replay_mode"] == "fresh_target"
    equal_state(snapshot, source.state_dict())
    with pytest.raises(ValueError, match="Restore"):
        target.begin_target_adaptation()


def test_target_recovery_preserves_adaptation_mode_and_next_update():
    first = learner(warmup_chunks=0)
    for i in range(5):
        transition(first, i)
    first.update(gradient_steps=1)
    first.begin_target_adaptation()
    transition(first, 8)
    saved = first.state_dict()
    second = learner(warmup_chunks=0)
    second.load_state_dict(saved)
    assert second.ready_to_update and second.target_replay_mode == "fresh_target"
    first.update(gradient_steps=1)
    second.update(gradient_steps=1)
    equal_state(first.state_dict(), second.state_dict())


def test_target_adaptation_cannot_be_claimed_without_source_training():
    with pytest.raises(ValueError, match="source learner"):
        learner().begin_target_adaptation()


def test_optimizer_sampling_does_not_consume_paired_interaction_random_stream():
    agent = learner(warmup_chunks=0)
    for i in range(5):
        transition(agent, i)
    action_rng = agent.torch_rng.get_state().clone()
    update_rng = agent.train_torch_rng.get_state().clone()
    agent.update(gradient_steps=1)
    torch.testing.assert_close(agent.torch_rng.get_state(), action_rng, rtol=0, atol=0)
    assert not torch.equal(agent.train_torch_rng.get_state(), update_rng)


def test_truncation_preserves_terminal_next_observation_and_bootstrap_mask():
    agent = learner()
    obs, following = np.zeros(3, np.float32), np.ones(3, np.float32)
    agent.observe(obs, np.zeros(2), 1., following, False, True, primitive_steps=2)
    np.testing.assert_array_equal(agent.replay[0][3], following)
    assert agent.replay[0][4] is False and agent.replay[0][5] is True


def test_ring_replay_and_validation():
    agent = learner(replay_capacity=4)
    for i in range(7):
        transition(agent, i)
    assert len(agent.replay) == 4 and agent.replay_position == 3
    assert agent.primitive_steps == 49
    with pytest.raises(ValueError, match="primitive-step"):
        agent.observe(np.zeros(3), np.zeros(2), 0., np.ones(3), False, primitive_steps=1.5)
    with pytest.raises(ValueError, match="configuration"):
        SACConfig(batch_size=256, replay_capacity=1)
    with pytest.raises(ValueError, match="replay"):
        learner().load_state_dict(learner().state_dict(include_replay=False))
