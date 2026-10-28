"""PPO vectorization parity, cached-history replay and explicit gradient boundaries."""

from copy import deepcopy
import os

import pytest
import torch
from gymnasium.spaces.utils import flatdim

from fluffy_bassoon.meta_rl.environments import HiddenGainReach, HiddenRewardBandit
from fluffy_bassoon.meta_rl.ppo import (GRUCore, PersistentCTMVectorCore, RecurrentActorCritic,
                                      collect_trials, replay_windows, update_ppo)
from fluffy_bassoon.meta_rl.transformer_xl import TransformerXLCore


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    os.environ.get("FLUFFY_META_RL_CUDA_QUALIFICATION") != "1", reason="Requires a probed allocated GPU"))]


def learner(env, kind, device="cpu"):
    torch.set_num_threads(1)
    inputs = int(flatdim(env.observation_space))
    if kind == "gru":
        core = GRUCore(inputs, width=8)
    elif kind == "persistent_ctm":
        core = PersistentCTMVectorCore(inputs, width=8, ticks=2, history_length=3,
                                       synapse_depth=2, heads=2, bounded_synchronisation=True)
    else:
        core = TransformerXLCore(inputs, width=8, memory_length=4, heads=2)
    return RecurrentActorCritic(core, env.action_space).to(device).double().eval()


def serial_reference(policy, observations, actions, resets, initial_state=None):
    state, rows = initial_state, []
    for i in range(len(observations)):
        _, _, log_prob, entropy, value, state = policy.step(observations[i], state,
            raw_action=actions[i], reset=resets[i])
        rows.append((log_prob, entropy, value))
    return tuple(torch.stack([row[i] for row in rows]) for i in range(3))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["gru", "persistent_ctm", "transformer_xl"])
@pytest.mark.parametrize("continuous", [False, True])
def test_batched_heads_preserve_serial_outputs_and_full_history_gradients(device, kind, continuous):
    torch.manual_seed(91)
    env = HiddenGainReach(attempts=2, horizon=3) if continuous else HiddenRewardBandit(attempts=6)
    optimized = learner(env, kind, device)
    original = deepcopy(optimized)
    inputs = torch.randn(6, 3, optimized.core.input_dim, device=device, dtype=torch.float64)
    actions = (torch.randn(6, 3, 2, device=device, dtype=torch.float64) if continuous else
               torch.randint(env.action_space.n, (6, 3), device=device))
    resets = torch.zeros(6, 3, device=device, dtype=torch.bool)
    resets[3, 0] = True
    outputs = optimized.evaluate_sequence(inputs, actions, resets)
    expected = serial_reference(original, inputs, actions, resets)
    for a, b in zip(outputs, expected):
        torch.testing.assert_close(a, b, rtol=1e-8, atol=1e-10)
    sum(x.square().mean() for x in outputs).backward()
    sum(x.square().mean() for x in expected).backward()
    for a, b in zip(optimized.parameters(), original.parameters()):
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["gru", "persistent_ctm", "transformer_xl"])
@pytest.mark.parametrize("reset_attempts", [False, True])
def test_cached_windows_replay_carried_history_and_preserve_optimizer_work(device, kind, reset_attempts):
    torch.manual_seed(18)
    envs = [HiddenGainReach(attempts=3, horizon=4) for _ in range(3)]
    policy = learner(envs[0], kind, device)
    rollout = collect_trials(policy, envs, [10, 11, 12], replay_chunk_length=3,
                             reset_each_attempt=reset_attempts)
    windows, states, chunks = replay_windows(rollout, 3)
    assert chunks == 4 and windows["observations"].shape[:2] == (3, 12)
    with torch.no_grad():
        log_probs, _, values = policy.evaluate_sequence(windows["observations"], windows["raw_actions"],
                                                       windows["resets"], initial_state=states)
    torch.testing.assert_close(log_probs, windows["log_probs"], rtol=1e-7, atol=1e-9)
    torch.testing.assert_close(values, windows["values"], rtol=1e-7, atol=1e-9)
    metrics = update_ppo(policy, torch.optim.Adam(policy.parameters(), lr=1e-3), rollout,
                         epochs=2, minibatch_trials=2, replay_chunk_length=3)
    assert metrics["optimizer_steps"] == 4
    assert metrics["forward_samples"] == metrics["active_action_samples"] == 72


@pytest.mark.parametrize("kind", ["gru", "persistent_ctm", "transformer_xl"])
def test_one_full_length_window_preserves_the_exact_ppo_objective(kind):
    torch.manual_seed(27)
    envs = [HiddenGainReach(attempts=2, horizon=3) for _ in range(3)]
    full = learner(envs[0], kind)
    cached = deepcopy(full)
    rollout = collect_trials(full, envs, [3, 4, 5], replay_chunk_length=6)
    for policy, length in ((full, 0), (cached, 6)):
        torch.manual_seed(34)
        update_ppo(policy, torch.optim.Adam(policy.parameters(), lr=1e-3), rollout,
                    epochs=2, minibatch_trials=2, replay_chunk_length=length)
    for a, b in zip(full.parameters(), cached.parameters()):
        torch.testing.assert_close(a, b, rtol=1e-8, atol=1e-10)


def test_chunk_boundary_detaches_credit_without_resetting_carried_state():
    torch.manual_seed(32)
    envs = [HiddenGainReach(attempts=2, horizon=3)]
    policy = learner(envs[0], "gru")
    rollout = collect_trials(policy, envs, [4], replay_chunk_length=3)
    rollout["observations"].requires_grad_()
    windows, states, _ = replay_windows(rollout, 3)
    _, _, value = policy.evaluate_sequence(windows["observations"][:, 1:2],
        windows["raw_actions"][:, 1:2], windows["resets"][:, 1:2], initial_state=states[1:2])
    value[-1].sum().backward()
    assert rollout["observations"].grad[:3].abs().sum() == 0
    assert rollout["observations"].grad[3:].abs().sum() > 0
    assert not torch.equal(states[1], policy.core.initial_state(1)[0])


@pytest.mark.parametrize("length", [-1, 4, True])
def test_invalid_chunk_lengths_are_rejected(length):
    envs = [HiddenGainReach(attempts=2, horizon=3)]
    with pytest.raises(ValueError, match="chunk length"):
        collect_trials(learner(envs[0], "gru"), envs, [4], replay_chunk_length=length)


def test_cached_replay_rejects_missing_or_attached_boundary_states():
    envs = [HiddenGainReach(attempts=2, horizon=3)]
    policy = learner(envs[0], "gru")
    rollout = collect_trials(policy, envs, [4])
    with pytest.raises(ValueError, match="requires states"):
        replay_windows(rollout, 3)
    rollout = collect_trials(policy, envs, [4], replay_chunk_length=3)
    rollout["replay_states"].requires_grad_()
    with pytest.raises(ValueError, match="detached"):
        replay_windows(rollout, 3)
