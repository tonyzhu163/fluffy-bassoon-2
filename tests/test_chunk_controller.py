"""Frozen physical controller parity, lane isolation and command binding."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.chunk_controller import FrozenChunkController
from fluffy_bassoon.meta_rl.chunk_history import ChunkHistoryLayout, trial_tokens
from fluffy_bassoon.meta_rl.chunk_policy import ChunkModelConfig, ChunkResidualPolicy


def controller(batch=2, mode='full'):
    torch.manual_seed(701)
    config = ChunkModelConfig(ChunkHistoryLayout(3, 2, 2), width=8, layers=1,
                              heads=2, context_length=4)
    policy = ChunkResidualPolicy(config, torch.full((2, 2), .3))
    return FrozenChunkController(policy, native_batch=batch, mode=mode)


def feedback(control, commands, *, active=None, terminal=None, reward=None):
    batch = control.native_batch
    active = np.ones(batch, np.bool_) if active is None else np.asarray(active, np.bool_)
    terminal = np.zeros(batch, np.bool_) if terminal is None else np.asarray(terminal, np.bool_)
    masks = np.repeat(active[:, None], 2, axis=1)
    # Retain full issued tails even when physics only delivers the first target.
    masks[:, -1] = False
    control.observe(commands.numpy(), masks, np.zeros(batch) if reward is None else reward,
                    terminated=terminal, truncated=np.zeros(batch, np.bool_),
                    physics_seconds=active.astype(np.float32)*.1)


@pytest.mark.parametrize('mode', ['full', 'episode', 'static'])
def test_online_matches_offline_physical_history_and_policy(mode):
    control = controller(mode=mode)
    reference = deepcopy(control.policy)
    rng = np.random.default_rng(29)
    features = rng.normal(size=(4, 2, 3)).astype(np.float32)
    bases = rng.normal(size=(4, 2, 2, 2)).astype(np.float32)
    starts = np.array([[1, 1], [0, 0], [1, 0], [0, 1]], np.bool_)
    term = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], np.bool_)
    rewards = np.array([[0, 0], [1, 0], [0, 1], [0, 0]], np.float32)
    issued = []
    for t in range(4):
        commands = control.propose(torch.from_numpy(features[t]), torch.from_numpy(bases[t]),
                                   episode_starts=starts[t])
        issued.append(commands.numpy().copy())
        feedback(control, commands, terminal=term[t], reward=rewards[t])
    issued = np.stack(issued)
    masks = np.zeros((4, 2), np.bool_);masks[:, 0] = True
    tokens, resets = [], []
    for lane in range(2):
        lane_tokens, lane_resets = trial_tokens(control.layout, features=features[:, lane],
            bases=bases[:, lane], issued=issued[:, lane], delivered_masks=masks,
            rewards=rewards[:, lane], terminated=term[:, lane], truncated=np.zeros(4, np.bool_),
            physics_seconds=np.full(4, .1, np.float32), episode_starts=starts[:, lane], mode=mode)
        tokens.append(lane_tokens);resets.append(lane_resets)
    tokens, resets = np.stack(tokens, axis=1), np.stack(resets, axis=1)
    state = None
    with torch.no_grad():
        for t in range(4):
            residual, _, state = reference.step(torch.from_numpy(tokens[t]), state,
                reset=torch.from_numpy(resets[t]), deterministic=True)
            torch.testing.assert_close(torch.from_numpy(issued[t]),
                                       torch.from_numpy(bases[t])+residual, rtol=0, atol=0)
    for key in state:
        torch.testing.assert_close(control.state[key], state[key], rtol=0, atol=0)


def test_terminal_tail_calls_preserve_inactive_lanes_and_ignore_raw_reward():
    control = controller()
    features, bases = torch.zeros(2, 3), torch.zeros(2, 2, 2)
    commands = control.propose(features, bases, episode_starts=np.ones(2, np.bool_))
    feedback(control, commands, terminal=[True, False], reward=[1., 0.])
    saved = deepcopy(control.state)
    saved_feedback = control.encoders[0].previous.copy()
    for _ in range(3):
        commands = control.propose(features+5, bases+7, episode_starts=np.zeros(2, np.bool_),
                                   active=np.array([False, True]))
        torch.testing.assert_close(commands[0], bases[0]+7, rtol=0, atol=0)
        feedback(control, commands, active=[False, True], reward=[1., 0.])
        for key in saved:
            torch.testing.assert_close(control.state[key][0], saved[key][0], rtol=0, atol=0)
        np.testing.assert_array_equal(control.encoders[0].previous, saved_feedback)
    assert control.state['length'].tolist() == [1, 4]
    saved = deepcopy(control.state)
    commands = control.propose(features, bases, episode_starts=np.zeros(2, np.bool_),
                               active=np.zeros(2, np.bool_))
    feedback(control, commands, active=[False, False], reward=[1., 1.])
    for key in saved:
        torch.testing.assert_close(control.state[key], saved[key], rtol=0, atol=0)


def test_trial_reset_clears_selected_history_before_next_real_query():
    control = controller()
    features, bases = torch.randn(2, 3), torch.randn(2, 2, 2)
    commands = control.propose(features, bases, episode_starts=np.ones(2, np.bool_))
    with pytest.raises(RuntimeError, match='Observe pending'):
        control.reset_trials([0])
    feedback(control, commands)
    control.reset_trials([0])
    fresh = FrozenChunkController(deepcopy(control.policy), native_batch=2)
    expected = fresh.propose(features, bases, episode_starts=np.ones(2, np.bool_))
    actual = control.propose(features, bases, episode_starts=np.array([True, False]))
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    assert control.state['length'].tolist() == [1, 2]
    assert fresh.state['length'].tolist() == [1, 1]


def test_issued_command_binding_is_atomic_and_failure_forbids_replay():
    control = controller()
    commands = control.propose(torch.zeros(2, 3), torch.zeros(2, 2, 2),
                               episode_starts=np.ones(2, np.bool_))
    altered = commands.clone();altered[1, -1, -1] += .01
    with pytest.raises(ValueError, match='exact full issued'):
        feedback(control, altered)
    assert control.failed and all(encoder.first for encoder in control.encoders)
    with pytest.raises(RuntimeError, match='without automatic replay'):
        feedback(control, commands)


def test_fixed_native_batch_and_explicit_physical_activity_are_required():
    control = controller()
    with pytest.raises(ValueError, match='fixed native batch'):
        control.propose(torch.zeros(1, 3), torch.zeros(1, 2, 2), episode_starts=np.ones(1, np.bool_))
    with pytest.raises(ValueError, match='boolean'):
        control.propose(torch.zeros(2, 3), torch.zeros(2, 2, 2), episode_starts=[1, 1])
    with pytest.raises(ValueError, match='inactive tail'):
        control.propose(torch.zeros(2, 3), torch.zeros(2, 2, 2),
                        episode_starts=np.ones(2, np.bool_), active=np.array([True, False]))
    assert control.state is None and control.pending is None and not control.failed


@pytest.mark.parametrize('change', ['parameter', 'normalization', 'training'])
def test_deployment_model_is_frozen_including_statistics_and_mode(change):
    control = controller()
    assert not any(p.requires_grad for p in control.policy.parameters())
    if change == 'training':
        control.policy.train()
    else:
        value = next(control.policy.parameters()) if change == 'parameter' else control.policy.feature_mean
        with torch.no_grad():
            value.add_(1)
    with pytest.raises(RuntimeError, match='Frozen decoded-controller'):
        control.propose(torch.zeros(2, 3), torch.zeros(2, 2, 2), episode_starts=np.ones(2, np.bool_))


@pytest.mark.parametrize('fail', [False, True])
def test_student_precision_is_explicit_and_motor_setting_is_restored(monkeypatch, fail):
    control = controller()
    original = control.policy.step

    def checked_step(*args, **kwargs):
        assert torch.get_float32_matmul_precision() == 'highest'
        if fail:
            raise RuntimeError('Injected policy failure')
        return original(*args, **kwargs)

    monkeypatch.setattr(control.policy, 'step', checked_step)
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision('high')
        if fail:
            with pytest.raises(RuntimeError, match='Injected policy failure'):
                control.propose(torch.zeros(2, 3), torch.zeros(2, 2, 2),
                    episode_starts=np.ones(2, np.bool_))
        else:
            control.propose(torch.zeros(2, 3), torch.zeros(2, 2, 2),
                episode_starts=np.ones(2, np.bool_))
        assert torch.get_float32_matmul_precision() == 'high'
    finally:
        torch.set_float32_matmul_precision(previous)
