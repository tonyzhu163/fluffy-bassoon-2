"""Numerical physical-action maps and causal actual-feedback recovery."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.flow_history import DirectActionCodec, FlowDirectMotor, teacher_noise
from fluffy_bassoon.meta_rl.steering_history_eval import collect_case_attempts
from fluffy_bassoon.meta_rl.vla_history import HistoryConfig, HistoryNoisePolicy


def codec():
    return DirectActionCodec.from_stats(dict(actions=dict(mean=np.arange(12) * .1, std=np.arange(12) + .5)))


def test_physical_action_roundtrip_and_binary_threshold_reference():
    spec = codec()
    actions = np.random.default_rng(21).normal(size=(3, 50, 12)).astype(np.float32)
    actions[..., [6, 11]] = np.where(actions[..., [6, 11]] > 0, 1., -1.)
    encoded = spec.encode(actions)
    commands = spec.decode(encoded)
    np.testing.assert_allclose(spec.mapped_physical(commands), actions, atol=2e-6)
    np.testing.assert_array_equal(commands[..., [6, 11]], (actions[..., [6, 11]] > 0).astype(np.float32))
    np.testing.assert_allclose(spec.feedback_control(commands), encoded, atol=2e-6)
    with pytest.raises(ValueError, match="binary"):
        spec.encode(np.zeros((50, 12), np.float32))
    assert np.array_equal(teacher_noise("ab" * 32, 3), teacher_noise("ab" * 32, 3))
    assert not np.array_equal(teacher_noise("ab" * 32, 3), teacher_noise("ab" * 32, 4))


def test_direct_motor_uses_all_public_state_and_consumes_prefix_without_denoising():
    class Native:
        model = object()
        def prepare_batch(self, obs):
            self._prepared = dict(batch_size=len(obs["states"]))
            return np.ones((len(obs["states"]), 2064), np.float32), self._prepared
    native = Native()
    motor, spec = FlowDirectMotor(native, codec()), codec()
    states = np.arange(50, dtype=np.float32).reshape(2, 25)
    features, prepared = motor.prepare_batch(dict(states=states))
    np.testing.assert_array_equal(features[:, :25], states)
    assert features.shape == (2, 2073)
    commands = motor.execute_batch(prepared, noise=None, steering=np.zeros((2, 600), np.float32), spec=spec)
    assert commands.shape == (2, 50, 12) and native._prepared is None
    with pytest.raises(ValueError, match="unconsumed"):
        motor.execute_batch(prepared, noise=None, steering=np.zeros((2, 600), np.float32), spec=spec)


@pytest.mark.parametrize("mode", ["full", "episode"])
def test_mapped_feedback_and_reset_memory_resume_match_uninterrupted(mode):
    from test_steering_history_eval import collect
    torch.manual_seed(124)
    policy = HistoryNoisePolicy(HistoryConfig(3, 600, width=8, heads=2, layers=1, context_length=12))
    spec, case = codec(), dict(task="fake", seed=23)
    args, *_ = collect(policy, spec, case)
    args["controller_mode"] = mode
    original = collect_case_attempts(policy, spec, case, **args)
    expected = [2, 4, 6, 8, 10] if mode == "full" else [2] * 5
    assert [row["memory_tokens"] for row in original] == expected
    for attempt in original:
        for row in attempt["records"]:
            np.testing.assert_array_equal(row["executed_control"], spec.feedback_control(row["requested_commands"]))
    args, environments, *_ = collect(policy, spec, case, completed_prefix=original[:2])
    args["controller_mode"] = mode
    recovered = collect_case_attempts(policy, spec, case, **args)
    assert [env.attempt for env in environments] == [3, 4, 5]
    for first, second in zip(original, recovered):
        for a, b in zip(first["records"], second["records"]):
            np.testing.assert_array_equal(a["control"], b["control"])
    changed = deepcopy(original[:1])
    changed[0]["records"][0]["executed_control"][0] += .25
    args, *_ = collect(policy, spec, case, completed_prefix=changed)
    args["controller_mode"] = mode
    with pytest.raises(ValueError, match="physical-action feedback"):
        collect_case_attempts(policy, spec, case, **args)
