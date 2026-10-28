"""Offline bounds and real sampler-path qualification for private theta teachers."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fluffy_bassoon.meta_rl.pi05_interface_gate import (
    PrivilegedThetaSelector, fit_privileged_teacher, paired_sample_diagnostics,
    residual_headroom, slice_history_lanes,
)
from fluffy_bassoon.rl.pi05_bridge_policy import StochasticPi05BridgePolicy
from fluffy_bassoon.rl.pi05_feature_bridge import Pi05FeatureBridge


def history():
    records = []
    for chunk in range(3):
        native = torch.zeros(2, 8, 10)
        target = native.clone()
        target[:, :5, :7] = torch.tensor([-2., -1., -.5, 0., .5, 1., 2.])
        target[:, 5:] = 1000  # No executed-prefix diagnostic may count this tail.
        target[:, :5, 7:] = 1000  # Neither may it count padded model channels.
        records.append(dict(native_velocity=native, target_velocity=target,
                            feedback=torch.tensor([[chunk, 10.], [chunk, 20.]])))
    rewards = torch.tensor([[0., 0.], [1., 0.], [0., 1.]])
    ends = torch.zeros(3, 2, dtype=torch.bool)
    ends[-1] = True
    return dict(records=records, rewards=rewards, resets=torch.zeros_like(ends),
                attempt_ends=ends.clone(), trial_ends=ends, supervised_action_steps=5)


def test_saturation_is_weighted_over_only_executed_active_coordinates():
    result = residual_headroom(history(), residual_scale=1.)
    assert result["scalar_count"] == 3 * 2 * 5 * 7
    assert result["target_residual_abs_max"] == 2
    assert result["target_residual_abs_mean"] == 1
    assert result["saturation_fraction"] == pytest.approx(2 / 7)
    assert result["zero_residual_flow_matching_loss"] == pytest.approx(10.5 / 7)
    assert result["bounded_flow_matching_loss_floor"] == pytest.approx(2 / 7)
    assert result["per_channel"][0]["scalar_count"] == 30
    assert result["per_channel"][1]["saturation_fraction"] == 0  # Equality is feasible.
    repeated = residual_headroom([history(), slice_history_lanes(history(), [1])], residual_scale=1.)
    assert repeated["scalar_count"] == 315
    assert repeated["bounded_flow_matching_loss_floor"] == result["bounded_flow_matching_loss_floor"]
    with pytest.raises(ValueError, match="positive residual"):
        residual_headroom(history(), residual_scale=0)


def test_teacher_lane_slice_retains_chronology_and_counts_only_selected_steps():
    original = history()
    selected = slice_history_lanes(original, [1])
    assert selected["supervised_action_steps"] == 5
    assert selected["policy_chunks"] == 3 and selected["simulator_steps"] == 15
    assert len(selected["records"]) == len(original["records"])
    for name in ("rewards", "resets", "attempt_ends", "trial_ends"):
        assert torch.equal(selected[name], original[name][:, 1:2])
    for index, record in enumerate(selected["records"]):
        assert record["feedback"].tolist() == [[index, 20.]]
        record["feedback"].zero_()
        assert original["records"][index]["feedback"][1, 1] == 20
    for invalid in ([1, 1], [], [2], [True], [1.]):
        with pytest.raises(ValueError):
            slice_history_lanes(original, invalid)


class RecordingTeacher(nn.Module):
    active_dims = 7

    def __init__(self, value):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(float(value)))
        self.calls = []

    def config(self):
        return dict(noise_scale=.05, active_dims=7, residual_scale=1.)

    def sample_chunk(self, motor, observation, feedback, memory, **kwargs):
        self.calls.append((motor, observation, feedback, memory, kwargs))
        return self.weight, memory, dict(selected=True)


def test_private_selector_never_passes_theta_to_public_teacher_inputs():
    negative, positive = RecordingTeacher(-1), RecordingTeacher(1)
    selector = PrivilegedThetaSelector({-45: negative, 45: positive})
    with pytest.raises(ValueError, match="Select private"):
        selector.sample_chunk(None, None, torch.zeros(2, 11), deterministic=True)
    with pytest.raises(ValueError, match="homogeneous"):
        selector.select_theta([-45, 45])
    with pytest.raises(ValueError, match="No separately"):
        selector.select_theta([0, 0])
    selector.select_theta([45, 45])
    observation, memory = object(), {"state": torch.ones(2)}
    feedback = torch.randn(2, 11)
    result, next_memory, _ = selector.sample_chunk(None, observation, feedback, memory, deterministic=True)
    assert result is positive.weight and next_memory is memory and not negative.calls
    assert positive.calls[0] == (None, observation, feedback, memory, dict(deterministic=True))
    assert selector.active_dims == 7 and not selector.training and not positive.training
    with pytest.raises(ValueError, match="deterministic"):
        selector.sample_chunk(None, observation, feedback, memory)
    with pytest.raises(ValueError, match="lane counts"):
        selector.sample_chunk(None, observation, feedback[:1], memory, deterministic=True)
    with pytest.raises(ValueError, match="independent fitted"):
        PrivilegedThetaSelector({-45: positive, 45: positive})


class GateMotor(nn.Module):
    action_horizon, action_dim = 8, 10

    def build_prefix_cache(self, observation):
        key = observation.state[:, None, None].expand(-1, 3, 1, -1)
        return None, torch.ones(observation.state.shape[0], 3, dtype=torch.bool), ((key, key + .1),)

    def run_suffix(self, observation, actions, time, cache, mask):
        return actions

    def velocity_from_suffix(self, suffix):
        return .2 * suffix

    def sample_actions(self, observation, *, num_steps, noise):
        assert num_steps == 4
        actions = noise
        for _ in range(4):
            actions = actions - .25 * self.velocity_from_suffix(actions)
        return actions

    def decode_actions(self, actions, state):
        return actions[:, :5, :7]


def gate_policy():
    return StochasticPi05BridgePolicy(Pi05FeatureBridge(core="transformer_xl", kv_dim=8,
        suffix_dim=10, action_dim=10, width=8, ticks=1, memory_length=4, residual_scale=1.)).eval()


def test_paired_gate_uses_actual_sampler_not_external_labels_and_same_native_noise():
    policy, motor = gate_policy(), GateMotor().eval()
    observation, feedback = SimpleNamespace(state=torch.randn(2, 4)), torch.zeros(2, 11)
    noise = torch.ones(2, 8, 10)
    actions, memory, record, metrics = paired_sample_diagnostics(policy, motor, observation,
        observation, feedback, [45., -45.], noise=noise)
    assert torch.equal(actions, motor.sample_actions(observation, num_steps=4, noise=noise))
    assert memory and record["deterministic"] and record["gaussian_scalars_per_lane"] == 224
    assert metrics["learned"] == metrics["zero"]
    assert metrics["learned"]["decoded_xy_mse"] > 0
    assert metrics["external_oracle"]["decoded_action_mse"] == 0
    assert metrics["executed_action_steps"] == 5 and metrics["num_steps"] == 4
    assert torch.equal(noise, torch.ones_like(noise))
    with pytest.raises(ValueError, match="explicit native noise"):
        paired_sample_diagnostics(policy, motor, observation, observation, feedback, [45., -45.])


def test_bounded_teacher_fit_uses_updates_and_reports_validation(monkeypatch):
    import fluffy_bassoon.meta_rl.pi05_interface_gate as gate
    calls = []
    monkeypatch.setattr(gate, "distillation_loss", lambda policy, data: torch.tensor(data["loss"]))
    monkeypatch.setattr(gate, "update_distillation", lambda policy, optimizer, data: dict(loss=data))
    validation = [dict(records=[None], rewards=torch.zeros(1, 1), loss=2.),
                  dict(records=[None] * 3, rewards=torch.zeros(3, 1), loss=4.)]
    result = fit_privileged_teacher(None, None, [1., 2.], updates=3,
        validation_histories=validation,
        on_update=lambda update, metrics: calls.append((update, metrics)))
    assert result["completed"] and result["updates_completed"] == 3
    assert result["initial_validation_loss"] == result["final_validation_loss"] == 3.5
    assert [update for update, _ in calls] == [1, 2, 3]
    with pytest.raises(ValueError, match="positive bounded"):
        fit_privileged_teacher(None, None, [], updates=1)


def test_real_teacher_fit_updates_only_the_declared_bridge_path():
    from fluffy_bassoon.meta_rl.bridge_distillation import make_dpt_record
    policy, motor = gate_policy(), GateMotor().eval()
    observation = SimpleNamespace(state=torch.randn(2, 4))
    record = make_dpt_record(motor, observation, torch.zeros(2, 11), torch.ones(2, 8, 10),
                             generator=torch.Generator().manual_seed(11))
    data = dict(records=[record], rewards=torch.zeros(1, 2), supervised_action_steps=5)
    before = policy.bridge.residual_output.weight.detach().clone()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    result = fit_privileged_teacher(policy, optimizer, [data], updates=2, validation_histories=[data])
    assert result["completed"] and result["updates_completed"] == 2
    assert not torch.equal(before, policy.bridge.residual_output.weight)
    assert all(torch.isfinite(torch.tensor(row["gradient_norm"])) for row in result["metrics"])
    assert record["feedback"].eq(0).all() and "theta_degrees" not in record
    assert not any(parameter.requires_grad for parameter in motor.parameters())
