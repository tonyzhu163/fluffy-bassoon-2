"""CPU synthetic qualification candidates; execute only on allocated Linux.

No real VLA/simulator is loaded. These tests cannot establish live motor parity
or controller competence, even after a successful allocated CPU run.
"""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from fluffy_bassoon.rl.pi05_residual import (
    ReactiveALOHAActionResidual, ReactiveKVResidual, velocity_with_kv_residual,
)
from fluffy_bassoon.rl.robotwin_noise import RobotwinNoiseEnv, _RoboTwinBackend


def cache_inputs(dtype=torch.float32):
    torch.manual_seed(9)
    cache = tuple((torch.randn(2, 4, 1, 4, dtype=dtype, requires_grad=True),
                   torch.randn(2, 4, 1, 4, dtype=dtype, requires_grad=True)) for _ in range(2))
    mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
    return cache, mask, torch.randn(2, 5, requires_grad=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_kv_zero_identity_preserves_layout_dtype_padding_and_inputs(dtype):
    cache, mask, features = cache_inputs(dtype)
    copies = tuple(tuple(t.detach().clone() for t in pair) for pair in cache)
    adapter = ReactiveKVResidual(public_dim=5, kv_heads=1, head_dim=4, rank=2)
    with torch.no_grad():
        result = adapter(cache, mask, features)
    assert isinstance(result, tuple) and len(result) == len(cache)
    for before, after, saved in zip(cache, result, copies):
        for source, edited, original in zip(before, after, saved):
            assert edited.shape == source.shape and edited.dtype == source.dtype
            assert torch.equal(edited, original) and torch.equal(source, original)
            assert not edited.requires_grad


def test_nonzero_kv_edit_is_low_rank_selected_masked_and_padding_invariant():
    cache, mask, features = cache_inputs()
    adapter = ReactiveKVResidual(public_dim=5, kv_heads=1, head_dim=4, rank=2)
    with torch.no_grad():
        adapter.output_bases[0].weight.copy_(torch.arange(16).reshape(8, 2) / 30)
    original = tuple(tuple(t.detach().clone() for t in pair) for pair in cache)
    result = adapter(cache, mask, features)
    poisoned = tuple(tuple(t.detach().masked_fill(~mask[..., None, None], 10000) for t in pair) for pair in cache)
    equivalent = adapter(poisoned, mask, features)
    assert all(torch.equal(a, b) for a, b in zip(result[0], cache[0]))
    assert any(not torch.equal(a, b) for a, b in zip(result[-1], cache[-1]))
    for before, after, other in zip(cache[-1], result[-1], equivalent[-1]):
        assert torch.equal(after[~mask], before[~mask])
        torch.testing.assert_close(after[mask], other[mask])
    delta = torch.cat(tuple((a - b).flatten(2) for a, b in zip(result[-1], cache[-1])), -1)
    for lane in delta:
        assert torch.linalg.matrix_rank(lane, atol=1e-5, rtol=0) <= 2
    assert all(torch.equal(a, b) for pair, saved in zip(cache, original) for a, b in zip(pair, saved))


@pytest.mark.parametrize("fault", ["transpose", "mask", "layer_alias", "empty_lane", "nonfinite"])
def test_kv_rejects_wrong_layout_mask_or_layer_binding(fault):
    cache, mask, features = cache_inputs()
    layers = (-1, 1) if fault == "layer_alias" else (-1,)
    if fault == "transpose":
        cache = tuple(tuple(t.transpose(1, 2) for t in pair) for pair in cache)
    if fault == "mask":
        mask = mask.float()
    if fault == "empty_lane":
        mask[0] = False
    if fault == "nonfinite":
        with torch.no_grad():
            cache[-1][0][0, 0, 0, 0] = float("nan")
    adapter = ReactiveKVResidual(public_dim=5, kv_heads=1, head_dim=4, layers=layers)
    with pytest.raises(ValueError):
        adapter(cache, mask, features)


class FrozenSuffix(nn.Module):
    action_horizon, action_dim, num_steps = 3, 32, 5
    device = torch.device("cpu")

    def __init__(self):
        super().__init__()
        self.cache_projection = nn.Linear(8, 32, bias=False)
        self.velocity_projection = nn.Linear(32, 32, bias=False)
        with torch.no_grad():
            self.cache_projection.weight.fill_(.125)
            self.velocity_projection.weight.copy_(torch.eye(32) * .25)
        self.eval().requires_grad_(False)

    def run_suffix(self, observation, noisy, time, cache, mask):
        key, value = cache[-1]
        tokens = torch.cat((key.flatten(2), value.flatten(2)), -1)
        pooled = tokens.masked_fill(~mask[..., None], 0).sum(1) / mask.sum(1, keepdim=True)
        return noisy + self.cache_projection(pooled)[:, None]

    def velocity_from_suffix(self, suffix):
        return self.velocity_projection(suffix)

    def _sample_actions_from_prefix_cache(self, prepared, mask, cache, *, noise):
        value = noise.clone()
        for index in range(self.num_steps):
            time = torch.full((noise.shape[0],), 1 - index / self.num_steps)
            value = value - self.velocity_from_suffix(self.run_suffix(prepared, value, time, cache, mask)) / self.num_steps
        return value

    def decode_actions(self, targets, state):
        return targets[..., :14]


def test_nonzero_gradient_through_frozen_suffix_only_reaches_adapter():
    cache, mask, features = cache_inputs()
    motor = FrozenSuffix()
    saved = {name: p.detach().clone() for name, p in motor.named_parameters()}
    adapter = ReactiveKVResidual(public_dim=5, kv_heads=1, head_dim=4, rank=2)
    noisy = torch.zeros(2, 3, 32, requires_grad=True)
    velocity = velocity_with_kv_residual(motor, SimpleNamespace(), noisy, torch.ones(2),
                                        cache, mask, features, adapter)
    velocity.sum().backward()
    assert adapter.output_bases[0].weight.grad.abs().sum() > 0
    assert features.grad is None and noisy.grad is None
    assert all(t.grad is None for pair in cache for t in pair)
    assert all(p.grad is None and torch.equal(p, saved[name]) for name, p in motor.named_parameters())
    motor.cache_projection.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        velocity_with_kv_residual(motor, SimpleNamespace(), noisy, torch.ones(2), cache, mask, features, adapter)


@pytest.mark.parametrize("context", [torch.no_grad, torch.inference_mode])
def test_training_suffix_rejects_disabled_autograd(context):
    cache, mask, features = cache_inputs()
    adapter = ReactiveKVResidual(public_dim=5, kv_heads=1, head_dim=4)
    with context(), pytest.raises(RuntimeError, match="requires autograd"):
        velocity_with_kv_residual(FrozenSuffix(), SimpleNamespace(), torch.zeros(2, 3, 32),
                                 torch.ones(2), cache, mask, features, adapter)


def test_training_suffix_rejects_a_runtime_that_detaches_the_cache_path():
    class DetachedSuffix(FrozenSuffix):
        def run_suffix(self, *args):
            return super().run_suffix(*args).detach()

    cache, mask, features = cache_inputs()
    adapter = ReactiveKVResidual(public_dim=5, kv_heads=1, head_dim=4)
    with pytest.raises(RuntimeError, match="detached the KV adapter gradient"):
        velocity_with_kv_residual(DetachedSuffix(), SimpleNamespace(), torch.zeros(2, 3, 32),
                                 torch.ones(2), cache, mask, features, adapter)


def test_aloha_zero_identity_gradient_isolation_and_distinct_gripper_bounds():
    torch.manual_seed(7)
    base = torch.randn(2, 3, 14, requires_grad=True)
    features = torch.randn(2, 5, requires_grad=True)
    head = ReactiveALOHAActionResidual(public_dim=5, arm_joint_bound=.02, gripper_bound=.003)
    assert head.config()["width"] == 64
    result = head(base, features)
    assert torch.equal(result.actions, base) and not result.clipped.any()
    result.actions.sum().backward()
    assert head.output.bias.grad.abs().sum() > 0
    assert base.grad is None and features.grad is None
    before = base.detach().clone()
    with torch.no_grad():
        head.output.bias.fill_(10)
    result = head(base, features)
    assert (result.requested_delta[..., [6, 13]].abs() <= .003).all()
    assert (result.requested_delta[..., [0, 5, 7, 12]].abs() <= .02).all()
    assert torch.equal(base, before)
    torch.testing.assert_close(result.actions, before + result.requested_delta)


def test_decoded_command_clipping_is_explicit_and_auditable():
    head = ReactiveALOHAActionResidual(public_dim=5, arm_joint_bound=.2, gripper_bound=.05,
                                      command_lower=[-1.] * 14, command_upper=[1.] * 14)
    with torch.no_grad():
        head.output.bias.fill_(10)
    base = torch.full((1, 3, 14), .99)
    result = head(base, torch.zeros(1, 5))
    assert result.clipped.all() and (result.actions == 1).all()
    torch.testing.assert_close(result.applied_delta, result.actions - base)
    assert (result.applied_delta.abs() <= result.requested_delta.abs()).all()
    with pytest.raises(ValueError, match="Base command"):
        head(base + 1, torch.zeros(1, 5))
    with pytest.raises(ValueError, match="14"):
        head(torch.zeros(1, 3, 7), torch.zeros(1, 5))


def backend(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    motor = FrozenSuffix()
    instance = object.__new__(_RoboTwinBackend)
    instance.torch, instance.model = torch, motor
    instance._prepared = SimpleNamespace(state=torch.zeros(1, 14))
    instance._residual_prepared = instance._prepared
    instance._prefix_mask = torch.tensor([[True, True, False]])
    instance._kv_cache = ((torch.ones(1, 3, 1, 4), torch.ones(1, 3, 1, 4)),)
    instance._public_features = torch.zeros(1, 5)
    instance._versions = {n: p._version for n, p in motor.named_parameters()}
    instance.motor_seconds = instance.prefix_seconds = 0.
    instance.motor_queries, instance.native_horizon = 0, 3
    instance.kv_adapter = instance.action_residual = None
    return instance


@pytest.mark.parametrize("kind", ["none", "kv", "action"])
def test_backend_preserves_native_zero_parity_and_records_exact_hook(monkeypatch, kind):
    instance = backend(monkeypatch)
    noise = np.arange(96, dtype=np.float32).reshape(3, 32) / 100
    native = instance.decode(noise)
    original = tuple(tuple(t.clone() for t in pair) for pair in instance._kv_cache)
    if kind == "kv":
        instance.kv_adapter = ReactiveKVResidual(public_dim=5, kv_heads=1, head_dim=4).eval()
    if kind == "action":
        instance.action_residual = ReactiveALOHAActionResidual(
            public_dim=5, arm_joint_bound=.02, gripper_bound=.003).eval()
    assert np.array_equal(instance.decode(noise), native)
    assert instance.motor_queries == 2
    assert all(torch.equal(a, b) for pair, saved in zip(instance._kv_cache, original) for a, b in zip(pair, saved))
    record = instance._last_residual_metadata
    if kind == "none":
        assert record is None
    else:
        assert record["configuration"]["kind"].startswith("reactive_")
        if kind == "action":
            assert torch.equal(record["base_actions"], record["actions"])
            assert not record["clipped"].any()


def test_backend_forwards_applied_commands_metadata_and_actual_duration(monkeypatch):
    instance = backend(monkeypatch)
    head = ReactiveALOHAActionResidual(public_dim=5, arm_joint_bound=.02, gripper_bound=.003).eval()
    with torch.no_grad():
        head.output.bias.fill_(1)
    instance.action_residual = head
    instance._decision_metadata = {"owner": "synthetic-test"}
    captured = {}
    raw = dict(main_images=np.zeros((1, 2, 2, 3), np.uint8),
               wrist_images=np.zeros((1, 2, 2, 2, 3), np.uint8), states=np.zeros((1, 14)),
               task_descriptions=["source fixture"])

    def step(commands):
        captured["commands"] = commands.clone()
        return [raw], torch.ones(1), torch.ones(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool), [
            dict(valid_action_mask=torch.tensor([True, True, False]), episode=dict(success_once=True))]

    instance.env = SimpleNamespace(chunk_step=step, venv=SimpleNamespace(envs=[SimpleNamespace(task=object())]))
    instance._encode = lambda observation: np.zeros(5, np.float32)
    instance.corpus_writer = SimpleNamespace(
        begin_decision=lambda noise, actions, metadata: captured.update(issued=actions.copy(), metadata=metadata),
        complete_native=lambda **values: captured.update(native=values))
    monkeypatch.setattr("fluffy_bassoon.rl.robotwin_noise._capture_sparse_feedback", lambda task: nullcontext({}))
    _, _, terminated, _, info = instance.step(np.zeros((3, 32), np.float32), 3)
    record = captured["metadata"]["frozen_pi05_residual"]
    np.testing.assert_array_equal(captured["commands"].numpy()[0], record["actions"].numpy()[0])
    np.testing.assert_array_equal(captured["issued"], record["actions"].numpy()[0])
    torch.testing.assert_close(record["actions"], record["base_actions"] + record["applied_delta"])
    assert captured["metadata"]["owner"] == "synthetic-test"
    assert terminated and info["primitive_steps"] == 2 and captured["native"]["executed_steps"] == 2


def test_ambiguous_hooks_and_stale_public_features_fail_closed(monkeypatch):
    instance = backend(monkeypatch)
    instance.kv_adapter = ReactiveKVResidual(public_dim=5, kv_heads=1, head_dim=4).eval()
    instance._residual_prepared = object()
    with pytest.raises(RuntimeError, match="bound"):
        instance.decode(np.zeros((3, 32), np.float32))
    with pytest.raises(ValueError, match="only one"):
        RobotwinNoiseEnv(kv_adapter=object(), action_residual=object())
    with pytest.raises(ValueError, match="injected backend"):
        RobotwinNoiseEnv(backend=object(), kv_adapter=object())
