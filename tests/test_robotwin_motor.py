"""Core batching/cache/gradient checks; run on allocated Linux CPU."""
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from torch import nn

from fluffy_bassoon.rl.robotwin_motor import (
    FrozenRoboTwinMotor, case_noise, public_batch, stack_public_observations,
)
from fluffy_bassoon.rl.pi05_residual import ReactiveALOHAActionResidual


def observation(batch=3):
    return dict(main_images=torch.arange(batch * 4 * 4 * 3, dtype=torch.uint8).reshape(batch, 4, 4, 3),
        wrist_images=torch.zeros(batch, 2, 4, 4, 3, dtype=torch.uint8),
        states=torch.arange(batch * 14).reshape(batch, 14).float(),
        task_descriptions=[f"place object {i}" for i in range(batch)],
        privileged_success=torch.ones(batch), ignored_task_id=17)


def row(raw, i):
    return {key: value[i:i + 1] for key, value in public_batch(raw).items()}


class NativeMotor(nn.Module):
    action_dim, action_env_dim, device = 32, 14, torch.device("cpu")

    def __init__(self, horizon=32):
        super().__init__()
        self.action_horizon = self.action_chunk = horizon
        self.weight = nn.Parameter(torch.ones(1))
        self.register_buffer("offset", torch.tensor(.1))
        self.prefix_calls = self.sample_calls = self.transforms = 0

    def build_prefix_cache(self, prepared):
        self.prefix_calls += 1
        hidden = prepared.state[:, :4, None].expand(-1, -1, 4).clone()
        mask = torch.ones(hidden.shape[:2], dtype=torch.bool)
        mask[:, -1] = False
        hidden[:, -1] = float("nan")  # Padding must not poison public features.
        key = hidden.masked_fill(~mask[..., None], 0)
        return hidden, mask, ((key, key + self.offset),)

    def _sample_actions_from_prefix_cache(self, prepared, mask, cache, *, noise):
        self.sample_calls += 1
        key, value = cache[0]
        return noise + (key + value).mean((1, 2))[:, None, None] * self.weight

    def decode_actions(self, targets, state):
        return targets[..., :14] + state[:, None, :14]

    def predict_action_batch(self, raw, mode="eval", *, noise):
        assert mode == "eval" and "privileged_success" not in raw
        self.transforms += 1
        prepared = SimpleNamespace(state=raw["states"] / 3.)
        _, mask, cache = self.build_prefix_cache(prepared)
        targets = self._sample_actions_from_prefix_cache(prepared, mask, cache, noise=noise)
        return self.decode_actions(targets, prepared.state), {}


@pytest.mark.parametrize("horizon", [32, 50])
def test_native_batch_matches_original_and_rows_with_one_prefix(horizon):
    native = NativeMotor(horizon)
    port = FrozenRoboTwinMotor(native, prefix_width=4)
    raw = observation()
    noise = case_noise(["trial/a/0", "trial/b/0", "trial/c/0"], seed=9,
                       horizon=horizon, action_dim=32, device="cpu")
    features, prepared = port.prepare_batch(raw)
    assert native.prefix_calls == native.transforms == 1 and native.sample_calls == 0
    assert features.shape == (3, 18) and not features.requires_grad
    torch.testing.assert_close(features[:, 4:], raw["states"] / 3.)
    actual = port.propose_batch(prepared, noise=noise)
    assert native.prefix_calls == native.transforms == native.sample_calls == 1
    expected, _ = native.predict_action_batch(public_batch(raw), noise=noise)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    individual = [port.infer_batch(row(raw, i), noise=noise[i:i + 1]) for i in range(3)]
    torch.testing.assert_close(actual, torch.cat([r[0] for r in individual]), atol=0, rtol=0)
    torch.testing.assert_close(features, torch.cat([r[1] for r in individual]), atol=0, rtol=0)
    assert not actual.requires_grad


def test_case_noise_and_observation_order_do_not_depend_on_partitions_or_global_rng():
    kwargs = dict(seed=19, horizon=32, action_dim=32, device="cpu")
    ids = ["t/e1/c0", "t/e2/c0", "t/e2/c1"]
    state = torch.get_rng_state().clone()
    noise = case_noise(ids, **kwargs)
    assert torch.equal(torch.get_rng_state(), state)
    assert torch.equal(noise, torch.cat([case_noise([i], **kwargs) for i in ids]))
    assert torch.equal(noise.flip(0), case_noise(ids[::-1], **kwargs))
    assert not torch.equal(noise[0], noise[1])
    raw = observation()
    stacked = stack_public_observations([row(raw, i) for i in range(3)])
    assert stacked.keys() == public_batch(raw).keys()
    assert "privileged_success" not in stacked
    assert torch.equal(stacked["states"], raw["states"])
    assert stacked["task_descriptions"] == raw["task_descriptions"]
    with pytest.raises(ValueError):
        case_noise([ids[0], ids[0]], **kwargs)


def test_prefix_is_current_single_use_and_failure_restores_native_method():
    native = NativeMotor()
    original = native.build_prefix_cache
    port = FrozenRoboTwinMotor(native, prefix_width=4)
    _, first = port.prepare_batch(observation(1))
    assert native.build_prefix_cache == original
    with pytest.raises(RuntimeError, match="Consume"):
        port.prepare_batch(observation(1))
    with pytest.raises(ValueError, match="noise"):
        port.propose_batch(first, noise=torch.zeros(1, 31, 32))
    _, second = port.prepare_batch(observation(1))
    with pytest.raises(ValueError, match="unconsumed"):
        port.propose_batch(first, noise=torch.zeros(1, 32, 32))
    port.discard(second)
    with pytest.raises(ValueError):
        port.discard(second)


@pytest.mark.parametrize("changed", ["parameter", "buffer", "replacement", "train", "requires_grad"])
def test_frozen_motor_rejects_weight_buffer_or_mode_changes(changed):
    native = NativeMotor()
    port = FrozenRoboTwinMotor(native, prefix_width=4)
    if changed == "parameter":
        with torch.no_grad():
            native.weight.add_(1)
    elif changed == "buffer":
        native.offset.add_(1)
    elif changed == "replacement":
        native.weight = nn.Parameter(native.weight.clone(), requires_grad=False)
    elif changed == "train":
        native.train()
    else:
        native.weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match="Frozen"):
        port.prepare_batch(observation(1))


def test_residual_gradient_survives_the_frozen_motor_boundary():
    native = NativeMotor()
    port = FrozenRoboTwinMotor(native, prefix_width=4)
    noise = torch.zeros(2, 32, 32, requires_grad=True)
    base, features = port.infer_batch(observation(2), noise=noise)
    residual = ReactiveALOHAActionResidual(public_dim=18, arm_joint_bound=.02, gripper_bound=.003)
    result = residual(base, features)
    torch.testing.assert_close(result.actions, base, atol=0, rtol=0)
    result.actions.sum().backward()
    assert residual.output.bias.grad.abs().sum() > 0
    assert noise.grad is None and all(p.grad is None for p in native.parameters())
    port.check_frozen()


def test_failed_preparation_does_not_leave_a_hook_or_stale_token():
    native = NativeMotor()
    port = FrozenRoboTwinMotor(native, prefix_width=4)
    original = native.build_prefix_cache
    raw = observation(1)
    raw["task_descriptions"] = [""]
    with pytest.raises(ValueError, match="instruction"):
        port.prepare_batch(raw)
    assert native.build_prefix_cache == original and port._prepared is None
    native.predict_action_batch = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("native fault"))
    with pytest.raises(RuntimeError, match="native fault"):
        port.prepare_batch(observation(1))
    assert native.build_prefix_cache == original and port._prepared is None


def test_numpy_process_observations_share_the_same_public_batch_path():
    raw = public_batch(observation(2))
    cpu = {key: value if key == "task_descriptions" else value.numpy() for key, value in raw.items()}
    batched = stack_public_observations([{key: value[i:i+1] for key, value in cpu.items()}
                                         for i in range(2)])
    for key in ("main_images", "wrist_images", "states"):
        assert torch.equal(batched[key], raw[key])


def test_foreign_thread_cannot_consume_or_invalidate_the_current_prefix():
    port = FrozenRoboTwinMotor(NativeMotor(), prefix_width=4)
    _, token = port.prepare_batch(observation(1))
    noise = torch.zeros(1, 32, 32)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(RuntimeError, match="owning"):
            pool.submit(port.propose_batch, token, noise=noise).result()
    assert port.propose_batch(token, noise=noise).shape == (1, 32, 14)
