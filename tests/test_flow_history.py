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


def test_native_reference_uses_saved_draw_and_restores_hooks_on_failure():
    from fluffy_bassoon.meta_rl.flow_history import FlowReferenceMotor
    class Reference(torch.nn.Module):
        device = "cpu"
        def build_prefix_cache(self):
            return "original"
        def predict_action_batch(self, observation, *, mode, noise):
            assert self.build_prefix_cache() == "captured" and mode == "eval"
            np.testing.assert_array_equal(noise.numpy(), observation["noise"])
            return torch.ones((len(noise), 50, 12)) * .75, {}
    native = SimpleNamespace(model=object(), _prepared=None)
    reference = Reference()
    spec = codec()
    motor = FlowReferenceMotor(native, spec, reference)
    noise = teacher_noise("ab"*32, 3)[None]
    prepared = dict(batch_size=1, prefix="captured", observation=dict(noise=noise))
    native._prepared = prepared
    actual = motor.execute_batch(prepared, noise=noise, steering=np.zeros((1, 600)), spec=spec)
    assert actual.shape == (1, 50, 12) and native._prepared is None
    assert reference.build_prefix_cache() == "original"
    prepared["observation"]["noise"] = np.zeros_like(noise)
    native._prepared = prepared
    with pytest.raises(AssertionError):
        motor.execute_batch(prepared, noise=noise, steering=np.zeros((1, 600)), spec=spec)
    assert reference.build_prefix_cache() == "original" and native._prepared is None


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


def test_export_corpus_supports_h600_multi_version_ppo_and_grpo_streams(tmp_path):
    from fluffy_bassoon.meta_rl.flow_history import FEATURE_DIM, export_corpus
    from fluffy_bassoon.meta_rl.vla_history import load_dataset
    rng = np.random.default_rng(7)
    streams, versions, chunks = 8, 3, 12  # H=600 (12 H50 chunks per episode)
    total = versions * chunks
    states = rng.normal(size=(streams, total, 25)).astype(np.float32)
    features = np.concatenate((states, rng.normal(size=(streams, total, FEATURE_DIM - 25)).astype(np.float32)), axis=-1)
    actions = rng.normal(size=(streams, total, 50, 12)).astype(np.float32)
    actions[..., [6, 11]] = 1.0
    teacher = actions.copy()
    teacher[..., 0] += 0.5
    rewards = np.zeros((streams, total, 50), dtype=np.float32)
    rewards[:, chunks - 1, -1] = 1.0
    raw = dict(
        public_states=states,
        physical_actions=actions,
        episode_starts=np.broadcast_to(np.arange(total) % chunks == 0, (streams, total)),
        physical_episode_ends=np.broadcast_to(np.arange(total) % chunks == chunks - 1, (streams, total)),
        producer_versions=np.broadcast_to(np.repeat(np.arange(versions), chunks), (streams, total)),
        primitive_rewards=rewards,
    )
    manifests = export_corpus(
        tmp_path / "grpo_corpus", raw, features, teacher, codec(),
        provenance=dict(source_algorithm="grpo", group_size=8),
        validation_slots=(6, 7),
    )
    assert set(manifests) == {
        "ad_train", "ad_validation", "dpt_train", "dpt_validation",
        "dit_train", "dit_validation", "ic_iql_train", "ic_iql_validation",
    }
    ad_trials, ad_manifest = load_dataset(manifests["ad_train"]["path"])
    dpt_trials, _ = load_dataset(manifests["dpt_train"]["path"])
    assert len(ad_trials) == 6 and ad_manifest["metadata"]["schedule"] == dict(
        streams=8, policy_versions=3, chunks_per_episode=12)
    assert np.array_equal(ad_trials[0]["executed_noise"], dpt_trials[0]["executed_noise"])
    assert not np.array_equal(ad_trials[0]["target_noise"], dpt_trials[0]["target_noise"])


@pytest.mark.parametrize("interface", ["kv_bridge", "velocity_bridge"])
def test_history_flow_steering_policy_supports_interfaces_and_four_meta_learners(interface):
    from fluffy_bassoon.meta_rl.flow_history import FlowBridgeConfig, HistoryFlowSteeringPolicy
    from fluffy_bassoon.meta_rl.history_value import ValueConfig

    class FrozenMotor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.kv_proj = torch.nn.Linear(8, 32, bias=False)
            self.action_proj = torch.nn.Linear(32, 32, bias=False)
            self.eval().requires_grad_(False)

        def run_suffix(self, observation, noisy_actions, timestep, cache, prefix_mask):
            key, value = cache[-1]
            pooled = (key + value).flatten(2).mean(dim=1)[:, None, :]
            return self.action_proj(noisy_actions) + self.kv_proj(pooled)

        def velocity_from_suffix(self, suffix):
            return suffix

    motor = FrozenMotor()
    cfg = FlowBridgeConfig(
        interface=interface, observation_dim=16, history_action_dim=12,
        width=8, layers=1, heads=2, context_length=8,
        kv_heads=1, head_dim=4, kv_rank=2, kv_layers=(-1,), scale=0.1,
        action_horizon=4, model_action_dim=32, active_action_dim=12,
    )
    policy = HistoryFlowSteeringPolicy(cfg)
    tokens = torch.randn(3, 2, 16 + 12 + 3)
    valid = torch.ones(3, 2, dtype=torch.bool)
    starts = torch.tensor([[True, True], [False, False], [False, False]])
    contexts = policy.encode_sequence(tokens, valid, starts)
    context = contexts[-1]
    cache = ((torch.randn(2, 5, 1, 4), torch.randn(2, 5, 1, 4)),)
    mask = torch.ones(2, 5, dtype=torch.bool)
    target = torch.randn(2, 4, 32)
    noise = torch.randn(2, 4, 32)
    tau = torch.tensor([0.25, 0.75])

    # Zero-initialized output preserves exact native velocity before any step.
    with torch.no_grad():
        base_v = motor.velocity_from_suffix(motor.run_suffix({}, noise, tau, cache, mask))
        init_v = policy.predict_velocity(motor, {}, noise, tau, cache, mask, context)
        assert torch.allclose(base_v, init_v)

    for objective in ("ad", "dpt", "dit", "ic_iql"):
        weights = HistoryFlowSteeringPolicy.meta_objective_weights(
            objective,
            q_values=torch.tensor([0.8, 0.1]),
            v_values=torch.tensor([0.3, 0.2]),
            value_config=ValueConfig(objective="ic_iql" if objective not in ("dit", "ic_iql") else objective),
            batch_size=2,
        )
        policy.zero_grad(set_to_none=True)
        loss = policy.flow_matching_loss(
            motor, {}, target, noise, tau, cache, mask, context, sample_weights=weights)
        loss.backward()
        assert torch.isfinite(loss)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.parameters())
        assert all(p.grad is None for p in motor.parameters())


def test_export_grpo_group_trials_orders_failures_before_successes_within_scene(tmp_path):
    from fluffy_bassoon.meta_rl.flow_history import FEATURE_DIM, export_grpo_group_trials
    from fluffy_bassoon.meta_rl.vla_history import load_dataset
    rng = np.random.default_rng(19)
    group_size, num_groups, versions, chunks = 8, 4, 2, 12
    streams = group_size * num_groups
    total = versions * chunks
    states = rng.normal(size=(streams, total, 25)).astype(np.float32)
    for v in range(versions):
        c0 = v * chunks
        for g in range(num_groups):
            s0, s1 = g * group_size, (g + 1) * group_size
            states[s0:s1, c0] = states[s0, c0]
    features = np.concatenate((states, rng.normal(size=(streams, total, FEATURE_DIM - 25)).astype(np.float32)), axis=-1)
    actions = rng.normal(size=(streams, total, 50, 12)).astype(np.float32)
    actions[..., [6, 11]] = 1.0
    teacher = actions.copy()
    teacher[..., 0] += 0.25
    rewards = np.zeros((streams, total, 50), dtype=np.float32)
    for v in range(versions):
        c_end = (v + 1) * chunks - 1
        for g in range(num_groups):
            s0 = g * group_size
            rewards[s0 + 1, c_end, -1] = 1.0
            rewards[s0 + 5, c_end, -1] = 1.0
    raw = dict(
        public_states=states,
        physical_actions=actions,
        episode_starts=np.broadcast_to(np.arange(total) % chunks == 0, (streams, total)),
        physical_episode_ends=np.broadcast_to(np.arange(total) % chunks == chunks - 1, (streams, total)),
        producer_versions=np.broadcast_to(np.repeat(np.arange(versions), chunks), (streams, total)),
        primitive_rewards=rewards,
    )
    manifests = export_grpo_group_trials(
        tmp_path / "grpo_groups", raw, features, teacher, codec(),
        provenance=dict(source_algorithm="grpo", group_size=8),
        group_size=8, attempts_per_trial=5, validation_groups=(3,),
    )
    ad_trials, ad_manifest = load_dataset(manifests["ad_train"]["path"])
    assert len(ad_trials) == 3 * versions
    first_trial = ad_trials[0]
    assert first_trial["observations"].shape == (5 * chunks, FEATURE_DIM)
    ep_returns = first_trial["rewards"].reshape(5, chunks).sum(-1)
    np.testing.assert_array_equal(ep_returns, [0., 0., 0., 1., 1.])
    assert ad_manifest["metadata"]["schedule"]["attempts_per_trial"] == 5


@pytest.mark.parametrize("interface", ["kv_bridge", "velocity_bridge"])
def test_flow_bridge_motor_zero_init_parity_and_closed_loop_controller(interface):
    from fluffy_bassoon.meta_rl.flow_history import (
        ACTION_DIM, FEATURE_DIM, FlowBridgeCodec, FlowBridgeConfig,
        FlowBridgeMotor, HistoryFlowSteeringPolicy)
    from fluffy_bassoon.meta_rl.vla_history import FrozenHistoryController

    class ToyNativeModel(torch.nn.Module):
        device = "cpu"
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

        def build_prefix_cache(self, observation):
            return (
                torch.ones(1, 4, 8),
                torch.ones(1, 4, dtype=torch.bool),
                ((torch.ones(1, 4, 1, 4) * 0.5, torch.ones(1, 4, 1, 4) * -0.25),),
            )

        def velocity_from_suffix(self, suffix):
            return suffix

        def predict_action_batch(self, observation, *, mode, noise):
            assert mode == "eval"
            _, _, cache = self.build_prefix_cache(observation)
            k, v = cache[-1]
            kv_term = (k.mean() + v.mean()) * torch.ones((len(noise), 50, 32))
            vel = self.velocity_from_suffix(noise * 0.5 + kv_term)
            return vel[..., :12], {}

    model = ToyNativeModel().eval()
    prefix = model.build_prefix_cache({})

    class NativePort:
        def __init__(self):
            self.model = model
            self._prepared = None

        def prepare_batch(self, obs):
            self._prepared = dict(batch_size=len(obs["states"]), prefix=prefix, observation=obs)
            return np.ones((len(obs["states"]), 16 + 2048), np.float32), self._prepared

    native = NativePort()
    spec = FlowBridgeCodec(codec().mean, codec().std, interface=interface)
    cfg = FlowBridgeConfig(
        interface=interface, observation_dim=FEATURE_DIM, history_action_dim=ACTION_DIM,
        width=16, layers=1, heads=2, context_length=16,
        kv_heads=1, head_dim=4, kv_rank=2, kv_layers=(-1,), scale=0.1,
    )
    policy = HistoryFlowSteeringPolicy(cfg).eval()
    motor = FlowBridgeMotor(native, spec, policy)
    controller = FrozenHistoryController(policy, mode="full")

    obs = dict(states=np.ones((1, 25), np.float32))
    features, prepared = motor.prepare(obs)
    control = controller.act(features, deterministic=True)
    assert control.shape == (ACTION_DIM,)
    draw = teacher_noise("cd" * 32, 0)[None]
    commands = motor.execute(prepared, noise=draw, steering=control, spec=spec)
    base_commands, _ = model.predict_action_batch(obs, mode="eval", noise=torch.as_tensor(draw))
    np.testing.assert_allclose(commands, base_commands[0].numpy(), atol=1e-6)
    feedback = spec.feedback_control(commands)
    controller.observe(feedback, 1.0, True)
    assert np.array_equal(controller.previous_noise, feedback)
