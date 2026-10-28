"""ECET method/boundary/recovery checks; execute on allocated Linux resources."""

import ast
from collections import namedtuple
import hashlib
import importlib.util
import os
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from fluffy_bassoon.meta_rl.checkpoint import restore_checkpoint, save_checkpoint
from fluffy_bassoon.meta_rl.metaworld_ecet import (
    ECETConfig, ECETOptimizers, ECETPolicy, augmented_observation, collect_ecet,
    context_indices, discounted_returns, materialize_context, replay_error, update_ecet,
)
from fluffy_bassoon.meta_rl.metaworld_env import MetaWorldTrial, benchmark_pools, make_lanes


Task = namedtuple("Task", "env_name data")


class FakeSawyer:
    def set_task(self, task):
        self.task = pickle.loads(task.data)

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def observation(self):
        state = self.rng.uniform(-.2, .2, 39).astype(np.float32)
        state[-3:] = self.task["goal"]
        return state

    def reset(self):
        self.steps = 0
        return self.observation(), {}

    def step(self, action):
        assert action.shape == (4,) and np.max(np.abs(action)) <= 1
        self.steps += 1
        return self.observation(), float(self.steps), False, False, {"success": self.steps == 1}

    def close(self):
        pass


def fake_env():
    tasks = [Task("push-v3", pickle.dumps(dict(partially_observable=False, goal=[x, 2, 3])))
             for x in (1, 2, 3)]
    return MetaWorldTrial(FakeSawyer, tasks, attempts=3, horizon=6,
                          goal_visible=True, resample_instance=True)


def small_policy():
    torch.set_num_threads(1)
    return ECETPolicy(ECETConfig(width=8, heads=2, feedforward=16, intra_layers=1,
                                cross_layers=1, transitions=3, episodes=3))


def test_context_contains_actual_distinct_episodes_and_no_future_or_other_lane():
    np.random.seed(11)
    batch, horizon = 3, 6
    for step in range(24):
        indices = context_indices(step, batch, horizon=horizon, episodes=3, transitions=3)
        episode, offset = divmod(step, horizon)
        for lane in range(batch):
            valid = indices[lane][indices[lane] >= 0]
            assert (valid % batch == lane).all()
            assert (valid // batch <= step).all()
            for context_slot in range(2):
                slot = indices[lane, context_slot]
                if (slot >= 0).any():
                    assert (np.diff(slot) == batch).all()
                    expected_episode = episode - min(episode, 2) + context_slot - max(0, 2 - episode)
                    assert (slot // batch // horizon == expected_episode).all()
            current = indices[lane, -1]
            assert current[-1] == step * batch + lane
            assert (current[current >= 0] // batch // horizon == episode).all()
            assert (current < 0).sum() == max(0, 2 - offset)
        reset = context_indices(step, batch, horizon=horizon, episodes=3, transitions=3,
                                 memory_reset=True)
        assert (reset[:, :-1] == -1).all()
        np.testing.assert_array_equal(reset[:, -1], indices[:, -1])


def test_current_bypass_has_real_state_from_first_decision_and_no_clock():
    env = fake_env()
    observation, _ = env.reset(seed=12)
    augmented = augmented_observation([observation])
    assert augmented.shape == (1, 45)
    np.testing.assert_array_equal(augmented[0, :39], observation["state"])
    history = torch.tensor(augmented).reshape(1, 1, 45)
    indices = torch.tensor(context_indices(0, 1, horizon=6, episodes=3, transitions=3))
    context = materialize_context(history, indices)
    assert torch.count_nonzero(context[:, :-1]) == 0
    assert torch.count_nonzero(context[:, -1, :-1]) == 0
    torch.testing.assert_close(context[0, -1, -1], history[0, 0])
    for _ in range(6):
        observation, *_ = env.step(np.ones(4, np.float32))
    boundary = augmented_observation([observation])
    assert boundary[0, -1] == 1
    assert boundary[0, -2] == pytest.approx(.6)
    assert augmented_observation([observation], reward_blind=True)[0, -2] == 0


def test_cet_is_invariant_to_permutation_of_past_episode_summaries():
    torch.manual_seed(12)
    policy = small_policy().eval()
    episodes = torch.randn(4, 3, 8)
    left = policy.cross_context(episodes)
    right = policy.cross_context(episodes[:, [1, 0, 2]])
    torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-6)


def test_official_empty_episode_tokens_are_explicitly_not_padding_neutral():
    torch.manual_seed(91)
    policy = small_policy().eval()
    contexts = torch.zeros(2, 3, 3, 45)
    episodes = policy.encode_episodes(contexts)
    # Upstream does not mask padded histories: IET positions give even an
    # empty episode a representation, and CET attends to those empty tokens.
    assert episodes.abs().max() > 0
    episodes[:, -1] = torch.randn(2, 8)
    full = policy.cross_context(episodes)
    only_current = policy.cross_context(episodes[:, -1:])
    assert (full - only_current).abs().max() > 1e-5


def test_cross_episode_input_affects_policy_and_has_gradient():
    torch.manual_seed(13)
    policy = small_policy().eval()
    context = torch.randn(2, 3, 3, 45, requires_grad=True)
    current = context[:, -1, -1].detach()
    actor, _ = policy.distributions(context, current)
    actor.mean.square().sum().backward()
    assert context.grad[:, :-1].abs().sum() > 0
    assert policy.phi.weight.grad.abs().sum() > 0
    assert policy.intra[0].self_attn.in_proj_weight.grad.abs().sum() > 0
    assert policy.cross[0].self_attn.in_proj_weight.grad.abs().sum() > 0


def test_returns_are_monte_carlo_and_continue_across_inner_boundaries():
    rewards = torch.tensor([[1.], [2.], [3.], [4.]])
    ends = torch.tensor([[False], [False], [False], [True]])
    torch.testing.assert_close(discounted_returns(rewards, ends, .5),
                               torch.tensor([[3.25], [4.5], [5.], [4.]]))


def test_context_stream_is_independent_of_simulator_global_numpy_draws():
    policy = small_policy()
    envs = [fake_env(), fake_env()]
    torch.manual_seed(44)
    expected = collect_ecet(policy, envs, [17, 18])
    np.random.standard_normal(123)
    torch.manual_seed(44)
    actual = collect_ecet(policy, envs, [17, 18])
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_likelihood_replay_and_exact_next_update_recovery(tmp_path):
    torch.manual_seed(12)
    np.random.seed(12)
    envs = [fake_env(), fake_env()]
    policy = small_policy()
    optimizer = ECETOptimizers(policy)
    sampler = np.random.default_rng(4)
    rollout = collect_ecet(policy, envs, [3, 4])
    assert max(replay_error(policy, rollout, batch_size=7).values()) < 2e-5
    assert (rollout["observations"][6, :, -1] == 1).all()
    torch.testing.assert_close(rollout["observations"][6, :, -2], torch.full((2,), .6))
    metrics = update_ecet(policy, optimizer, rollout, epochs=1, minibatch_size=7)
    assert metrics["actor_steps"] == metrics["critic_steps"] == 6
    save_checkpoint(tmp_path / "latest.pt", policy=policy, optimizer=optimizer,
        environment_rng=sampler, iteration=1, counters={"updates": 1}, contract={"test": "ecet"})
    seeds = sampler.integers(0, 1000, 2)
    next_rollout = collect_ecet(policy, envs, seeds)
    update_ecet(policy, optimizer, next_rollout, epochs=1, minibatch_size=7)
    expected = {key: value.clone() for key, value in policy.state_dict().items()}
    restore_checkpoint(tmp_path / "latest.pt", policy=policy, optimizer=optimizer,
        environment_rng=sampler, contract={"test": "ecet"})
    np.testing.assert_array_equal(sampler.integers(0, 1000, 2), seeds)
    restored = collect_ecet(policy, envs, seeds)
    for key in restored:
        torch.testing.assert_close(restored[key], next_rollout[key], rtol=0, atol=0)
    update_ecet(policy, optimizer, restored, epochs=1, minibatch_size=7)
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)


@pytest.mark.skipif(not os.environ.get("FLUFFY_ECET_UPSTREAM"), reason="Fetch pinned official ECET source for parity")
def test_pinned_official_encoder_numerical_parity():
    """Run the actual upstream encode function and no-LN layer, without Garage deps.

    AST extraction imports only the named encode function/positional module.
    File hashes reject a changed checkout before any reference source executes.
    """
    root = Path(os.environ["FLUFFY_ECET_UPSTREAM"])
    policy_path = root / "src/garage/torch/policies/ecet_policy.py"
    layer_path = root / "src/garage/torch/modules/transformer_no_ln.py"
    assert hashlib.sha256(policy_path.read_bytes()).hexdigest() == "b4ee1533cbe3d1b055346e9f71aecebedbc76bf62e3d4a1e31dd20a6b6b44fbf"
    assert hashlib.sha256(layer_path.read_bytes()).hexdigest() == "18a5c3009b25f3b9fdd8beb08f06d6401006eb79b0b03f0501c657a3e34d883f"
    source = ast.parse(policy_path.read_text())
    position = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "PositionalEncoding")
    original = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "ECETPolicy")
    encode = next(node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == "encode")
    namespace = dict(torch=torch, nn=nn, np=np, math=__import__("math"))
    exec(compile(ast.Module(body=[position, encode], type_ignores=[]), str(policy_path), "exec"), namespace)
    spec = importlib.util.spec_from_file_location("ecet_reference_no_ln", layer_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(23)
    policy = small_policy().eval()
    config = policy.config

    class OriginalIntra(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([module.TransformerEncoderLayerNoLN(config.width, config.heads,
                config.feedforward, dropout=0.) for _ in range(config.intra_layers)])
            for original_layer, port_layer in zip(self.layers, policy.intra, strict=True):
                original_layer.load_state_dict(port_layer.state_dict())

        def forward(self, value, mask):
            for layer in self.layers:
                value = layer(value, src_mask=mask)
            return value

    cross = nn.TransformerEncoder(nn.TransformerEncoderLayer(config.width, config.heads,
        config.feedforward, dropout=0.), config.cross_layers, enable_nested_tensor=False)
    for original_layer, port_layer in zip(cross.layers, policy.cross, strict=True):
        original_layer.load_state_dict(port_layer.state_dict())
    original_model = SimpleNamespace(_obs_horizon=config.transitions, _phi=policy.phi,
        _wm_positional_encoding=namespace["PositionalEncoding"](config.width, dropout=0., max_len=config.transitions),
        _IET=OriginalIntra().eval(), _CET=cross.eval(), get_mask=lambda: policy.causal_mask)
    contexts = torch.randn(4, config.episodes, config.transitions, config.observation_dim)
    expected, _ = namespace["encode"](original_model, contexts)
    actual = policy.cross_context(policy.encode_episodes(contexts))
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(os.environ.get("FLUFFY_METAWORLD_REAL_QUALIFICATION") != "1",
                    reason="Real simulator qualification belongs in allocated Linux resources")
def test_real_ml10_ecet_trial_and_behavior_replay():
    torch.set_num_threads(1)
    torch.manual_seed(47)
    device = "cuda" if os.environ.get("FLUFFY_META_RL_CUDA_QUALIFICATION") == "1" else "cpu"
    if device == "cuda":
        assert torch.cuda.is_available(), "Requested CUDA qualification requires the allocated GPU"
    pools = benchmark_pools("ml10")
    envs = make_lanes(pools["train"], 10, benchmark="ml10", attempts=2, horizon=500)
    try:
        policy = ECETPolicy().to(device)
        rollout = collect_ecet(policy, envs, np.arange(100, 110))
        assert max(replay_error(policy, rollout).values()) < 2e-5
        assert all(len(env.outcomes) == 2 for env in envs)
        assert torch.isfinite(rollout["rewards"]).all()
    finally:
        for env in envs:
            env.close()
