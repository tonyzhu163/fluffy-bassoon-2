"""RL² data flow, fitted baseline, PPO semantics and complete recovery.

Run inside an allocated Linux step, alongside the common Meta-World tests.
"""

from collections import namedtuple
from copy import deepcopy
import json
import pickle
import sys

import numpy as np
import pytest
import torch
from torch.distributions import Normal

from fluffy_bassoon.meta_rl.checkpoint import restore_checkpoint, save_checkpoint
from fluffy_bassoon.meta_rl.metaworld_env import MetaWorldTrial
from fluffy_bassoon.meta_rl.metaworld_rl2 import (
    GARAGE_COMMIT, LinearFeatureBaseline, RL2Policy, _advantages,
    make_rl2_policy, match_rl2_policy, method_contract, update_rl2_ppo,
)
from fluffy_bassoon.meta_rl.metaworld_train import replay_error
from fluffy_bassoon.meta_rl.ppo import RecurrentActorCritic, collect_trials, generalized_advantages


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Model qualification requires allocated Linux")
Task = namedtuple("Task", "env_name data")


class FakeSawyer:
    def set_task(self, task):
        self.task = pickle.loads(task.data)

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def observation(self):
        obs = np.zeros(39, dtype=np.float32)
        obs[0] = self.rng.uniform()
        obs[-3:] = self.task["goal"]
        return obs

    def reset(self):
        self.steps = 0
        return self.observation(), {}

    def step(self, action):
        self.steps += 1
        return self.observation(), float(self.steps), False, False, {"success": self.steps == 2}

    def close(self):
        pass


def fake_env():
    tasks = [Task("push-v3", pickle.dumps(dict(partially_observable=False, goal=[x, 2, 3])))
             for x in (1, 2, 3)]
    return MetaWorldTrial(FakeSawyer, tasks, attempts=2, horizon=3,
                          goal_visible=True, resample_instance=True)


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    previous_matmul = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn = torch.backends.cudnn.allow_tf32
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yield
    torch.set_num_threads(previous)
    torch.backends.cuda.matmul.allow_tf32 = previous_matmul
    torch.backends.cudnn.allow_tf32 = previous_cudnn


def test_reference_width_is_distinct_from_optional_parameter_matching():
    settings = dict(width=64, baseline_reg_coeff=1e-5, parameter_matching=False)
    config, size = match_rl2_policy(settings, fake_env())
    assert config["width"] == 64 and size["parameters"] == 21960
    assert not size["parameter_cap_enforced"]
    config, size = match_rl2_policy(dict(settings, parameter_matching=True), fake_env())
    assert config["width"] > 64 and 212500 <= size["parameters"] <= 250000
    assert size["parameter_cap_enforced"]
    manifest = method_contract(config)
    assert manifest["reference"]["commit"] == GARAGE_COMMIT
    assert "TRPO" in manifest["original_algorithm"]
    assert manifest["normalization"].startswith("none")


def test_gaussian_saves_preclip_likelihood_and_actor_has_no_state_bypass():
    policy = RL2Policy(fake_env(), width=8)
    features = torch.randn(2, policy.core.output_dim)
    other = features.clone()
    other[:, :47] += 100
    torch.testing.assert_close(policy.actor(features), policy.actor(other), rtol=0, atol=0)
    draw = torch.tensor([[2., -3., .5, 0.], [-2., 3., -.5, 0.]])
    action, raw, logprob, entropy, _ = policy._action_and_value(features, draw)
    torch.testing.assert_close(action, draw.clamp(-1, 1), rtol=0, atol=0)
    torch.testing.assert_close(raw, draw, rtol=0, atol=0)
    distribution = Normal(policy.actor(features), policy.log_std.exp())
    torch.testing.assert_close(logprob, distribution.log_prob(draw).sum(-1))
    torch.testing.assert_close(entropy, distribution.entropy().sum(-1))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fused_gru_replay_matches_serial_outputs_and_parameter_gradients(device, record_property):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA qualification requires an allocated GPU")
    torch.manual_seed(5)
    policy = RL2Policy(fake_env(), width=8).to(device).eval()
    observations = torch.randn(9, 2, 47, device=device)
    actions = torch.randn(9, 2, 4, device=device)
    resets = torch.zeros(9, 2, dtype=torch.bool, device=device)
    initial = torch.randn(2, 8, device=device)
    fast = policy.evaluate_sequence(observations, actions, resets, initial_state=initial)
    serial = RecurrentActorCritic.evaluate_sequence(policy, observations, actions, resets,
                                                   initial_state=initial)
    for actual, expected in zip(fast, serial, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
    parameters = tuple(policy.parameters())
    fast_grad = torch.autograd.grad(fast[0].square().mean() + fast[1].mean(), parameters)
    serial_grad = torch.autograd.grad(serial[0].square().mean() + serial[1].mean(), parameters)
    # Use the exact represented FP32 weights/inputs in a CPU float64 oracle.
    # A direct elementwise FP32-gradient comparison confounds correctness with
    # CuDNN's different reduction order, especially for cancellation near zero.
    reference = deepcopy(policy).cpu().double()
    reference_observations, reference_actions = observations.cpu().double(), actions.cpu().double()
    reference_resets, reference_initial = resets.cpu(), initial.cpu().double()
    reference_serial = RecurrentActorCritic.evaluate_sequence(reference, reference_observations,
        reference_actions, reference_resets, initial_state=reference_initial)
    reference_fast = reference.evaluate_sequence(reference_observations, reference_actions,
        reference_resets, initial_state=reference_initial)
    reference_parameters = tuple(reference.parameters())
    reference_grad = torch.autograd.grad(reference_serial[0].square().mean() + reference_serial[1].mean(),
                                         reference_parameters)
    reference_fast_grad = torch.autograd.grad(reference_fast[0].square().mean() + reference_fast[1].mean(),
                                              reference_parameters)
    for actual, expected in zip(reference_fast, reference_serial, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
    for actual, expected in zip(reference_fast_grad, reference_grad, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)
    for outputs in (fast, serial):
        for actual, expected in zip(outputs, reference_serial, strict=True):
            torch.testing.assert_close(actual.cpu().double(), expected, rtol=1e-5, atol=2e-6)

    # Normwise qualification measures the backward error at each parameter
    # tensor's scale, avoiding unstable relative errors on near-zero entries.
    # This is a declared numerical acceptance budget, not a formal stability
    # theorem: four unit-roundoff budgets per input/recurrent/time dimension.
    terms = observations.shape[-1] + policy.core.width + observations.shape[0]
    error_budget = 4 * terms * torch.finfo(torch.float32).eps
    diagnostics = []
    for (name, _), oracle, fused_gradient, serial_gradient in zip(
            policy.named_parameters(), reference_grad, fast_grad, serial_grad, strict=True):
        scale = max(float(oracle.norm()), 1.)
        item = dict(parameter=name, reference_l2=float(oracle.norm()), normalized_l2_budget=error_budget)
        for label, gradient in (("fused", fused_gradient), ("serial", serial_gradient)):
            error = gradient.cpu().double() - oracle
            item[label] = dict(max_absolute_error=float(error.abs().max()),
                               normalized_l2_error=float(error.norm()) / scale)
        diagnostics.append(item)
    record_property("fp32_backward_error_against_cpu_float64", json.dumps(diagnostics))
    print(json.dumps(dict(device=device, fp32_backward_error_against_cpu_float64=diagnostics)), flush=True)
    assert all(item[label]["normalized_l2_error"] <= error_budget
               for item in diagnostics for label in ("fused", "serial")), diagnostics


def test_trial_memory_feedback_and_frozen_behavior_replay():
    torch.manual_seed(7)
    envs = [fake_env(), fake_env()]
    policy = RL2Policy(envs[0], width=8)
    rollout = collect_trials(policy, envs, [3, 4], record_states=True)
    assert not rollout["resets"].any()
    assert rollout["observations"][3, :, 0].eq(1).all()
    torch.testing.assert_close(rollout["observations"][3, :, 7], torch.full((2,), .3))
    torch.testing.assert_close(rollout["observations"][3, :, 3:7], rollout["raw_actions"][2].clamp(-1, 1))
    assert rollout["attempt_ends"].sum() == 4
    assert rollout["trial_ends"].sum() == 2
    assert max(replay_error(policy, rollout).values()) < 2e-5
    inputs = rollout["observations"]
    state = None
    for t in range(3):
        _, state = policy.core(inputs[t], state)
    torch.testing.assert_close(state, rollout["attempt_states"]["state"][0])
    carried, _ = policy.core(inputs[3], state)
    cleared, _ = policy.core(inputs[3], None)
    assert not torch.allclose(carried[:, 47:], cleared[:, 47:])
    _, _, _, _, _, reset = policy.step(inputs[3], state, reset=torch.ones(2, dtype=torch.bool))
    torch.testing.assert_close(reset, cleared[:, 47:])


def test_ridge_baseline_uses_global_trial_clock_across_episode_reset():
    torch.manual_seed(4)
    env = fake_env()
    policy = RL2Policy(env, width=8)
    rollout = collect_trials(policy, [env], [42])
    features = policy.critic.features(rollout["observations"])
    torch.testing.assert_close(features[:, 0, -4], torch.arange(6).double() / 100)
    # A window predicts identically without knowing its sequence start.
    target = torch.arange(6).float()[:, None] * .1 + 3
    policy.critic.fit(rollout["observations"], target)
    whole = policy.critic(rollout["observations"])
    window = policy.critic(rollout["observations"][3:])
    torch.testing.assert_close(whole[3:], window, rtol=0, atol=0)
    torch.testing.assert_close(whole.squeeze(-1), target, rtol=0, atol=2e-3)
    assert policy.critic.fit_count == 1
    assert policy.critic.last_reg_coeff >= policy.critic.reg_coeff


def test_ridge_solution_agrees_with_independent_numpy_reference():
    rng = np.random.default_rng(3)
    obs = torch.tensor(rng.normal(size=(12, 2, 47)), dtype=torch.float32)
    obs[..., 1:3] = torch.rand(12, 2, 2)
    target = torch.tensor(rng.normal(size=(12, 2)), dtype=torch.float32)
    baseline = LinearFeatureBaseline(47, 2, 3)
    x = baseline.features(obs).reshape(24, -1).numpy()
    expected = np.linalg.solve(x.T @ x + 1e-5 * np.eye(x.shape[1]), x.T @ target.numpy().reshape(-1))
    baseline.fit(obs, target)
    np.testing.assert_allclose(baseline.coefficients.numpy(), expected, rtol=2e-3, atol=2e-4)


def test_gae_crosses_physical_episode_boundaries_and_stops_entropy_gradients():
    policy = RL2Policy(fake_env(), width=8)
    rewards = torch.zeros(6, 1)
    rewards[3] = 1
    ends = torch.zeros(6, 1, dtype=torch.bool)
    ends[-1] = True
    advantage = _advantages(policy, rewards, torch.zeros_like(rewards), ends,
                             gamma=1., gae_lambda=1., entropy_coef=0.)
    assert advantage[2] == 1  # credit crosses the physical reset at step 3.
    assert advantage[4] == 0
    entropy_advantage = _advantages(policy, torch.zeros_like(rewards), torch.zeros_like(rewards),
                                    ends, gamma=1., gae_lambda=1., entropy_coef=.02)
    assert not entropy_advantage.requires_grad
    assert entropy_advantage[0] > entropy_advantage[-1] > 0
    values = torch.randn_like(rewards)
    direct = _advantages(policy, rewards, values, ends, gamma=.99, gae_lambda=.95, entropy_coef=.02)
    base = generalized_advantages(rewards, values, ends, gamma=.99, gae_lambda=.95)[0]
    weights = generalized_advantages(torch.ones_like(rewards), torch.zeros_like(rewards), ends,
                                     gamma=.99, gae_lambda=.95)[0]
    entropy = Normal(torch.zeros_like(policy.log_std), policy.log_std.exp()).entropy().sum().detach()
    torch.testing.assert_close(base + .02 * entropy * weights, direct, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_complete_rl2_checkpoint_recovers_baseline_and_exact_next_update(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA recovery qualification requires an allocated GPU")
    torch.manual_seed(12)
    envs = [fake_env(), fake_env()]
    config = dict(width=8, baseline_reg_coeff=1e-5, parameter_matching=False)
    policy = make_rl2_policy(config, envs[0]).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    sampler = np.random.default_rng(4)
    rollout = collect_trials(policy, envs, [3, 4])
    metrics = update_rl2_ppo(policy, optimizer, rollout, epochs=2, minibatch_trials=1)
    assert metrics["optimizer_steps"] == 4 and metrics["forward_samples"] == 24
    assert metrics["baseline_fits"] == 1
    save_checkpoint(tmp_path / "latest.pt", policy=policy, optimizer=optimizer,
                    environment_rng=sampler, iteration=1, counters={"updates": 1}, contract={"test": config})
    seeds = sampler.integers(0, 1000, 2)
    following = collect_trials(policy, envs, seeds)
    update_rl2_ppo(policy, optimizer, following, epochs=2, minibatch_trials=1)
    expected = {key: value.clone() for key, value in policy.state_dict().items()}
    restore_checkpoint(tmp_path / "latest.pt", policy=policy, optimizer=optimizer,
                       environment_rng=sampler, contract={"test": config})
    assert policy.critic.fit_count == 1
    np.testing.assert_array_equal(sampler.integers(0, 1000, 2), seeds)
    restored = collect_trials(policy, envs, seeds)
    for key in restored:
        torch.testing.assert_close(restored[key], following[key], rtol=0, atol=0)
    update_rl2_ppo(policy, optimizer, restored, epochs=2, minibatch_trials=1)
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)


def test_named_source_adapter_rejects_silent_algorithm_changes():
    env = fake_env()
    policy = RL2Policy(env, width=8)
    optimizer = torch.optim.Adam(policy.parameters())
    rollout = collect_trials(policy, [env], [3])
    for kwargs in (dict(replay_chunk_length=3), dict(normalize_advantages=True),
                   dict(value_coef=.5), dict(max_grad_norm=.5)):
        with pytest.raises(ValueError, match="full-trial actor-only"):
            update_rl2_ppo(policy, optimizer, rollout, **kwargs)
