"""Complete live-trial boundaries, private perturbations, PPO replay and recovery."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from fluffy_bassoon.meta_rl.pi05_bridge_train import (
    CHUNKS, collect_live_trials, qualify_replay, replay_live_trials, sample_trial_plan,
    save_live_checkpoint, restore_live_checkpoint, update_live_ppo,
)
from fluffy_bassoon.rl.pi05_bridge_policy import StochasticPi05BridgePolicy
from fluffy_bassoon.rl.pi05_feature_bridge import Pi05FeatureBridge


def test_private_scene_plan_has_disjoint_overlay_and_theta_splits():
    infos = (25, None, 21, None, 1, None, 4, 4)
    rng = np.random.default_rng(2)
    train = sample_trial_plan(infos, [0, 1], 4, 2, rng)
    evaluation = sample_trial_plan(infos, [0, 1], 4, 2, rng, evaluation=True)
    assert set(train["theta_degrees"]) <= {-45., 0., 45.}
    assert evaluation["theta_degrees"] == [-30., 30.]
    assert all(row["overlay"] < 12 for batch in train["scenes"] for row in batch)
    assert all(row["overlay"] >= 12 for batch in evaluation["scenes"] for row in batch)
    assert not {r["episode_id"] for b in train["scenes"] for r in b} & {
        r["episode_id"] for b in evaluation["scenes"] for r in b}


class CollectorPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.reference = nn.Parameter(torch.zeros(()))
        self.feedback, self.memories = [], []

    def sample_chunk(self, motor, observation, feedback, memory, *, reset, **kwargs):
        memory = torch.zeros(2) if memory is None else torch.where(reset, 0., memory)
        self.memories.append(memory.clone())
        self.feedback.append(feedback.clone())
        return torch.ones(2, 8, 7), memory + 1, dict(old_log_prob=torch.zeros(2), old_value=torch.zeros(2))


class CollectorMotor:
    def env_obs_to_observation(self, obs):
        return SimpleNamespace(state=torch.zeros(2, 32))

    def decode_actions(self, actions, state):
        return actions


class CollectorEnv:
    device = "cpu"

    def __init__(self):
        self.env = SimpleNamespace(unwrapped=SimpleNamespace(select_carrot_ids=torch.tensor([0, 1])))
        self.issued = []

    def reset(self, **kwargs):
        self.index = 0
        return {}, {}

    def chunk_step(self, actions):
        self.issued.append(actions.clone())
        infos = [dict(success=torch.tensor([self.index == 0 and step == 0, False]),
                      carrot_plate_dist=torch.tensor([999., 999.])) for step in range(5)]
        self.index += 1
        return [{}] * 5, None, None, None, infos


@pytest.mark.parametrize("lesion", ["normal", "memory_wiped", "reward_blind"])
def test_live_collector_latches_success_and_never_feeds_rotated_commands(lesion):
    policy, motor, env = CollectorPolicy(), CollectorMotor(), CollectorEnv()
    plan = dict(theta_degrees=[45., -45.], reset_seeds=[1, 2],
                scenes=[[dict(object=0, episode_id=0), dict(object=1, episode_id=1)]] * 2)
    rollout = collect_live_trials(policy, motor, env, plan, generator=torch.Generator(), lesion=lesion,
                                  preprocess=lambda obs: obs, prepare_actions=lambda actions: actions[:, :5])
    assert len(rollout["records"]) == 2 * CHUNKS
    assert rollout["simulator_steps"] == 2 * 2 * 80
    assert rollout["rewards"][:-1].sum() == 1
    torch.testing.assert_close(rollout["rewards"][CHUNKS - 1], torch.tensor([1., 0.]))
    assert not rollout["trial_ends"][:-1].any() and rollout["trial_ends"][-1].all()
    boundary = policy.feedback[CHUNKS]
    torch.testing.assert_close(boundary[:, :7], torch.ones(2, 7))
    assert not torch.equal(env.issued[-1][:, -1], boundary[:, :7])
    torch.testing.assert_close(boundary[:, 7], torch.tensor([0. if lesion == "reward_blind" else 1., 0.]))
    assert boundary[:, 8].eq(1).all()
    assert policy.feedback[CHUNKS + 1][:, 7:9].eq(0).all()
    expected = 0 if lesion == "memory_wiped" else CHUNKS
    assert policy.memories[CHUNKS].eq(expected).all()
    assert policy.feedback[0].shape == (2, 11)


class FlowMotor(nn.Module):
    action_horizon, action_dim = 2, 8

    def build_prefix_cache(self, observation):
        key = observation.state[:, None, None].expand(-1, 3, 1, -1)
        return None, torch.ones(2, 3, dtype=torch.bool), ((key, key + .1),)

    def run_suffix(self, observation, actions, time, cache, mask):
        return actions

    def velocity_from_suffix(self, suffix):
        return .2 * suffix


def flow_batch(kind):
    torch.manual_seed(16)
    bridge = Pi05FeatureBridge(core=kind, kv_dim=8, suffix_dim=8, action_dim=8,
                               width=8, ticks=1, memory_length=4, residual_scale=1.)
    policy = StochasticPi05BridgePolicy(bridge).eval()
    motor = FlowMotor().eval()
    records, memory = [], None
    for step in range(3):
        _, memory, record = policy.sample_chunk(motor, SimpleNamespace(state=torch.randn(2, 4)),
                                                torch.randn(2, 11), memory)
        records.append(record)
    ends = torch.zeros(3, 2, dtype=torch.bool)
    ends[-1] = True
    rollout = dict(records=records, old_log_probs=torch.stack([r["old_log_prob"] for r in records]),
                   old_values=torch.stack([r["old_value"] for r in records]), resets=torch.zeros_like(ends),
                   rewards=torch.tensor([[0., 0.], [1., 0.], [0., 1.]]), trial_ends=ends,
                   attempt_ends=torch.tensor([[False, False], [True, True], [True, True]]))
    return policy, rollout


@pytest.mark.parametrize("kind", ["persistent_ctm", "transformer_xl"])
def test_live_ppo_replays_in_eval_mode_and_updates_full_sequences(kind):
    policy, rollout = flow_batch(kind)
    assert qualify_replay(policy, rollout)["max_joint_log_prob_error"] == 0
    before = policy.bridge.residual_output.weight.detach().clone()
    metrics = update_live_ppo(policy, torch.optim.AdamW(policy.parameters(), lr=1e-5), rollout, epochs=2)
    assert not policy.training and not policy.bridge.training
    assert metrics["optimizer_steps"] == 2 and metrics["replayed_policy_chunks"] == 12
    assert not torch.equal(before, policy.bridge.residual_output.weight)
    assert all(np.isfinite(epoch["gradient_norm"]) for epoch in metrics["epochs"])


def test_ctm_optimizer_projection_precedes_frozen_evaluation_fingerprint():
    policy, rollout = flow_batch("persistent_ctm")
    class PushingAdam(torch.optim.AdamW):
        def step(self, *args, **kwargs):
            result = super().step(*args, **kwargs)
            with torch.no_grad():
                policy.bridge.core.decay_params_action.fill_(-1)
                policy.bridge.core.decay_params_out.fill_(16)
            return result
    update_live_ppo(policy, PushingAdam(policy.parameters(), lr=1e-5), rollout, epochs=1)
    assert policy.bridge.core.decay_params_action.eq(0).all()
    assert policy.bridge.core.decay_params_out.eq(15).all()
    before = {key: value.clone() for key, value in policy.state_dict().items()}
    replay_live_trials(policy, rollout)
    assert all(torch.equal(value, before[key]) for key, value in policy.state_dict().items())


def test_live_checkpoint_restores_explicit_motor_rng_and_sampler(tmp_path):
    policy = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=.001)
    policy(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    sampler = np.random.default_rng(7)
    motor_rng = torch.Generator().manual_seed(17)
    path, contract = tmp_path / "latest.pt", dict(protocol="live-test", source_commit="a" * 40)
    save_live_checkpoint(path, policy=policy, optimizer=optimizer, environment_rng=sampler,
                         motor_rng=motor_rng, iteration=3, counters={"steps": 12}, contract=contract)
    expected_motor, expected_sampler, expected_global = (torch.randn(5, generator=motor_rng),
                                                        sampler.integers(1000, size=5), torch.rand(5))
    restored = nn.Linear(2, 1)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=.001)
    iteration, counters = restore_live_checkpoint(path, policy=restored, optimizer=restored_optimizer,
                                                   environment_rng=sampler, motor_rng=motor_rng, contract=contract)
    assert iteration == 3 and counters == {"steps": 12}
    assert torch.equal(torch.randn(5, generator=motor_rng), expected_motor)
    np.testing.assert_array_equal(sampler.integers(1000, size=5), expected_sampler)
    assert torch.equal(torch.rand(5), expected_global)
    for key, value in policy.state_dict().items():
        assert torch.equal(restored.state_dict()[key], value)
    assert restored_optimizer.state_dict()["state"]
    torch.testing.assert_close(restored_optimizer.state_dict(), optimizer.state_dict(), rtol=0, atol=0)
