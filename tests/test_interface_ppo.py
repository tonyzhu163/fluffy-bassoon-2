"""Core numerical, causality and recovery checks; run on allocated Linux."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.flora_success import FloraSuccessRule, first_success, timing_metrics
from fluffy_bassoon.meta_rl.interface_ppo import InterfaceLearner, InterfaceSpec, collect
from fluffy_bassoon.meta_rl.paired_evaluation import EvaluationPlan, evaluate, paired_readout
from fluffy_bassoon.meta_rl.residual_history import HistoryResidualActor
from fluffy_bassoon.meta_rl.vla_history import HistoryConfig, HistoryNoisePolicy


def test_success_hold_is_causal_and_confirmed_without_backdating():
    rule = FloraSuccessRule(consecutive=2)
    score = lambda step: dict(step=step, observed_through=step, reward=.99, p5=.99)
    assert first_success([score(50), score(100)], rule) == 100
    assert first_success([score(50), score(150)], rule) is None
    with pytest.raises(ValueError, match="causal"):
        first_success([dict(score(50), observed_through=100)], rule)
    metrics = timing_metrics([50, 150, 100, None], [200, 100, None, 50])
    assert metrics["prematurely_detected"] == metrics["false_on_never_success"] == 1
    assert metrics["first_success_precision"] == pytest.approx(1 / 3)


def test_joint_policy_scores_every_noise_coordinate_and_preserves_native_initial_std():
    spec = InterfaceSpec()
    learner = InterfaceLearner(3, spec, width=8, layers=1)
    control, log_prob, _ = learner.policy.act(torch.zeros(1, 3), generator=learner.generator)
    native, steering = spec.decode(control[0].numpy())
    assert native.shape == (1, 50, 32) and steering.shape == (6,)
    torch.testing.assert_close(learner.policy.log_std[:1600], torch.zeros(1600))
    distribution, _ = learner.policy._distribution(torch.zeros(1, 3))
    torch.testing.assert_close(log_prob, distribution.log_prob(control).sum(-1))
    changed = control.clone(); changed[0, 1000] += 2
    assert learner.policy.evaluate(torch.zeros(1, 3), changed)[0] != log_prob


class Motor:
    def prepare(self, observation):
        return np.array([observation], np.float32), observation

    def execute(self, prepared, *, noise, steering, spec):
        assert noise.shape == (1, 50, 32) and not len(steering)
        return noise[0, :, :12]


class Environment:
    def __init__(self):
        self.ended = self.counter = 0

    def reset(self):
        self.counter = 0
        return 0.

    def step_chunk(self, chunk):
        self.counter += 50
        return 1., self.counter == 100, dict(executed_controls=50, time=self.counter)

    def end_episode(self):
        self.ended += 1

    def state_dict(self):
        return dict(counter=self.counter, ended=self.ended)

    def load_state_dict(self, state):
        self.counter, self.ended = state["counter"], state["ended"]


def test_first_success_masks_later_chunks_and_checkpoint_restores_next_adam_step():
    spec = InterfaceSpec(kind="noise", steering_dim=0)
    learner = InterfaceLearner(1, spec, width=8, layers=1, epochs=1, minibatch_size=8)
    env = Environment()
    reward = lambda payload: dict(first_success_step=payload["time"], reward=1.)
    rows = collect(learner.policy, Motor(), env, spec, reward, episodes=3, generator=learner.generator)
    assert len(rows) == env.ended == 3
    assert all(r["done"] and r["episode_start"] and r["first_success_step"] == 50 for r in rows)
    sampler = Environment()
    learner.train_batch(rows)
    saved = learner.state_dict(sampler=sampler, environment=env)
    learner.train_batch(rows)
    expected = deepcopy(learner.policy.state_dict())
    resumed = InterfaceLearner(1, spec, seed=99, width=8, layers=1, epochs=1, minibatch_size=8)
    resumed.load_state_dict(saved, sampler=sampler, environment=env)
    resumed.train_batch(rows)
    assert resumed.updates == 2 and resumed.controls == 300
    for key, value in expected.items():
        torch.testing.assert_close(resumed.policy.state_dict()[key], value, rtol=0, atol=0)
    for key, state in learner.optimizer.state_dict()["state"].items():
        for field, value in state.items():
            torch.testing.assert_close(resumed.optimizer.state_dict()["state"][key][field], value, rtol=0, atol=0)


def test_reward_blind_removes_only_reward_and_memory_reset_clears_attempts():
    torch.manual_seed(2)
    policy = HistoryNoisePolicy(HistoryConfig(3, 2, width=8, layers=1, heads=2, context_length=128))
    actor = HistoryResidualActor(policy, reset_each_episode=False, reward_blind=True)
    actor.begin_trial(); actor.begin_attempt()
    control = actor.act(torch.ones(1, 3), deterministic=True)[0]
    actor.observe(control[0].numpy(), 1., True)
    actor.install_terminal_reward(1.)
    actor.begin_attempt()
    captured = []
    original = policy.step
    def step(token, *args, **kwargs):
        captured.append(token.clone())
        return original(token, *args, **kwargs)
    policy.step = step
    actor.act(torch.ones(1, 3), deterministic=True)
    assert captured[0][0, 5] == 0 and captured[0][0, 6] == 1
    actor = HistoryResidualActor(policy, mode="episode", reset_each_episode=False)
    actor.begin_trial(); actor.begin_attempt()
    first = actor.act(torch.ones(1, 3), deterministic=True)[0]
    actor.observe(first[0].numpy(), 1., True); actor.begin_attempt()
    second = actor.act(torch.ones(1, 3), deterministic=True)[0]
    torch.testing.assert_close(first, second)


def test_paired_harness_shares_cases_streams_and_clusters_draws():
    plan = EvaluationPlan(("OpenDrawer",), cases=2, draws=2, attempts=2, horizon=100)
    assert plan.manifest() == plan.manifest()
    seen = []
    def factory(mode):
        return SimpleNamespace(mode=mode, begin_trial=lambda: None, begin_attempt=lambda: None)
    def rollout(actor, case, seeds):
        seen.append((actor.mode, case["native_seed"], seeds))
        success = actor.mode == "full"
        return dict(completed=True, controls=100, pairing_sha256=str(case["native_seed"]),
                    simulator_success_once=success, simulator_final_success=success,
                    first_success_step=50 if success else None)
    rows = evaluate(plan, factory, rollout)
    assert len(rows) == 24
    for index in range(0, len(seen), 6):
        assert seen[index][1:] == seen[index + 2][1:] == seen[index + 4][1:]
    report = paired_readout(rows, "full", "memory_reset", attempt=2, bootstraps=100)
    assert report["cases"] == report["draws_per_case"] == 2
    assert report["difference"] == 1
    assert report["exact_case_sign_randomization_p"] == .5
    with pytest.raises(ValueError, match="every paired"):
        paired_readout(rows[:-2], "full", "reward_blind", attempt=2)
