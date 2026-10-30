"""Allocated Linux tests: real controller, isolated fake simulator lanes."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.vla_noise_batched_collection import (
    CollectionNotStarted, collect_trials_batched,
)
from fluffy_bassoon.meta_rl.vla_noise_ppo import PPOConfig, RecurrentNoisePPO


class Lane:
    observation_dim = 2
    noise_shape = (1, 2)
    native_horizon = 2

    def reset(self, task, seed):
        self.seed, self.primitive = seed, 0
        self.length = seed % 5 + 1
        return self.features()

    def features(self):
        return np.array([self.seed % 7, self.primitive], np.float32)

    def step(self, noise):
        count = min(self.native_horizon, self.length-self.primitive)
        assert count > 0
        self.primitive += count
        done = self.primitive == self.length
        return self.features(), float(done), done, False, dict(primitive_steps=count, success=done)


class Vector:
    observation_dim = 2
    noise_shape = (1, 2)
    native_horizon = 2

    def __init__(self):
        self.lanes = []
        self.dispatches = []
        self.reset_calls = []

    def reset_batch(self, tasks, seeds):
        self.lanes = [Lane() for _ in tasks]
        return self.reset_lanes(list(range(len(tasks))), tasks, seeds)

    def reset_lanes(self, indices, tasks, seeds):
        self.reset_calls.extend(zip(indices, tasks, seeds))
        return np.stack([self.lanes[lane].reset(task, seed)
                         for lane, task, seed in zip(indices, tasks, seeds)])

    def step_batch(self, noises, *, active_mask):
        assert noises.shape == (len(self.lanes), *self.noise_shape)
        self.dispatches.append(active_mask.copy())
        features, rewards, terms, truncs, counts, successes = [], [], [], [], [], []
        for lane, noise, active in zip(self.lanes, noises, active_mask):
            if active:
                feature, reward, term, trunc, info = lane.step(noise)
            else:
                feature, reward, term, trunc, info = lane.features(), 0., False, False, dict(
                    primitive_steps=0, success=False)
            features.append(feature); rewards.append(reward); terms.append(term); truncs.append(trunc)
            counts.append(info["primitive_steps"]); successes.append(info["success"])
        return (np.stack(features), np.array(rewards), np.array(terms), np.array(truncs),
                dict(primitive_steps=np.array(counts), success=np.array(successes)))


def model(reset=False):
    return RecurrentNoisePPO(2, (1, 2), device="cpu", seed=17, config=PPOConfig(
        width=8, layers=1, heads=2, memory_length=8, epochs=1,
        minibatch_trials=2, train_episodes=2, episode_limit=5, reset_each_episode=reset))


@pytest.mark.parametrize("trained_reset,memory_reset", [(False, None), (True, None), (False, True)])
@pytest.mark.parametrize("controller_microbatch", [0, 1])
def test_each_lane_matches_serial_causality_and_replays(trained_reset, memory_reset, controller_microbatch):
    learner = model(trained_reset)
    tasks, seeds = ["scan_object", "handover_mic"], [[0, 1], [4, 2]]
    before = deepcopy(learner.policy.state_dict())
    env = Vector()
    collected = collect_trials_batched(learner, env, tasks, seeds,
        deterministic=True, reset_memory=memory_reset, controller_microbatch=controller_microbatch)
    for lane, (rollout, result) in enumerate(collected):
        expected, expected_result = learner.collect_trial(Lane(), tasks[lane], seeds[lane],
            deterministic=True, reset_memory=memory_reset)
        assert result == expected_result
        for key in expected:
            torch.testing.assert_close(rollout[key], expected[key],
                rtol=0 if controller_microbatch else 2e-5,
                atol=0 if controller_microbatch else 2e-6)
        errors = learner.replay_error(rollout)
        assert errors["log_prob_max_error"] < 2e-5
        assert errors["value_max_error"] < 2e-6
        assert rollout["trial_ends"].sum() == 1 and rollout["trial_ends"][-1, 0]
        assert rollout["action_mask"].all()
    assert any(mask.tolist() == [False, True] for mask in env.dispatches)
    assert learner.updates == learner.optimizer_steps == learner.primitive_steps == learner.chunks == 0
    for key, value in before.items():
        torch.testing.assert_close(value, learner.policy.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("trained_reset", [False, True])
def test_scalar_controller_kernels_replay_exactly_without_finished_lane_rng_draws(trained_reset):
    learner = model(trained_reset)
    original_step = learner.policy.step
    batches = []
    def audited_step(observation, *args, **kwargs):
        assert observation.storage_offset() == 0
        batches.append(observation.shape[0])
        return original_step(observation, *args, **kwargs)
    learner.policy.step = audited_step
    before_rng = torch.get_rng_state().clone()
    precision = torch.get_float32_matmul_precision()
    tf32 = torch.backends.cuda.matmul.allow_tf32
    collected = collect_trials_batched(learner, Vector(), ["scan_object"]*2,
        [[0], [4, 4]], controller_microbatch=1)
    learner.policy.step = original_step
    after_rng = torch.get_rng_state().clone()
    valid_decisions = sum(result["chunks"] for _, result in collected)
    assert valid_decisions == 7 and batches == [1] * valid_decisions
    # Normal.sample draws once per valid B1 policy call, with no other random
    # operations in this fake environment/controller. No inactive draws occur.
    torch.set_rng_state(before_rng)
    for _ in range(valid_decisions):
        torch.normal(torch.zeros((1, learner.action_dim)), torch.ones((1, learner.action_dim)))
    torch.testing.assert_close(torch.get_rng_state(), after_rng, rtol=0, atol=0)
    torch.set_rng_state(after_rng)
    for rollout, _ in collected:
        assert learner.replay_error(rollout) == dict(log_prob_max_error=0., value_max_error=0.)
    assert torch.get_float32_matmul_precision() == precision
    assert torch.backends.cuda.matmul.allow_tf32 == tf32


@pytest.mark.parametrize("value", [-1, 2, True])
def test_unknown_controller_microbatch_fails_before_reset(value):
    env = Vector()
    with pytest.raises(ValueError, match="controller_microbatch"):
        collect_trials_batched(model(), env, ["scan_object"], [[0]], controller_microbatch=value)
    assert not env.reset_calls and not env.dispatches


def test_ragged_trial_lengths_do_not_add_padding_to_global_ppo_weighting():
    learner = model()
    collected = collect_trials_batched(learner, Vector(), ["scan_object"]*2,
                                      [[0], [4, 4]], deterministic=False)
    rollouts = [rollout for rollout, _ in collected]
    assert [len(row["rewards"]) for row in rollouts] == [1, 6]
    assert sum(int(row["action_mask"].sum()) for row in rollouts) == 7
    metrics = learner.update(rollouts)
    assert metrics["chunks"] == 7 and metrics["optimizer_steps"] == 1
    assert learner.chunks == 7 and learner.updates == 1
    assert np.isfinite(list(metrics.values())).all()


def test_budget_and_prelaunch_cancel_do_not_dispatch_any_simulator_work():
    env = Vector()
    with pytest.raises(ValueError, match="reserve every complete"):
        collect_trials_batched(model(), env, ["scan_object"], [[1, 2]], primitive_budget=9)
    assert not env.reset_calls and not env.dispatches
    with pytest.raises(CollectionNotStarted):
        collect_trials_batched(model(), env, ["scan_object"], [[1, 2]], should_stop=lambda: True)
    assert not env.reset_calls and not env.dispatches


def test_inflight_stop_drains_complete_trials_and_callbacks_keep_lane_identity():
    env = Vector()
    callbacks = []
    collected = collect_trials_batched(model(), env, ["scan_object", "handover_mic"],
        [[0], [4, 4]], should_stop=lambda: bool(env.dispatches),
        episode_callback=lambda lane, row: callbacks.append((lane, row["episode"])))
    assert callbacks == [(0, 1), (1, 1), (1, 2)]
    for rollout, result in collected:
        assert result["stop_requested"] and rollout["trial_ends"][-1, 0]
        assert len(result["episodes"]) == result["requested_episodes"]


def test_external_cut_and_inactive_commands_fail_closed():
    class Cut(Vector):
        def step_batch(self, *args, **kwargs):
            result = super().step_batch(*args, **kwargs)
            result[-1]["budget_truncated"] = np.ones(len(self.lanes), bool)
            return result
    with pytest.raises(RuntimeError, match="external budget cut"):
        collect_trials_batched(model(), Cut(), ["scan_object"], [[4]])
    class BadInactive(Vector):
        def step_batch(self, *args, **kwargs):
            result = super().step_batch(*args, **kwargs)
            result[-1]["primitive_steps"][~kwargs["active_mask"]] = 1
            return result
    with pytest.raises(ValueError, match="zero inactive"):
        collect_trials_batched(model(), BadInactive(), ["scan_object"]*2, [[0], [4]])


def test_native_per_task_horizons_and_decision_audit_match_scalar_rollouts():
    limits={"scan_object":3,"handover_mic":5}
    class NativeLane(Lane):
        def reset(self, task, seed):
            value=super().reset(task,seed)
            self.episode_limit=limits[task]
            self.length=min(self.length,self.episode_limit)
            return value
    class NativeVector(Vector):
        def episode_limit_for_task(self,task): return limits[task]
        def reset_batch(self,tasks,seeds):
            self.lanes=[NativeLane() for _ in tasks]
            return self.reset_lanes(list(range(len(tasks))),tasks,seeds)
    learner=model()
    emitted=[[],[]]
    tasks=["scan_object","handover_mic"]
    seeds=[[4]*4,[4]*4]
    collected=collect_trials_batched(learner,NativeVector(),tasks,seeds,
        primitive_budget=32,controller_microbatch=1,
        decision_callback=lambda lane,row: emitted[lane].append(row))
    assert [result["primitive_steps"] for _,result in collected]==[12,20]
    for lane,(rollout,result) in enumerate(collected):
        assert rollout["episode_limits"].eq(limits[tasks[lane]]).all()
        assert len(emitted[lane])==result["chunks"]
        for index,metadata in enumerate(emitted[lane]):
            assert metadata["history_index"]==index
            assert metadata["episode_limit"]==limits[tasks[lane]]
            for field,saved in (("policy_input","observations"),("pre_squash_actions","raw_actions"),
                                ("old_log_prob","log_probs"),("old_value","values")):
                np.testing.assert_array_equal(metadata[field],rollout[saved][index,0].numpy())
            assert metadata["controller_rng"]["torch_before"].dtype==torch.uint8
            assert metadata["controller_rng"]["cuda_before"]==[]
        errors=learner.replay_error(rollout)
        assert errors==dict(log_prob_max_error=0.,value_max_error=0.)


def test_audited_callback_requires_b1_before_any_reset():
    env=Vector()
    with pytest.raises(ValueError,match="scalar controller kernels"):
        collect_trials_batched(model(),env,["scan_object"],[[0]],
                              decision_callback=lambda lane,row:None)
    assert not env.reset_calls
