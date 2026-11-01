"""Numerical equivalence of serial and scheduled parallel residual rollouts."""
from copy import deepcopy

import numpy as np
import torch

from fluffy_bassoon.meta_rl.residual_ppo import ResidualActorCritic, ResidualPPOConfig, stack_records, update
from fluffy_bassoon.meta_rl.robocasa_residual import collect_pending, install_terminal_labels
from fluffy_bassoon.meta_rl.robocasa_residual_parallel import draw_schedule
from fluffy_bassoon.rl.action_residual import ResidualBound


class Motor:
    def __init__(self):
        self.seed_rng = np.random.default_rng(710100)

    def infer(self, observation, *, noise_seed=None):
        seed = int(self.seed_rng.integers(0, 2**63 - 1)) if noise_seed is None else noise_seed
        return np.full((50, 12), .1, np.float32), np.array([seed / 2**63, 1.], np.float32), seed


class Environment:
    def __init__(self, first=0):
        self.cursor = first

    def reset(self):
        self.steps = 0
        self.cursor += 1
        return None

    def step_chunk(self, chunk):
        self.steps += 1
        done = self.steps == 2
        return None, done, dict(id=str(self.cursor)) if done else {}


def learner():
    torch.manual_seed(42)
    policy = ResidualActorCritic(ResidualPPOConfig(2, 6, width=8, epochs=2, minibatch_size=8))
    return policy, torch.optim.Adam(policy.parameters(), lr=policy.config.learning_rate)


def test_eight_partitions_preserve_draws_complete_gae_and_next_adam_update():
    policy, optimizer = learner()
    bound = ResidualBound(12, tuple(range(6)), (.25,) * 6)
    serial_motor, serial_generator = Motor(), torch.Generator().manual_seed(42)
    expected, episodes = collect_pending(policy, bound, serial_motor, Environment(), episodes=16,
                                         generator=serial_generator)
    scheduled_motor, scheduled_generator = Motor(), torch.Generator().manual_seed(42)
    seeds, noise = draw_schedule(scheduled_motor, policy, scheduled_generator, 32, deterministic=False)
    actual, actual_episodes = [], []
    for rank in range(8):
        rows, finished = collect_pending(policy, bound, Motor(), Environment(rank * 2), episodes=2,
            generator=None, planned_motor_seeds=seeds[rank * 4:(rank + 1) * 4],
            planned_action_noise=noise[rank * 4:(rank + 1) * 4])
        actual.extend(rows)
        actual_episodes.extend(finished)
    assert actual_episodes == episodes
    for left, right in zip(expected, actual):
        for key in left:
            if isinstance(left[key], np.ndarray):
                np.testing.assert_array_equal(left[key], right[key])
            else:
                assert left[key] == right[key]
    assert serial_motor.seed_rng.integers(2**63 - 1) == scheduled_motor.seed_rng.integers(2**63 - 1)
    torch.testing.assert_close(torch.randn((1, 6), generator=serial_generator),
                               torch.randn((1, 6), generator=scheduled_generator), rtol=0, atol=0)
    second, second_optimizer = learner()
    labels = {row["id"]: index % 2 for index, row in enumerate(episodes)}
    update(policy, optimizer, stack_records(install_terminal_labels(expected, episodes, labels, arm="S")),
           generator=torch.Generator().manual_seed(43))
    update(second, second_optimizer, stack_records(install_terminal_labels(actual, actual_episodes, labels, arm="S")),
           generator=torch.Generator().manual_seed(43))
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)
    for left, right in zip(optimizer.state_dict()["state"].values(), second_optimizer.state_dict()["state"].values()):
        for key in left:
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)


def test_deterministic_parallel_plan_keeps_training_actor_rng_untouched():
    policy, _ = learner()
    generator = torch.Generator().manual_seed(42)
    before = generator.get_state().clone()
    seeds, noise = draw_schedule(Motor(), policy, generator, 16, deterministic=True)
    assert len(seeds) == len(set(seeds)) == 16 and noise is None
    torch.testing.assert_close(before, generator.get_state(), rtol=0, atol=0)
    rows, episodes = collect_pending(policy, ResidualBound(12, tuple(range(6)), (.25,) * 6),
        Motor(), Environment(), episodes=8, generator=generator, deterministic=True,
        planned_motor_seeds=seeds)
    assert len(episodes) == 8 and all(np.array_equal(row["control"], np.zeros(6)) for row in rows)
    torch.testing.assert_close(before, generator.get_state(), rtol=0, atol=0)
