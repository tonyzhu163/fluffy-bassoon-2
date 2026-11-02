"""Core physical-attempt causality and recovery; allocated Linux only."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.steering_history_eval import collect_case_attempts, weights_digest
from fluffy_bassoon.meta_rl.vla_history import HistoryConfig, HistoryNoisePolicy


class Environment:
    def __init__(self, case, attempt, *, paired=True, early_stop=False):
        self.case, self.attempt, self.horizon = case, attempt, 100
        self.initial_evidence = dict(pairing=dict(matched=paired))
        self.steps, self.success_steps = 0, []
        self.closed, self.early_stop = False, early_stop

    def reset(self):
        return np.ones(3, np.float32)

    def step_chunk(self, commands):
        assert commands.shape == (50, 12)
        self.steps += 50
        if self.steps == 50:
            self.success_steps = [25]
        return np.ones(3, np.float32), self.early_stop or self.steps == self.horizon, {}

    def close(self):
        self.closed = True


class Motor:
    def __init__(self):
        self.executed = []

    def prepare(self, observation):
        return observation, "prepared"

    def execute(self, prepared, *, noise, steering, spec):
        assert prepared == "prepared" and steering.shape == (spec.steering_dim,)
        assert noise.shape == (1, 50, 32)
        self.executed.append((noise.copy(), steering.copy()))
        return np.zeros((50, 12), np.float32)


def setup(dim=32):
    torch.manual_seed(63)
    policy = HistoryNoisePolicy(HistoryConfig(3, dim, width=8, heads=2, layers=1, context_length=12))
    return policy, SimpleNamespace(steering_dim=dim)


def collect(policy, spec, case, *, completed_prefix=(), attempts=5, paired=True, early_stop=False):
    environments, before, after, completed = [], [], [], []
    motor = Motor()

    def factory(case, attempt):
        env = Environment(case, attempt, paired=paired, early_stop=early_stop)
        environments.append(env)
        return env

    def noise_for(case, attempt, chunk):
        seed = 1000 + attempt * 100 + chunk
        return np.random.default_rng(seed).standard_normal((50, 32)).astype(np.float32), dict(seed=seed)

    def completed_attempt(result, controller):
        assert not controller.pending and controller.previous_done
        completed.append(result)

    kwargs = dict(horizon=100, environment_factory=factory, motor=motor, noise_for=noise_for,
        before_dispatch=lambda row, env: before.append((deepcopy(row), env.steps)),
        after_dispatch=lambda row, env: after.append((deepcopy(row), env.steps)),
        completed_attempt=completed_attempt, attempts=attempts, completed_prefix=completed_prefix)
    return kwargs, environments, motor, before, after, completed


@pytest.mark.parametrize("dim", [32, 1152])
def test_full_attempts_keep_actual_feedback_and_memory_then_isolate_cases(dim):
    policy, spec = setup(dim)
    before_weights = weights_digest(policy)
    args, environments, motor, before, after, complete = collect(policy, spec, dict(task="A", seed=42))
    results = collect_case_attempts(policy, spec, dict(task="A", seed=42), **args)
    assert len(results) == len(complete) == 5
    assert [r["memory_tokens"] for r in results] == [2, 4, 6, 8, 10]
    assert all(env.closed and env.steps == 100 and env.case == dict(task="A", seed=42)
               for env in environments)
    assert [steps for _, steps in before] == [0, 50] * 5
    assert [steps for _, steps in after] == [50, 100] * 5
    assert [row["reward"] for result in results for row in result["records"]] == [1., 0.] * 5
    assert all(result["success_once"] and not result["success_at_end"]
               and result["first_success_step"] == 25 for result in results)
    assert not np.array_equal(motor.executed[0][1], motor.executed[2][1])
    # Identical public features but a fresh physical case must start with no past-case memory.
    fresh_args, *_ = collect(policy, spec, dict(task="B", seed=99), attempts=1)
    fresh = collect_case_attempts(policy, spec, dict(task="B", seed=99), **fresh_args)
    np.testing.assert_array_equal(fresh[0]["records"][0]["control"], results[0]["records"][0]["control"])
    assert weights_digest(policy) == before_weights
    assert all(not parameter.requires_grad for parameter in policy.parameters())


def test_recovery_replays_only_complete_prefix_and_runs_only_remaining_physics():
    policy, spec = setup()
    case = dict(task="A", seed=42)
    args, *_ = collect(policy, spec, case)
    original = collect_case_attempts(policy, spec, case, **args)
    args, environments, _, _, _, _ = collect(policy, spec, case, completed_prefix=original[:2])
    recovered = collect_case_attempts(policy, spec, case, **args)
    assert [env.attempt for env in environments] == [3, 4, 5]
    assert [row["memory_tokens"] for row in recovered] == [2, 4, 6, 8, 10]
    for old, new in zip(original, recovered, strict=True):
        for a, b in zip(old["records"], new["records"], strict=True):
            np.testing.assert_array_equal(a["control"], b["control"])
            np.testing.assert_array_equal(a["native_noise"], b["native_noise"])
    changed = deepcopy(original[:1])
    changed[0]["records"][0]["control"][0] += .25
    args, environments, *_ = collect(policy, spec, case, completed_prefix=changed)
    with pytest.raises(ValueError, match="differs"):
        collect_case_attempts(policy, spec, case, **args)
    assert not environments
    changed = deepcopy(original[:1])
    changed[0]["records"].pop()
    args, environments, *_ = collect(policy, spec, case, completed_prefix=changed)
    with pytest.raises(ValueError, match="complete"):
        collect_case_attempts(policy, spec, case, **args)
    assert not environments


def test_pairing_capacity_and_early_stop_fail_before_fabricated_completion():
    policy, spec = setup()
    case = dict(task="A", seed=42)
    args, environments, motor, before, after, completed = collect(policy, spec, case, paired=False)
    with pytest.raises(ValueError, match="canonical"):
        collect_case_attempts(policy, spec, case, **args)
    assert environments[0].closed and not motor.executed and not before and not after and not completed
    args, environments, motor, _, _, _ = collect(policy, spec, case)
    wrong = SimpleNamespace(steering_dim=1152)
    with pytest.raises(ValueError, match="dimensions"):
        collect_case_attempts(policy, wrong, case, **args)
    assert not environments and not motor.executed
    args, environments, _, _, after, completed = collect(policy, spec, case, early_stop=True)
    with pytest.raises(ValueError, match="suffix"):
        collect_case_attempts(policy, spec, case, **args)
    assert environments[0].closed and not after and not completed


@pytest.mark.parametrize("kind", ["velocity", "kv_all_layers"])
def test_physical_loop_consumes_the_real_native_interface_serial_api(kind):
    from test_interface_ppo import batch_native
    from fluffy_bassoon.meta_rl.interface_ppo import InterfaceSpec
    from fluffy_bassoon.meta_rl.robocasa_interface import FrozenRoboCasaInterface
    native, observation = batch_native()
    native.model.layers = 18
    observation = dict(states=observation["states"][:1])
    spec = (InterfaceSpec(kind=kind, steering_dim=32, fraction=.05) if kind == "velocity"
            else InterfaceSpec(kind=kind, steering_dim=1152, kv_layers=18, fraction=.04))
    actor = HistoryNoisePolicy(HistoryConfig(20, spec.steering_dim, width=8,
                                            heads=2, layers=1, context_length=12))
    args, environments, _, before, after, completed = collect(actor, spec, dict(task="A"))

    class PublicEnvironment(Environment):
        def reset(self):
            return deepcopy(observation)
        def step_chunk(self, commands):
            _, done, info = super().step_chunk(commands)
            return deepcopy(observation), done, info

    def factory(case, attempt):
        env = PublicEnvironment(case, attempt)
        environments.append(env)
        return env

    args.update(environment_factory=factory, motor=FrozenRoboCasaInterface(native))
    results = collect_case_attempts(actor, spec, dict(task="A"), **args)
    assert len(results) == len(completed) == 5 and len(before) == len(after) == 10
    assert native.model.prefixes == native.model.draws == 10
    assert [row["memory_tokens"] for row in results] == [2, 4, 6, 8, 10]
    assert all(row["records"][0]["native_noise"].shape == (50, 32) for row in results)
