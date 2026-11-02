"""PPO on a frozen motor with ALL sampling noise inside its policy action.

The pre-action feature port and deterministic execution map are shared by
ordinary PPO, its history source and the deployed history controller. Existing
residual-PPO runs and checkpoints keep their original semantics.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import random

import numpy as np
import torch

from .residual_ppo import (ResidualActorCritic, ResidualPPOConfig, history_trial,
                           stack_records, update)


KV_ALL_LAYERS_MAP = "independent_layer_kv_rademacher32_rms_v1"


@dataclass(frozen=True)
class InterfaceSpec:
    kind: str = "action"
    motor_shape: tuple[int, int] = (50, 32)
    steering_dim: int = 6
    candidates: int = 8
    fraction: float = .25
    kv_layers: int | None = None

    def __post_init__(self):
        all_kv = self.kind == "kv_all_layers"
        if (all_kv and (type(self.kv_layers) is not int or self.kv_layers < 1)
                or not all_kv and self.kv_layers is not None):
            raise ValueError("Bind kv_layers only for the all-layer K/V interface")
        dimensions = {"action": 6, "noise": 0, "conditioning": 32, "conditioning_all": 32,
                      "velocity": 32, "select": self.candidates, "direct": 600,
                      "kv_all_layers": 2 * self.kv_layers * 32 if all_kv else None}
        if (self.kind not in dimensions or self.steering_dim != dimensions[self.kind]
                or self.motor_shape != (50, 32) or type(self.candidates) is not int or self.candidates < 2
                or not math.isfinite(self.fraction) or self.fraction <= 0):
            raise ValueError("Require the declared H50/32D steering interface")

    @property
    def noise_dim(self):
        # Gate0 noise32 is the tied tanh map. Other interfaces retain native
        # iid full-dimensional N(0,1) noise, now a likelihood-scored action.
        return 32 if self.kind == "noise" else 0 if self.kind == "direct" else 1600 * (
            self.candidates if self.kind == "select" else 1)

    @property
    def control_dim(self):
        return self.noise_dim + self.steering_dim

    def record(self):
        record = asdict(self)
        record["motor_shape"] = list(self.motor_shape)
        if self.kind == "kv_all_layers":
            record["kv_map"] = dict(version=KV_ALL_LAYERS_MAP, rank=32, targets=["key", "value"],
                order="layer,key_or_value,coefficient", seed_base=921003,
                seed="921003 + 2 * layer_index + key_or_value_index",
                scale="fraction * per_row_per_layer_per_tensor_RMS; tanh coefficients")
        else:
            record.pop("kv_layers")  # Preserve existing checkpoint contracts.
        return record

    def decode(self, control):
        control = np.asarray(control, dtype=np.float32)
        if control.shape != (self.control_dim,) or not np.isfinite(control).all():
            raise ValueError("Require the entire finite joint policy action")
        noise = control[:self.noise_dim]
        if self.kind == "noise":
            noise = np.broadcast_to(np.tanh(noise), self.motor_shape).copy()[None]
        elif self.kind == "direct":
            noise = np.empty((0, *self.motor_shape), dtype=np.float32)
        else:
            noise = noise.reshape(-1, *self.motor_shape).copy()
        return noise, control[self.noise_dim:].copy()


def policy_config(feature_dim, spec, **overrides):
    return ResidualPPOConfig(feature_dim, spec.control_dim,
                            **dict(dict(gamma=.99, gae_lambda=.95, epochs=10,
                                        minibatch_size=32, target_kl=.02), **overrides))


def execute_control(motor, prepared, control, spec):
    """Identical deterministic map for PPO and meta-learner deployment."""
    noise, steering = spec.decode(control)
    return motor.execute(prepared, noise=noise, steering=steering, spec=spec)


class FrozenInterfaceController:
    """History deployment through the exact PPO feature/noise/steering ports."""
    def __init__(self, controller, spec):
        if controller.policy.config.noise_dim != spec.control_dim:
            raise ValueError("History action must include the same motor noise and steering")
        self.controller, self.spec = controller, spec

    def reset_trial(self):
        self.controller.reset_trial()

    def act(self, motor, observation, *, deterministic=True):
        features, prepared = motor.prepare(observation)
        control = self.controller.act(features, deterministic=deterministic)
        return execute_control(motor, prepared, control, self.spec), control

    def observe(self, control, reward, done):
        self.controller.observe(control, reward, done)


def collect(policy, motor, environment, spec, reward, *, episodes, generator, deterministic=False):
    """Collect first-success transitions with a truth-free learned-reward port.

    prepare(obs)->(features, cache) performs no sampling. execute() takes only
    explicit noise/steering and frozen weights. step_chunk() returns public obs,
    horizon_done and a payload with executed_controls and opaque video refs.
    reward(payload)->{first_success_step,reward}; Flora must never read truth.
    Learned success ends the learner episode at its confirmation chunk. Any
    physical post-success suffix remains in the environment's action ledger.
    end_episode() closes videos/accounting even on early learned termination.
    """
    if (type(episodes) is not int or episodes < 1 or policy.config.residual_dim != spec.control_dim
            or policy.config.gamma != .99):
        raise ValueError("Require positive episodes, matching joint action and gamma=.99")
    reference = next(policy.parameters())
    records = []
    for _ in range(episodes):
        observation, start, step, done = environment.reset(), True, 0, False
        while not done:
            features, prepared = motor.prepare(observation)
            features = np.array(features, dtype=np.float32, copy=True)
            if features.shape != (policy.config.feature_dim,) or not np.isfinite(features).all():
                raise ValueError("Require pre-action public features")
            control, log_prob, value = policy.act(torch.as_tensor(features[None], device=reference.device),
                                                  generator=generator, deterministic=deterministic)
            control = control[0].cpu().numpy()
            chunk = execute_control(motor, prepared, control, spec)
            observation, horizon_done, payload = environment.step_chunk(chunk)
            outcome = reward(payload)
            increment = payload["executed_controls"]
            if type(increment) is not int or not 1 <= increment <= spec.motor_shape[0]:
                raise ValueError("Require observed physical control count")
            first, scalar = outcome["first_success_step"], float(outcome["reward"])
            if (not math.isfinite(scalar) or not 0 <= scalar <= 1
                    or first is not None and (type(first) is not int or not step < first <= step + increment)
                    or first is None and scalar != 0):
                raise ValueError("Reward must correspond to first success in this executed chunk")
            done = bool(horizon_done or first is not None)
            records.append(dict(features=features, control=control.copy(),
                                log_prob=float(log_prob[0]), value=float(value[0]), reward=scalar,
                                done=done, episode_start=start, executed_controls=increment,
                                first_success_step=first))
            start, step = False, step + increment
        environment.end_episode()
    return records


class InterfaceLearner:
    """Ordinary PPO with explicit interface and complete boundary recovery."""
    def __init__(self, feature_dim, spec, *, seed=0, device="cpu", **overrides):
        self.spec = spec
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.policy = ResidualActorCritic(policy_config(feature_dim, spec, **overrides)).to(device)
        with torch.no_grad():
            # Preserve the native iid noise law on the native-noise interfaces.
            if spec.kind != "noise":
                self.policy.log_std[:spec.noise_dim].zero_()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.policy.config.learning_rate)
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.updates = self.controls = 0

    def train_batch(self, records):
        metrics = update(self.policy, self.optimizer,
                         stack_records(records, device=next(self.policy.parameters()).device), generator=self.generator)
        self.updates += 1
        self.controls += sum(r["executed_controls"] for r in records)
        return metrics

    def state_dict(self, *, sampler, environment):
        return deepcopy(dict(format="frozen_motor_interface_ppo_v1", interface=self.spec.record(),
                             config=asdict(self.policy.config), policy=self.policy.state_dict(),
                             optimizer=self.optimizer.state_dict(), generator=self.generator.get_state(),
                             sampler=sampler.state_dict(), environment=environment.state_dict(),
                             torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                             numpy_rng=np.random.get_state(),
                             cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                             updates=self.updates, controls=self.controls))

    def load_state_dict(self, state, *, sampler, environment):
        if (state["format"] != "frozen_motor_interface_ppo_v1" or state["interface"] != self.spec.record()
                or state["config"] != asdict(self.policy.config)):
            raise ValueError("Checkpoint interface or optimizer contract changed")
        self.policy.load_state_dict(state["policy"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.generator.set_state(state["generator"])
        sampler.load_state_dict(state["sampler"])
        environment.load_state_dict(state["environment"])
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        if state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        self.updates, self.controls = state["updates"], state["controls"]

    @staticmethod
    def history(records_by_update):
        return history_trial(records_by_update)
