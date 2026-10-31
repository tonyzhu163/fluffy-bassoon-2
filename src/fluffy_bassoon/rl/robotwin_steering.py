"""Fixed four-coefficient control maps for the bounded frozen-Motus pilot.

No learner, allocation or second simulator loop. All maps use the existing
scalar RoboTwin backend, current public2062 features and H32/five-step motor.
"""
from copy import deepcopy
import hashlib
import json
import math

import numpy as np
import torch
from torch import nn

from .pi05_residual import DecodedActionResidual, ReactiveALOHAActionResidual, ReactiveKVResidual, _add


ARMS = ("noise", "kv", "action")
TASKS = ("place_phone_stand", "place_object_stand")
DIMENSION = 4
ARM_DESCRIPTIONS = {"noise_coefficients":"noise", "native_KV_coefficients":"kv",
                    "decoded_ALOHA_joint_coefficients":"action"}


def normalize_arm(name):
    value = ARM_DESCRIPTIONS.get(name, name)
    if value not in ("native", *ARMS):
        raise ValueError("Unknown restricted steering arm")
    return value


def clipped_coefficients(proposed):
    value = np.asarray(proposed, dtype=np.float32)
    if value.shape != (4,) or not np.isfinite(value).all():
        raise ValueError("Require four finite proposed coefficients")
    return value.copy(), np.clip(value, -1., 1.)


def fixed_basis(seed, shape):
    value = np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
    value /= np.maximum(np.abs(value).reshape(4, -1).max(1), 1e-12).reshape(4, *([1] * (len(shape)-1)))
    return value


def digest(value):
    array = np.ascontiguousarray(value)
    return hashlib.sha256(str(array.dtype).encode() + str(array.shape).encode() + array.tobytes()).hexdigest()


def temporal_map(position):
    """Public fixed basis [1, 2x-1, sin(pi*x), cos(pi*x)] on normalized slots."""
    return torch.stack((torch.ones_like(position), 2*position-1,
                        (math.pi*position).sin(), (math.pi*position).cos()), -1)


def validate_interface(contract):
    required = dict(schema="motus_coefficient_interface_v1", action_horizon=32,
                    flow_steps=5, decoded_action_dim=14, public_dim=2062,
                    coefficient_dim=4, coefficient_bound=1., gripper_delta=0.,
                    joint_units="radians", gripper_indices=[6, 13])
    if any(contract.get(k) != v for k, v in required.items()):
        raise ValueError("Interface differs from the explicit frozen Motus/ALOHA four-coefficient contract")
    for name in ("arm_joint_bound", "noise_delta_bound", "kv_delta_bound"):
        if (type(contract.get(name)) not in (int, float) or not math.isfinite(contract[name])
                or contract[name] <= 0):
            raise ValueError(f"Invalid explicit {name}")
    for name in ("basis_seed", "kv_heads", "head_dim", "kv_layers"):
        if type(contract.get(name)) is not int or contract[name] < int(name != "basis_seed"):
            raise ValueError(f"Invalid {name}")
    for name in ("motor_sha256", "normalization_sha256", "joint_convention_sha256"):
        if (not isinstance(contract.get(name), str) or len(contract[name]) != 64
                or any(c not in "0123456789abcdef" for c in contract[name])):
            raise ValueError(f"Require exact {name}")
    lower, upper = np.asarray(contract.get("joint_lower")), np.asarray(contract.get("joint_upper"))
    if (lower.shape != (14,) or upper.shape != (14,) or lower.dtype.kind not in "fi" or upper.dtype.kind not in "fi"
            or not np.isfinite(lower).all() or not np.isfinite(upper).all() or not (lower < upper).all()):
        raise ValueError("Require verified finite14D joint ranges, including absolute grippers")
    arms = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
    if contract["arm_joint_bound"] > .1 * (upper-lower)[arms].min():
        raise ValueError("Arm correction exceeds one tenth of a verified native arm-joint range")
    # No guessed clipping of absolute decoded joint targets; zero residual must
    # preserve the native proposal. The receipt establishes units and ranges.
    return deepcopy(contract)


class FixedKVResidual(ReactiveKVResidual):
    """A fixed rank<=4 direction family, not the learned reactive adapter."""
    def __init__(self, contract):
        nn.Module.__init__(self)
        self.public_dim = contract["public_dim"]
        self.kv_heads, self.head_dim = contract["kv_heads"], contract["head_dim"]
        self.scale = contract["kv_delta_bound"]
        array = fixed_basis(contract["basis_seed"] + 1, (4, 2*self.kv_heads*self.head_dim))
        self.basis_sha256 = digest(array)
        self.register_buffer("basis", torch.from_numpy(array))
        self.register_buffer("coefficients", torch.zeros(4))
        self.eval()

    def config(self):
        return dict(kind="fixed_rank4_last_layer_kv_v1", basis_sha256=self.basis_sha256,
                    coefficient_dim=4, delta_bound=self.scale, units="native post-RoPE cache units",
                    temporal="valid-prefix rank [1,2x-1,sin(pi*x),cos(pi*x)]", layer=-1)

    def forward(self, cache, mask, public_features):
        if (not cache or mask.dtype != torch.bool or mask.ndim != 2 or not mask.any(1).all()
                or public_features.shape != (mask.shape[0], self.public_dim)):
            raise ValueError("Invalid native cache/public-feature binding")
        expected = (*mask.shape, self.kv_heads, self.head_dim)
        if any(len(pair) != 2 or any(t.shape != expected or t.device != self.basis.device for t in pair) for pair in cache):
            raise ValueError("Native KV layout differs from the bound interface")
        key, value = cache[-1]
        position = (mask.long().cumsum(1)-1).clamp_min(0) / (mask.sum(1, keepdim=True)-1).clamp_min(1)
        weights = temporal_map(position.to(self.basis.dtype)) * self.coefficients
        delta = self.scale * (weights @ self.basis) / 4
        dk, dv = delta.chunk(2, -1)
        changed = tuple(torch.where(mask[..., None, None],
            _add(base.detach(), d.reshape_as(base).to(base.dtype)), base.detach())
            for base, d in ((key, dk), (value, dv)))
        return tuple(tuple(t.detach() for t in pair) for pair in cache[:-1]) + (changed,)


class FixedActionResidual(ReactiveALOHAActionResidual):
    """Four fixed decoded arm-joint directions; grippers are never corrected."""
    def __init__(self, contract):
        nn.Module.__init__(self)
        self.public_dim, self.scale = contract["public_dim"], contract["arm_joint_bound"]
        array = fixed_basis(contract["basis_seed"] + 2, (4, 14))
        array[:, [6, 13]] = 0
        array[0, [2, 3, 4, 5, 9, 10, 11, 12]] = 0
        array[0] /= max(float(np.abs(array[0]).max()), 1e-12)
        self.basis_sha256 = digest(array)
        self.register_buffer("basis", torch.from_numpy(array))
        self.register_buffer("coefficients", torch.zeros(4))
        self.eval()

    def config(self):
        return dict(kind="fixed_rank4_decoded_aloha_arm_v1", basis_sha256=self.basis_sha256,
                    coefficient_dim=4, arm_joint_bound=self.scale, gripper_delta=0.,
                    units="decoded arm joint radians; absolute grippers unchanged",
                    temporal="action-slot [1,2x-1,sin(pi*x),cos(pi*x)]", absolute_command_clipping=False,
                    mapping="arm_bound*tanh((temporal*coefficients)@basis); NO divisor",
                    constant_basis_joint_indices=[0, 1, 7, 8])

    def forward(self, base_actions, public_features):
        if (base_actions.ndim != 3 or base_actions.shape[-1] != 14 or not torch.isfinite(base_actions).all()
                or public_features.shape != (base_actions.shape[0], self.public_dim)):
            raise ValueError("Require finite decoded ALOHA commands and bound public features")
        base = base_actions.detach()
        position = torch.linspace(0, 1, base.shape[1], device=base.device, dtype=self.basis.dtype)
        delta = (self.scale * ((temporal_map(position)*self.coefficients) @ self.basis).tanh()).to(base.dtype)
        delta = delta[None].expand_as(base)
        actions = _add(base, delta)
        if not torch.equal(actions[..., [6, 13]], base[..., [6, 13]]):
            raise RuntimeError("Decoded arm mapping changed an absolute gripper")
        return DecodedActionResidual(base, delta, actions-base, torch.zeros_like(base, dtype=torch.bool), actions)


class CoefficientSteering:
    """Shared A1/A2 motor API. The caller owns whole-episode budget admission.

    Public history is the proposed/clipped coefficients and pre-condition
    command. Post-condition commands and latent condition fields are audit-only.
    """
    def __init__(self, env, interface_contract, *, post_command_transform=None, condition_contract=None,
                 interface_sha256=None):
        self.interface = validate_interface(interface_contract)
        self.interface_sha256 = interface_sha256 or hashlib.sha256(json.dumps(
            self.interface, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if (tuple(env.noise_shape) != (32, 32) or env.observation_dim != 2062
                or env.backend.model.num_steps != 5 or env.corpus_writer is not None):
            raise ValueError("Require an unrecorded scalar native H32/five-step Motus environment")
        if post_command_transform is not None and not isinstance(condition_contract, dict):
            raise ValueError("A post-command condition needs a separate explicit audit contract")
        self.env = env
        device = env.backend.model.device
        self.kv = FixedKVResidual(self.interface).to(device)
        self.action = FixedActionResidual(self.interface).to(device)
        self.noise_basis = fixed_basis(self.interface["basis_seed"], (4, 32, 32))
        self.condition_contract = deepcopy(condition_contract)
        self.post_command_transform = post_command_transform
        self.last_mapping = None
        self.arm, self.proposed, self.coefficients = "native", np.zeros(4, np.float32), np.zeros(4, np.float32)
        self.motor_rng = None
        self.previous_agent_command = np.zeros(14, np.float32)
        self._pending_metadata = {}
        env.backend.post_command_transform = post_command_transform
        env.backend.record_agent_requested_actions = True
        env.backend.contract = dict(env.backend.contract, coefficient_steering=self.config())
        env.contract = env.backend.contract

    def __getattr__(self, name):
        return getattr(self.env, name)

    def config(self):
        return dict(schema="motus_four_coefficient_mapping_v1", interface=self.interface,
                    noise_basis_sha256=digest(self.noise_basis), kv=self.kv.config(), action=self.action.config(),
                    noise="native independent Gaussian H32x32 plus bounded coefficient delta; total noise NOT clipped",
                    coefficient_clipping=[-1., 1.], condition_audit_contract=self.condition_contract,
                    order="coefficient clip -> fixed mapping -> native decode -> optional arm calibration -> native execution")

    def attach_corpus(self, writer):
        if self.env.active or self.env.corpus_writer is not None:
            raise RuntimeError("Attach the canonical corpus before the first reset")
        writer.register_runtime(self.env.backend.contract)
        self.env.corpus_writer = self.env.backend.corpus_writer = writer

    def set_coefficients(self, arm, proposed):
        arm = normalize_arm(arm)
        self.proposed, self.coefficients = clipped_coefficients(proposed)
        if arm == "native" and np.any(self.coefficients):
            raise ValueError("Native control requires zero coefficients")
        self.arm = arm
        with torch.no_grad():
            for hook in (self.kv, self.action):
                hook.coefficients.copy_(torch.as_tensor(self.coefficients, device=hook.coefficients.device))
        self.env.backend.kv_adapter = self.kv if arm == "kv" else None
        self.env.backend.action_residual = self.action if arm == "action" else None

    def reset(self, task, seed, *, motor_seed):
        if task not in TASKS:
            raise ValueError("Only the two declared source candidate tasks are admitted")
        self.motor_rng = np.random.default_rng(motor_seed)
        self.previous_agent_command = np.zeros(14, np.float32)
        features = self.env.reset(task, seed)
        cache = self.env.backend._kv_cache
        if len(cache) != self.interface["kv_layers"] or any(
            any(t.shape[-2:] != (self.interface["kv_heads"], self.interface["head_dim"]) for t in pair)
                for pair in cache):
            raise ValueError("Actual native KV cache differs from the source receipt")
        return features

    def step(self, proposed):
        if self.motor_rng is None:
            raise RuntimeError("Reset with an explicit motor seed before stepping")
        self.set_coefficients(self.arm, proposed)
        before = deepcopy(self.motor_rng.bit_generator.state)
        native = self.motor_rng.standard_normal((32, 32)).astype(np.float32)
        delta = np.zeros_like(native)
        if self.arm == "noise":
            x = np.linspace(0, 1, 32, dtype=np.float32)
            temporal = np.stack((np.ones_like(x), 2*x-1, np.sin(np.pi*x), np.cos(np.pi*x)))
            delta = self.interface["noise_delta_bound"] * np.einsum(
                "r,rh,rha->ha", self.coefficients, temporal, self.noise_basis) / 4
        mapped = native + delta
        public = dict(arm=self.arm, proposed_coefficients=self.proposed.copy(),
                      clipped_coefficients=self.coefficients.copy(), coefficient_clipped=self.proposed != self.coefficients,
                      native_initial_noise=native, mapped_initial_noise=mapped)
        self.last_mapping = dict(public, audit_only=dict(mapping=self.config(), motor_rng_before=before,
            motor_rng_after=deepcopy(self.motor_rng.bit_generator.state)))
        self.env.record_decision_metadata(dict(self._pending_metadata,
            steering_coefficients=self.coefficients.copy(), proposed_steering_coefficients=self.proposed.copy(),
            steering_mapping_sha256=self.mapping_sha256(), steering_interface_sha256=self.interface_sha256,
            student_previous_agent_command=self.previous_agent_command.copy(),
            coefficient_steering=self.last_mapping))
        self._pending_metadata = {}
        outcome = self.env.step(mapped)
        if "last_valid_agent_request" not in outcome[4]:
            raise RuntimeError("Backend did not bind public command history to the actual native valid mask")
        if outcome[4]["last_valid_agent_request"] is not None:
            self.previous_agent_command = np.asarray(outcome[4]["last_valid_agent_request"], np.float32).copy()
        return (*outcome[:4], dict(outcome[4], coefficient_steering=dict(
            arm=self.arm, proposed_coefficients=self.proposed.copy(), applied_coefficients=self.coefficients.copy())))

    def record_decision_metadata(self, metadata):
        self._pending_metadata = deepcopy(metadata)

    def snapshot(self):
        return dict(mapping=self.config(), arm=self.arm, proposed_coefficients=self.proposed.copy(),
                    coefficients=self.coefficients.copy(),
                    noise_basis=self.noise_basis.copy(), kv_basis=self.kv.basis.detach().cpu().clone(),
                    physical_basis=self.physical_basis(),
                    previous_agent_command=self.previous_agent_command.copy(),
                    motor_rng=None if self.motor_rng is None else deepcopy(self.motor_rng.bit_generator.state))

    def physical_basis(self):
        """Audit/control-design binding; not an automatically valid inverse label."""
        return self.action.basis.detach().cpu().numpy().copy()

    def mapping_sha256(self):
        binding = self.config()
        binding.pop("condition_audit_contract")
        binding["selected_arm"] = self.arm
        return hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def student_observation(self, native_features):
        """2076D causal student view; teacher/native APIs remain2062D."""
        value = np.asarray(native_features, np.float32)
        if value.shape != (2062,):
            raise ValueError("Require native2062 public features before appending previous requested command")
        return np.concatenate((value, self.previous_agent_command)).astype(np.float32)
