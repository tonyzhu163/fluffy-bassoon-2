"""Explicit hidden actuator positive control; no motor or simulator ownership.

The public action is the agent's requested pre-calibration command. Condition
parameters and post-calibration commands are audit-only. This is a controlled
robustness task, not a natural RoboTwin improvement or a DSRL reproduction.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re

import numpy as np


GRIPPERS = (6, 13)
ARM_JOINTS = tuple(i for i in range(14) if i not in GRIPPERS)
PUBLIC_HISTORY_FIELDS = ("public_features", "previous_coefficients", "previous_reward",
                         "terminated", "truncated", "student_previous_agent_command")


def contract_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _vector(value, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (14,) or not np.isfinite(array).all():
        raise ValueError(f"Require finite {name}[14]")
    return array


@dataclass(frozen=True)
class CalibrationSpec:
    """Radians on arm joint targets only, applied after the selected steering.

    Limits must be bound to the actual motor/execution range receipt. No robot
    limits are guessed here. The two signs share the same reset distribution;
    this transform is installed only on command execution, never on reset.
    """
    sign: int
    bias_radians: tuple[float, ...]
    command_lower: tuple[float, ...]
    command_upper: tuple[float, ...]
    range_receipt_sha256: str
    motor_sha256: str
    mapping_sha256: str
    reference_basis_sha256: str
    reference_arm_bound: float
    theta: float = .4
    same_mapping_inverse_coefficients: tuple[float, ...] | None = None
    kind: str = "hidden_arm_calibration_positive_control_v1"

    def __post_init__(self):
        if type(self.sign) is not int or self.sign not in (-1, 1):
            raise ValueError("The declared condition distribution has exactly two signs")
        bias = _vector(self.bias_radians, "bias_radians")
        lower, upper = _vector(self.command_lower, "command_lower"), _vector(self.command_upper, "command_upper")
        if (np.any(lower >= upper) or np.any(bias[list(GRIPPERS)] != 0)
                or not np.any(bias) or np.any(np.abs(bias) > .02)):
            raise ValueError("Require nonzero <=.02-radian arm bias, zero grippers and ordered limits")
        if (not math.isfinite(self.reference_arm_bound) or self.reference_arm_bound <= 0
                or not math.isfinite(self.theta) or not 0 < self.theta < 1):
            raise ValueError("Require the actual physical-map bound and a supported theta")
        if self.same_mapping_inverse_coefficients is not None:
            inverse = np.asarray(self.same_mapping_inverse_coefficients)
            if inverse.shape != (4,) or not np.array_equal(inverse, [-self.sign*self.theta, 0., 0., 0.]):
                raise ValueError("Only the declared aligned constant-mode inverse is supported")
        for value in (self.range_receipt_sha256, self.motor_sha256, self.mapping_sha256,
                      self.reference_basis_sha256):
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError("Bind actual range, motor and mapping SHA-256 receipts")

    @property
    def condition_id(self):
        return "calibration_minus" if self.sign < 0 else "calibration_plus"

    def contract(self):
        return dict(**asdict(self), condition_id=self.condition_id,
                    units="decoded arm joint target radians; native absolute grippers unchanged",
                    order="native decode -> selected 4D steering -> calibration bias -> arm limit clipping -> native execution",
                    reset="condition never changes reset, instruction, initial observation or public features",
                    public_history_action="agent pre-calibration request, not post-calibration actuator commands",
                    audit_only=["sign", "condition_id", "bias_radians", "post_calibration_commands"],
                    exact_inverse="command-space -bias before this transform, only inside declared limits; physical execution and same-coefficient-space reachability still require qualification")


class ActuatorCalibration:
    """Driver callback: commands[B,H,14] -> (final_commands, audit).

    Supports NumPy and inference Torch tensors, preserving shape/dtype/device.
    Grippers are copied bit-for-bit. The selected interface's base/proposal
    records remain the driver's canonical records, rather than a second logger.
    """
    def __init__(self, spec):
        if not isinstance(spec, CalibrationSpec):
            raise TypeError("Require a bound CalibrationSpec")
        self.spec = spec

    def __call__(self, base_commands, requested_commands=None):
        commands = base_commands if requested_commands is None else requested_commands
        if (getattr(base_commands, "shape", None) != getattr(commands, "shape", None)
                or getattr(base_commands, "dtype", None) != getattr(commands, "dtype", None)):
            raise ValueError("Base and requested commands must have identical shape/dtype")
        if isinstance(commands, np.ndarray):
            if not isinstance(base_commands, np.ndarray) or not np.isfinite(base_commands).all():
                raise ValueError("Require a finite actual decoded base proposal")
            return self._numpy(commands)
        import torch
        if (not isinstance(commands, torch.Tensor) or commands.ndim != 3
                or commands.shape[-1] != 14 or not all(commands.shape)
                or not commands.is_floating_point() or commands.requires_grad
                or commands.dtype not in (torch.float32, torch.float64)
                or not isinstance(base_commands, torch.Tensor)
                or base_commands.device != commands.device or base_commands.requires_grad
                or not torch.isfinite(commands).all() or not torch.isfinite(base_commands).all()):
            raise ValueError("Require finite inference commands [B,H,14]")
        lower = torch.as_tensor(self.spec.command_lower, dtype=commands.dtype, device=commands.device)
        upper = torch.as_tensor(self.spec.command_upper, dtype=commands.dtype, device=commands.device)
        if ((commands < lower).any() or (commands > upper).any()):
            raise ValueError("Input command exceeds receipt-bound limits; native parity is unqualified")
        bias = torch.as_tensor(self.spec.bias_radians, dtype=commands.dtype,
                               device=commands.device) * self.spec.sign
        proposal = torch.where(bias == 0, commands, commands + bias)
        final = proposal.clamp(min=lower, max=upper)
        final[..., list(GRIPPERS)] = commands[..., list(GRIPPERS)]
        audit = self._audit(commands.detach().cpu().numpy(), proposal.detach().cpu().numpy(),
                            final.detach().cpu().numpy())
        return final, audit

    def _numpy(self, commands):
        if (commands.ndim != 3 or commands.shape[-1] != 14 or not all(commands.shape)
                or commands.dtype.kind != "f" or not np.isfinite(commands).all()):
            raise ValueError("Require finite floating commands [B,H,14]")
        lower = np.asarray(self.spec.command_lower, dtype=commands.dtype)
        upper = np.asarray(self.spec.command_upper, dtype=commands.dtype)
        if np.any(commands < lower) or np.any(commands > upper):
            raise ValueError("Input command exceeds receipt-bound limits; native parity is unqualified")
        bias = np.asarray(self.spec.bias_radians, dtype=commands.dtype) * self.spec.sign
        proposal = np.where(bias == 0, commands, commands + bias)
        final = np.clip(proposal, lower, upper)
        final[..., list(GRIPPERS)] = commands[..., list(GRIPPERS)]
        return final, self._audit(commands, proposal, final)

    def _audit(self, commands, proposal, final):
        return dict(condition_contract_sha256=contract_sha256(self.spec.contract()),
                    condition_id=self.spec.condition_id, sign=self.spec.sign,
                    audit_only=True, public_request=commands.copy(),
                    pre_clip_actuator_commands=proposal.copy(),
                    post_calibration_commands=final.copy(),
                    actuator_delta=final - commands, clipped=(final != proposal),
                    grippers_unchanged=True)

    def inverse_request(self, nominal):
        """Command-space diagnostic, not automatically a 4D teacher action.

        The caller must physically qualify it and, for DPT coefficient labels,
        establish that the selected mapping can realize the inverse.
        """
        nominal = np.asarray(nominal)
        self._numpy(nominal)  # Validate the actual command shape/range.
        bias = self.spec.sign * np.asarray(self.spec.bias_radians, dtype=nominal.dtype)
        request = np.where(bias == 0, nominal, nominal - bias)
        lower, upper = np.asarray(self.spec.command_lower), np.asarray(self.spec.command_upper)
        if np.any(request < lower) or np.any(request > upper):
            raise ValueError("Inverse request is outside the declared command range")
        return request


class CalibrationDistribution:
    """One static canonical contract, switched only at caller-owned trial boundaries."""
    def __init__(self, specs):
        if len(specs) != 2 or {spec.sign for spec in specs} != {-1, 1}:
            raise ValueError("Require exactly the predeclared two conditions")
        contracts = [spec.contract() for spec in specs]
        for key in ("bias_radians", "command_lower", "command_upper", "mapping_sha256",
                    "motor_sha256", "range_receipt_sha256", "reference_basis_sha256",
                    "reference_arm_bound", "theta"):
            if contracts[0][key] != contracts[1][key]:
                raise ValueError("Condition pair differs beyond sign")
        self.specs = {spec.sign: spec for spec in specs}
        self.selected_sign = None

    def select(self, sign):
        if type(sign) is not int or sign not in self.specs:
            raise ValueError("Unknown persistent condition")
        self.selected_sign = sign

    def contract(self):
        return dict(schema="two_hidden_calibration_conditions_v1",
                    conditions=[self.specs[sign].contract() for sign in (-1, 1)],
                    scope="controlled robustness positive control; separate natural A1",
                    switching="only between complete trials/lineages; no reset alteration",
                    public_history_fields=list(PUBLIC_HISTORY_FIELDS), audit_only=True)

    def __call__(self, base_commands, requested_commands=None):
        if self.selected_sign is None:
            raise RuntimeError("Bind a persistent condition before issuing commands")
        return ActuatorCalibration(self.specs[self.selected_sign])(base_commands, requested_commands)


def calibration_pair(*, physical_basis, physical_basis_sha256, arm_joint_bound, selected_arm, command_lower,
                     command_upper, range_receipt_sha256, motor_sha256, mapping_sha256,
                     theta=.4):
    """Predeclared controlled distribution; no adaptive magnitude search.

    The constant physical mode is arm_bound*tanh(c[0]*B[0]). Deriving the
    bias from this SAME mode makes c[0]=-sign*theta a candidate inverse for
    that physical interface; an arbitrary joint-axis bias would not do so.
    For KV/noise this is only an external command-space reference, never a
    valid coefficient label. A2 is not used to select the natural A1 winner.
    """
    from .robotwin_steering import normalize_arm

    selected_arm = normalize_arm(selected_arm)
    basis = np.asarray(physical_basis, dtype=np.float32)
    if (basis.shape != (4, 14) or not np.isfinite(basis).all()
            or np.any(basis[:, list(GRIPPERS)] != 0)
            or np.any(basis[0, [i for i in range(14) if i not in (0, 1, 7, 8)]] != 0)
            or not np.isclose(np.max(np.abs(basis[0])), 1.)
            or not math.isfinite(arm_joint_bound) or arm_joint_bound <= 0
            or not math.isfinite(theta) or not 0 < theta < 1):
        raise ValueError("Require the receipt-bound [4,14] physical basis and constant mode")
    if selected_arm not in ("noise", "kv", "action"):
        raise ValueError("Unknown selected mapping")
    bias = tuple(float(x) for x in arm_joint_bound*np.tanh(theta*basis[0].astype(np.float64)))
    basis_sha = hashlib.sha256(str(basis.dtype).encode() + str(basis.shape).encode()
                               + np.ascontiguousarray(basis).tobytes()).hexdigest()
    if basis_sha != physical_basis_sha256:
        raise ValueError("Actual physical basis differs from the admitted mapping digest")
    return tuple(CalibrationSpec(sign, bias, tuple(command_lower), tuple(command_upper),
                                range_receipt_sha256, motor_sha256, mapping_sha256,
                                basis_sha, float(arm_joint_bound), float(theta),
                                (-sign*theta, 0., 0., 0.)
                                if selected_arm == "action" else None)
                 for sign in (-1, 1))


def verify_same_initial_public_observation(first, second):
    """Counterfactual reset audit before any conditioned commands execute."""
    required = ("task", "requested_seed", "actual_seed", "observation_sha256", "features_sha256")
    if (any(key not in first or key not in second for key in required)
            or any(first[key] != second[key] for key in required)):
        raise ValueError("Conditions do not share the same accepted initial public input")
    for key in ("observation_sha256", "features_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", first[key]):
            raise ValueError("Initial public input must have actual SHA-256 records")
    return dict(passed=True, scope="recorded initial public observation only",
                condition_not_applied_during_reset=True,
                transition_information="Later sensor dynamics may reveal the calibration; no reward-causal claim")


def verify_inverse_command_cancellation(base, final, clipped):
    """Command-level arithmetic check; no physical trajectory identity claim."""
    base, final, clipped = np.asarray(base), np.asarray(final), np.asarray(clipped)
    if (base.shape != final.shape or base.ndim != 3 or base.shape[-1] != 14
            or base.dtype.kind != "f" or final.dtype != base.dtype
            or clipped.shape != base.shape or clipped.dtype != np.bool_
            or not np.isfinite(base).all() or not np.isfinite(final).all()
            or clipped.any()):
        raise ValueError("Inverse proof needs finite identical command layouts and no clipping")
    if not np.array_equal(np.ascontiguousarray(base[..., list(GRIPPERS)]).view(np.uint8),
                          np.ascontiguousarray(final[..., list(GRIPPERS)]).view(np.uint8)):
        raise ValueError("Calibration inverse changed native grippers")
    tolerance = max(1e-6, 8*np.finfo(base.dtype).eps*max(1., float(np.max(np.abs(base)))))
    residual = float(np.max(np.abs(base.astype(np.float64)-final.astype(np.float64))))
    if residual > tolerance:
        raise ValueError("Actual inverse command residual exceeds the declared roundoff tolerance")
    return dict(passed=True, max_absolute_command_residual=residual,
                tolerance_command_units=tolerance, joint_limit_clipping=False,
                grippers_unchanged=True, scope="command cancellation only; real source episodes still qualify the teacher",
                bitwise_physical_trajectory_identity_claim=False)
