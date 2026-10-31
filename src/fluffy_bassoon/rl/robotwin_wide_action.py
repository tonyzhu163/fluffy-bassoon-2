"""Prospective wide decoded-action variant; no motor, learner or runtime owner.

This is deliberately separate from the qualified 0.05/zero-gripper capacity
head. The original motor, native H32 executor and native gripper clip stay
unchanged. Construct only with source-bound arm limits and explicit provenance.
"""
from copy import deepcopy
import hashlib
import json
import re

import torch

from .pi05_residual import ReactiveALOHAActionResidual, DecodedActionResidual


KIND = 'robotwin_wide_decoded_aloha_residual_v1'
ARM_INDICES = (0,1,2,3,4,5,7,8,9,10,11,12)
GRIPPER_INDICES = (6,13)
PUBLIC_DIM, WIDTH = 2062, 64
ARM_BOUND, GRIPPER_BOUND = 2.0, 1.1


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


class _ExactZeroAdd(torch.autograd.Function):
    """Bitwise zero identity in forward, ordinary delta derivative at zero.

    The generic residual uses this signed-zero preservation in inference only.
    This variant also guarantees it during fitting without killing the initial
    output-layer gradient. Native base/features are already detached upstream.
    """
    @staticmethod
    def forward(ctx, base, delta):
        return torch.where(delta == 0, base, base + delta)

    @staticmethod
    def backward(ctx, gradient):
        return None, gradient


class WideActionResidual(ReactiveALOHAActionResidual):
    """2062 current features + detached base14 + slot ->64 -> bounded14.

    Arm edits are <=2rad and clip to actual12-arm URDF headroom. Gripper edits
    are <=1.1 raw absolute-command units, with no policy-level gripper clamp.
    Zero outputs preserve every raw native bit including out-of-[0,1] grips.
    Provenance hashes identify the unchanged motor/norm and reviewed limits;
    verifying those files against the actual runtime remains the caller's job.
    """
    def __init__(self, *, joint_lower, joint_upper, motor_sha256, normalization_sha256,
                 joint_convention_sha256, base_interface_sha256):
        bindings=dict(motor_sha256=motor_sha256,normalization_sha256=normalization_sha256,
            joint_convention_sha256=joint_convention_sha256,base_interface_sha256=base_interface_sha256)
        if any(not isinstance(v,str) or not re.fullmatch('[0-9a-f]{64}',v) for v in bindings.values()):
            raise ValueError('Require four explicit source SHA256 bindings; legacy interface does not declare wide bounds')
        lower=torch.as_tensor(joint_lower,dtype=torch.float32).clone()
        upper=torch.as_tensor(joint_upper,dtype=torch.float32).clone()
        if (lower.shape!=(14,) or upper.shape!=(14,) or not torch.isfinite(lower).all()
                or not torch.isfinite(upper).all() or not (lower<upper).all()):
            raise ValueError('Require finite ordered actual source limits14 before selecting12 arms')
        super().__init__(public_dim=PUBLIC_DIM,width=WIDTH,arm_joint_bound=ARM_BOUND,
                         gripper_bound=GRIPPER_BOUND)  # No 14-coordinate policy clamp.
        self.register_buffer('arm_lower',lower[list(ARM_INDICES)].clone())
        self.register_buffer('arm_upper',upper[list(ARM_INDICES)].clone())
        self._joint_lower=lower.tolist();self._joint_upper=upper.tolist()
        self._source_bindings=deepcopy(bindings)

    def config(self):
        return dict(super().config(),kind=KIND,version=1,
            declared_arm_delta_radians=ARM_BOUND,declared_raw_gripper_delta=GRIPPER_BOUND,
            arm_indices=list(ARM_INDICES),arm_lower=self.arm_lower.detach().cpu().tolist(),
            arm_upper=self.arm_upper.detach().cpu().tolist(),
            joint_lower=deepcopy(self._joint_lower),joint_upper=deepcopy(self._joint_upper),
            source_bindings=deepcopy(self._source_bindings),
            inputs='current public2062 + detached native base14 + normalized H32 slot',
            arm_limit_enforcement='clip corrections to actual12-arm URDF headroom; reject out-of-range native arms',
            gripper_limit_enforcement='no policy clamp; unchanged native actuator clip after residual',
            zero_identity='all finite raw14 bits including signed zero and native gripper overshoot; gradient retained',
            base_interface_role='unchanged motor/norm/source limits only; wide bounds declared by this new variant',
            qualification='prospective; no physical success, full-bank admission or fit implied')

    def contract_sha256(self):
        return hashlib.sha256(_canonical(self.config())).hexdigest()

    @classmethod
    def from_config(cls, configuration):
        if not isinstance(configuration,dict):raise ValueError('Require explicit versioned wide-head configuration')
        try:
            head=cls(joint_lower=configuration['joint_lower'],joint_upper=configuration['joint_upper'],
                     **configuration['source_bindings'])
            if _canonical(head.config()) != _canonical(configuration):
                raise ValueError('Configuration differs from this exact prospective wide-head contract')
        except (KeyError,TypeError) as error:
            raise ValueError('Incomplete wide-head configuration/provenance') from error
        return head

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Assignment can install a higher-precision saved buffer after a
        # comparison rounded it to the current dtype. Copy-only loads keep
        # this variant's protected storage/dtype/contract immutable.
        if assign:raise ValueError('Wide-head protected buffers require assign=False')
        # Loading a legacy same-width policy must not silently overwrite this
        # variant's protected bounds/headroom buffers, even with strict=False.
        for name in ('bounds','arm_lower','arm_upper'):
            current=getattr(self,name);saved=state_dict.get(name)
            if (not isinstance(saved,torch.Tensor) or saved.shape!=current.shape
                    or not torch.equal(saved.detach().to(current),current)):
                raise ValueError('Checkpoint bound/headroom mismatch: '+name)
        return super().load_state_dict(state_dict,strict=strict,assign=assign)

    def forward(self, base_actions, public_features):
        # The unchanged generic head validates shape/device/all finite values,
        # detaches incoming tensors and computes its public+base+slot network.
        result=super().forward(base_actions,public_features)
        arms=result.base_actions[...,list(ARM_INDICES)]
        lower=self.arm_lower.to(arms);upper=self.arm_upper.to(arms)
        # A lower-precision command must not round a source limit outward.
        # Requiring exact representability also makes the final clipped bound
        # safe in the output dtype. The approved +/-10 limits are exact.
        if (not torch.equal(lower.to(self.arm_lower),self.arm_lower)
                or not torch.equal(upper.to(self.arm_upper),self.arm_upper)):
            raise ValueError('Command dtype cannot exactly represent source arm limits')
        if ((arms<lower).any() or (arms>upper).any()):
            raise ValueError('Native base ARM exceeds actual source limits; reject rather than break zero parity')
        proposal=_ExactZeroAdd.apply(result.base_actions,result.requested_delta)
        requested_arms=proposal[...,list(ARM_INDICES)]
        bounded_arms=torch.where(requested_arms<lower,lower,
                                torch.where(requested_arms>upper,upper,requested_arms))
        actions=proposal.clone()
        actions[...,list(ARM_INDICES)]=bounded_arms  # Grippers stay unmodified from requested proposal.
        return DecodedActionResidual(result.base_actions,result.requested_delta,
                                     actions-result.base_actions,actions!=proposal,actions)
