"""Prospective wide-head CPU tests; execute only on allocated Linux."""
from copy import deepcopy
from io import BytesIO
import json

import pytest
import torch

from fluffy_bassoon.rl.robotwin_wide_action import WideActionResidual, ARM_INDICES, GRIPPER_INDICES


ARMS=list(ARM_INDICES);GRIPS=list(GRIPPER_INDICES)


def head():
    return WideActionResidual(joint_lower=[-1.]*6+[0.]+[-1.]*6+[0.],
        joint_upper=[1.]*6+[1.]+[1.]*6+[1.],motor_sha256='a'*64,normalization_sha256='b'*64,
        joint_convention_sha256='c'*64,base_interface_sha256='d'*64)


def inputs(dtype=torch.float32):
    base=torch.zeros(2,32,14,dtype=dtype)
    base[...,6]=-.0101426;base[...,13]=1.0090805  # Actual type of native raw gripper overshoot.
    base[:,::2,0]=-0.;base[:,::2,6]=-0.
    base[:,1::2,1]=-1.;base[:,1::2,2]=1.  # Headroom endpoints must retain zero identity.
    return base.requires_grad_(True),torch.randn(2,2062,dtype=dtype,requires_grad=True)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64,torch.float16])
@pytest.mark.parametrize('grad_enabled',[False,True])
def test_zero_head_preserves_every_native_bit_signed_zero_and_gripper_overshoot(dtype,grad_enabled):
    module=head();base,features=inputs(dtype)
    original=base.detach().clone();public=features.detach().clone()
    with torch.set_grad_enabled(grad_enabled):result=module(base,features)
    assert torch.equal(result.actions.contiguous().view(torch.uint8),original.contiguous().view(torch.uint8))
    assert not result.requested_delta.any() and not result.applied_delta.any() and not result.clipped.any()
    assert torch.equal(base.detach().contiguous().view(torch.uint8),original.contiguous().view(torch.uint8))
    assert torch.equal(features.detach(),public)
    if grad_enabled:
        result.actions.float().sum().backward()
        assert module.output.bias.grad.abs().sum()>0 and base.grad is None and features.grad is None


def test_editable_grippers_have_nonzero_gradients_and_no_policy_level_absolute_clip():
    module=head();base,features=inputs()
    with torch.no_grad():module.output.bias[6]=.5;module.output.bias[13]=.7
    result=module(base,features)
    assert (result.requested_delta[...,GRIPS]>0).all()
    assert (result.requested_delta[...,GRIPS]<=torch.tensor(1.1)).all()
    assert (result.actions[...,13]>1).all()  # Native actuator, not this head, applies its old clip.
    assert torch.equal(result.actions[...,GRIPS],result.base_actions[...,GRIPS]+result.requested_delta[...,GRIPS])
    assert not result.clipped[...,GRIPS].any()
    result.actions[...,GRIPS].sum().backward()
    assert (module.output.bias.grad[GRIPS]>0).all() and base.grad is None and features.grad is None


@pytest.mark.parametrize('sign',[-1,1])
def test_two_rad_requested_bounds_are_distinct_from_actual_arm_headroom(sign):
    module=head();base=torch.zeros(1,32,14);features=torch.zeros(1,2062)
    with torch.no_grad():module.output.bias.fill_(sign*10.)
    result=module(base,features)
    assert (result.requested_delta[...,ARMS].abs()<=2.).all()
    torch.testing.assert_close(result.requested_delta[...,ARMS],torch.full((1,32,12),sign*2.))
    assert torch.equal(result.actions[...,ARMS],torch.full((1,32,12),float(sign)))
    assert result.clipped[...,ARMS].all() and not result.clipped[...,GRIPS].any()
    assert (result.actions[...,GRIPS].abs()>.999).all()


@pytest.mark.parametrize('fault',['low_arm','high_arm','nan_arm','inf_grip','wrong_shape','wrong_public'])
def test_invalid_native_or_public_inputs_fail_instead_of_repairing_zero_parity(fault):
    module=head();base,features=inputs();base=base.detach();features=features.detach()
    if fault=='low_arm':base[0,0,0]=-1.01
    if fault=='high_arm':base[0,0,12]=1.01
    if fault=='nan_arm':base[0,0,7]=float('nan')
    if fault=='inf_grip':base[0,0,13]=float('inf')
    if fault=='wrong_shape':base=base[...,:13]
    if fault=='wrong_public':features=features[:,:2061]
    with pytest.raises(ValueError):module(base,features)


def test_checkpoint_configuration_hash_and_policy_roundtrip_are_exact():
    module=head();base,features=inputs()
    with torch.no_grad():module.output.bias.copy_(torch.linspace(-.3,.4,14))
    config=json.loads(json.dumps(module.config()))
    sha=module.contract_sha256();payload=BytesIO()
    torch.save(dict(configuration=config,contract_sha256=sha,policy=module.state_dict()),payload)
    payload.seek(0);saved=torch.load(payload,map_location='cpu',weights_only=True)
    restored=WideActionResidual.from_config(saved['configuration'])
    restored.load_state_dict(saved['policy'])
    assert restored.config()==config and restored.contract_sha256()==saved['contract_sha256']
    with torch.no_grad():
        assert torch.equal(module(base,features).actions,restored(base,features).actions)
    config['source_bindings']['motor_sha256']='e'*64
    assert module.config()['source_bindings']['motor_sha256']=='a'*64
    assert WideActionResidual.from_config(config).contract_sha256()!=sha


@pytest.mark.parametrize('fault',['legacy_kind','legacy_bounds','grip_clamp','missing_provenance','changed_width','changed_limit'])
def test_legacy_or_ambiguous_config_cannot_be_loaded_as_wide_variant(fault):
    config=head().config()
    if fault=='legacy_kind':config['kind']='reactive_decoded_aloha_residual_v1'
    if fault=='legacy_bounds':config['bounds'][0]=.05
    if fault=='grip_clamp':config['command_lower']=[-1.]*14
    if fault=='missing_provenance':config['source_bindings'].pop('base_interface_sha256')
    if fault=='changed_width':config['width']=32
    if fault=='changed_limit':config['arm_lower'][0]=-.5
    with pytest.raises(ValueError):WideActionResidual.from_config(config)


@pytest.mark.parametrize('buffer',['bounds','arm_lower','arm_upper'])
def test_checkpoint_cannot_overwrite_protected_wide_bounds_or_source_headroom(buffer):
    module=head();saved=deepcopy(module.state_dict());saved[buffer][0]+=.01
    before=module.contract_sha256()
    with pytest.raises(ValueError):module.load_state_dict(saved,strict=False)
    assert module.contract_sha256()==before


@pytest.mark.parametrize('fault',['nonfinite_limit','reversed_limit','wrong_limits','invalid_sha'])
def test_constructor_requires_ordered_source_limits_and_explicit_provenance(fault):
    config=head().config();lower=config['joint_lower'];upper=config['joint_upper'];bindings=config['source_bindings']
    if fault=='nonfinite_limit':lower[0]=float('nan')
    if fault=='reversed_limit':upper[0]=lower[0]
    if fault=='wrong_limits':lower.pop()
    if fault=='invalid_sha':bindings['motor_sha256']='unbound'
    with pytest.raises(ValueError):WideActionResidual(joint_lower=lower,joint_upper=upper,**bindings)


def test_committed_prospective_config_constructs_exactly_with_its_declared_hash():
    import hashlib
    from pathlib import Path
    root=Path(__file__).parents[1]/'outputs/archive/robotwin-wide-action-head-20261101-r1'
    config=json.loads((root/'CONFIG.json').read_text());profile=json.loads((root/'PROFILE.json').read_text())
    assert hashlib.sha256((root/'CONFIG.json').read_bytes()).hexdigest()==profile['configuration_file_sha256']
    module=WideActionResidual.from_config(config)
    assert module.config()==config and module.contract_sha256()==profile['expected_head_contract_sha256']
    assert config['declared_arm_delta_radians']==2. and config['declared_raw_gripper_delta']==1.1
    assert profile['native_interface_reference']['legacy_arm_delta']==.05
    assert not any(profile['gates'].values())


@pytest.mark.parametrize('buffer',['bounds','arm_lower','arm_upper'])
@pytest.mark.parametrize('strict',[False,True])
def test_assign_true_cannot_install_changed_high_precision_protected_buffers(buffer,strict):
    module=head();before=deepcopy(module.state_dict());configuration=module.config();sha=module.contract_sha256()
    saved=deepcopy(before);saved[buffer]=saved[buffer].double();saved[buffer][0]+=1e-9
    # This is the review's actual lossy-comparison trap, not merely a large
    # mismatch already rejected by the original protected-buffer regression.
    assert torch.equal(saved[buffer].float(),before[buffer])
    assert not torch.equal(saved[buffer],before[buffer].double())
    saved['hidden.bias']+=.125  # Rejection must happen before ANY parameter copy.
    with pytest.raises(ValueError,match='assign=False'):
        module.load_state_dict(saved,strict=strict,assign=True)
    assert module.config()==configuration and module.contract_sha256()==sha
    after=module.state_dict()
    assert all(after[k].dtype==v.dtype and torch.equal(after[k],v) for k,v in before.items())


@pytest.mark.parametrize('side',['lower','upper'])
@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
def test_inexact_source_limit_cannot_round_outward_in_low_precision_commands(side,dtype):
    prototype=head().config();lower=prototype['joint_lower'];upper=prototype['joint_upper']
    if side=='upper':upper[0]=2.8
    else:lower[0]=-2.8
    module=WideActionResidual(joint_lower=lower,joint_upper=upper,**prototype['source_bindings'])
    base=torch.zeros(1,32,14,dtype=dtype);features=torch.zeros(1,2062,dtype=dtype)
    source=getattr(module,'arm_'+side)[0]
    rounded=source.to(dtype)
    assert rounded.float()!=source  # The +/-1 or approved +/-10 fixtures miss this case.
    # Try both zero-native commands and nonzero outputs: unsafe dtype/source
    # combinations are refused before any native command can be returned.
    for edit in (0.,10.):
        with torch.no_grad():module.output.bias.fill_(edit)
        with pytest.raises(ValueError,match='exactly represent source arm limits'):
            module(base,features)
    if dtype==torch.float16:
        # Float16 2.8 rounds outward to2.80078125. This native arm is actually
        # outside the float32 source bound, even though the old cast test allowed it.
        base[...,0]=rounded
        assert (base[...,0].float()>source).all() if side=='upper' else (base[...,0].float()<source).all()
        with pytest.raises(ValueError,match='exactly represent source arm limits'):
            module(base,features)
    # Float32 commands still use the exact source limits; both the zero head
    # and a saturated edit remain admissible and respect those same limits.
    precise=base.float();precise[...,0]=0.
    for edit in (0.,10.):
        with torch.no_grad():module.output.bias.fill_(edit)
        result=module(precise,features.float())
        assert (result.actions[...,ARMS]>=module.arm_lower).all()
        assert (result.actions[...,ARMS]<=module.arm_upper).all()
