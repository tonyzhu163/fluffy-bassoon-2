"""Run only in the owner's allocated Linux CPU qualification."""
from dataclasses import replace
import hashlib

import numpy as np
import pytest
import torch

from fluffy_bassoon.rl.steering_actuator_condition import (
    ActuatorCalibration, CalibrationDistribution, calibration_pair,
    verify_inverse_command_cancellation, verify_same_initial_public_observation,
)


def make_pair(arm="action"):
    basis = np.zeros((4, 14), np.float32)
    basis[0, [0, 1, 7, 8]] = [1., -.5, .25, -1.]
    basis[1, 2], basis[2, 9], basis[3, 5] = 1., 1., 1.
    sha = hashlib.sha256(str(basis.dtype).encode()+str(basis.shape).encode()+basis.tobytes()).hexdigest()
    return basis, calibration_pair(physical_basis=basis, physical_basis_sha256=sha,
        arm_joint_bound=.05, selected_arm=arm, command_lower=[-2.]*14,
        command_upper=[2.]*14, range_receipt_sha256="a"*64,
        motor_sha256="b"*64, mapping_sha256="c"*64)


@pytest.mark.parametrize("arm", ["action", "decoded_ALOHA_joint_coefficients"])
@pytest.mark.parametrize("sign", [-1, 1])
def test_actual_constant_mode_inverse_is_supported_and_grippers_exact(arm, sign):
    basis, specs = make_pair(arm)
    spec = next(s for s in specs if s.sign == sign)
    coefficients = np.asarray(spec.same_mapping_inverse_coefficients)
    base = np.full((1, 32, 14), .125, np.float32)
    base[..., 6] = -0.
    base[..., 13] = .75
    delta = (.05*np.tanh(coefficients[0]*basis[0])).astype(np.float32)
    requested = np.where(delta == 0, base, base+delta)
    final, audit = ActuatorCalibration(spec)(base, requested)
    proof = verify_inverse_command_cancellation(base, final, audit["clipped"])
    assert proof["passed"] and not proof["bitwise_physical_trajectory_identity_claim"]
    assert np.array_equal(np.ascontiguousarray(final[..., [6, 13]]).view(np.uint8),
                          np.ascontiguousarray(requested[..., [6, 13]]).view(np.uint8))
    assert np.array_equal(audit["public_request"], requested)
    assert abs(spec.bias_radians[0]) < .02


@pytest.mark.parametrize("arm", ["noise", "kv", "noise_coefficients", "native_KV_coefficients"])
def test_external_command_inverse_is_never_a_kv_or_noise_coefficient_label(arm):
    _, specs = make_pair(arm)
    assert all(s.same_mapping_inverse_coefficients is None for s in specs)


def test_clipping_is_audited_and_invalidates_inverse_proof():
    _, specs = make_pair()
    requested = np.zeros((1, 32, 14), np.float32)
    requested[..., 0] = 1.995
    final, audit = ActuatorCalibration(specs[1])(requested, requested)
    assert audit["clipped"][..., 0].all() and np.all(final[..., 0] == 2.)
    with pytest.raises(ValueError, match="no clipping"):
        verify_inverse_command_cancellation(requested, final, audit["clipped"])


def test_distribution_has_one_static_contract_and_explicit_boundary_selection():
    _, specs = make_pair()
    distribution = CalibrationDistribution(specs)
    before = distribution.contract()
    commands = np.zeros((1, 32, 14), np.float32)
    with pytest.raises(RuntimeError, match="persistent"):
        distribution(commands, commands)
    distribution.select(-1)
    minus, _ = distribution(commands, commands)
    distribution.select(1)
    plus, _ = distribution(commands, commands)
    np.testing.assert_array_equal(minus, -plus)
    assert distribution.contract() == before


def test_torch_callback_preserves_input_layout_device_dtype_and_no_mutation():
    _, specs = make_pair()
    base = torch.zeros((1, 32, 14), dtype=torch.float32)
    copy = base.clone()
    final, audit = ActuatorCalibration(specs[0])(base, base)
    assert final.shape == base.shape and final.dtype == base.dtype and final.device == base.device
    assert torch.equal(base, copy) and audit["audit_only"]
    assert torch.equal(final[..., [6, 13]], base[..., [6, 13]])


def test_reject_outside_range_and_nonzero_gripper_condition():
    _, specs = make_pair()
    bad = np.zeros((1, 32, 14), np.float32)
    bad[..., 2] = 3.
    with pytest.raises(ValueError, match="receipt-bound"):
        ActuatorCalibration(specs[0])(bad, bad)
    bias = list(specs[0].bias_radians)
    bias[6] = .001
    with pytest.raises(ValueError, match="zero grippers"):
        replace(specs[0], bias_radians=tuple(bias))


def test_reset_parity_uses_actual_public_hashes_and_accepted_seed():
    record = dict(task="place_phone_stand", requested_seed=10, actual_seed=12,
                  observation_sha256="a"*64, features_sha256="b"*64)
    assert verify_same_initial_public_observation(record, dict(record))["passed"]
    for key, value in (("actual_seed", 13), ("features_sha256", "c"*64)):
        with pytest.raises(ValueError, match="accepted initial"):
            verify_same_initial_public_observation(record, dict(record, **{key: value}))


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("sign", [-1, 1])
def test_actual_native_out_of_range_raw_grippers_stay_bitwise_for_native_clamp(backend, sign):
    _, specs = make_pair()
    spec = next(s for s in specs if s.sign == sign)
    lower, upper = list(spec.command_lower), list(spec.command_upper)
    for i in (6, 13):
        lower[i], upper[i] = 0., 1.
    calibration = ActuatorCalibration(replace(spec, command_lower=tuple(lower), command_upper=tuple(upper)))
    commands = np.zeros((1, 32, 14), np.float32)
    # Actual saved A1 native ranges: raw decoder requests precede the native
    # normalized-gripper clamp. They are legitimate unchanged input commands.
    commands[..., 6] = np.linspace(-.0101426, 1.0090805, 32)
    commands[..., 13] = np.linspace(1.0086253, -.0069030, 32)
    given = commands if backend == "numpy" else torch.from_numpy(commands.copy())
    final, audit = calibration(given, given)
    final = final if backend == "numpy" else final.cpu().numpy()
    np.testing.assert_array_equal(np.ascontiguousarray(final[..., [6, 13]]).view(np.uint8),
                                  np.ascontiguousarray(commands[..., [6, 13]]).view(np.uint8))
    assert np.any(final[..., 6] < 0) and np.any(final[..., 13] > 1)
    assert not audit["clipped"][..., [6, 13]].any()
    request = calibration.inverse_request(commands)
    restored, inverse_audit = calibration(request, request)
    assert verify_inverse_command_cancellation(commands, restored, inverse_audit["clipped"])["passed"]
