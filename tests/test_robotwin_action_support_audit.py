"""Analytic bounds/mask cases; run only on allocated Linux CPU."""
import numpy as np
import pytest

from fluffy_bassoon.meta_rl.robotwin_action_support_audit import support_bounds, combine, physical_plan


def inputs():
    base = np.zeros((32, 14), np.float32)
    return base, base.copy(), np.ones(32, bool), np.ones(32, bool), [-10.] * 14, [10.] * 14


def test_analytic_arm_and_gripper_floor_without_label_clipping():
    base, target, sub, valid, lower, upper = inputs()
    target[:, 0] = 3.; target[:, 6] = .5
    before = target.copy()
    report = support_bounds(base, target, sub, valid, lower, upper, arm_delta_bound=2.)['submitted_plan']
    assert report['squared_error_sum14'][0] == 32.
    assert report['squared_error_sum14'][6] == 8.
    assert report['MSE_floor_physical14'] == pytest.approx(40. / (32 * 14))
    assert report['MSE_floor_arms12'] == pytest.approx(1. / 12)
    assert report['MSE_floor_grippers2'] == .125
    assert np.array_equal(before, target)


def test_joint_headroom_and_raw_gripper_overshoot_are_separate():
    base, target, sub, valid, lower, upper = inputs()
    base[:, 0] = 9.5; target[:, 0] = 10.5
    base[:, 6] = target[:, 6] = 1.01
    report = support_bounds(base, target, sub, valid, lower, upper, arm_delta_bound=2.)['submitted_plan']
    assert report['squared_error_sum14'][0] == 8.
    assert report['squared_error_sum14'][6] == 0.
    base[:, 0] = 10.1
    with pytest.raises(ValueError, match='native ARM'):
        support_bounds(base, target, sub, valid, lower, upper, arm_delta_bound=2.)


def test_joint_endpoint_admission_matches_actual_float32_head_buffers():
    base, target, sub, valid, lower, upper = inputs()
    upper[0] = .1
    base[:, 0] = np.float32(.1)  # Actual allowed Float32 endpoint is above Python .1.
    target[:, 0] = base[:, 0]
    report = support_bounds(base, target, sub, valid, lower, upper, arm_delta_bound=.05)
    assert report['submitted_plan']['MSE_floor_physical14'] == 0.
    # Actual Float32 addition/subtraction can round beyond base +/- radius
    # computed in Float64. These native endpoints must remain inside support.
    lower[0], upper[0] = -10., 10.
    base[:, 0] = 1.
    target[:, 0] = base[:, 0] + np.float32(.05)
    report = support_bounds(base, target, sub, valid, lower, upper, arm_delta_bound=.05)
    assert report['submitted_plan']['MSE_floor_physical14'] == 0.
    base[:, 0] = -1.
    target[:, 0] = base[:, 0] - np.float32(.05)
    report = support_bounds(base, target, sub, valid, lower, upper, arm_delta_bound=.05)
    assert report['submitted_plan']['MSE_floor_physical14'] == 0.
    lower[0] = -.1
    base[:, 0] = target[:, 0] = np.float32(-.1)
    report = support_bounds(base, target, sub, valid, lower, upper, arm_delta_bound=.05)
    assert report['submitted_plan']['MSE_floor_physical14'] == 0.


def test_submitted_terminal_tail_is_supervised_but_not_called_executed():
    base, target, sub, valid, lower, upper = inputs()
    valid[1:] = False; target[1:, 13] = 1.
    report = support_bounds(base, target, sub, valid, lower, upper, arm_delta_bound=2.)
    assert report['submitted_plan']['squared_error_sum14'][13] == 31.
    assert report['native_valid_prefix']['squared_error_sum14'][13] == 0.


def test_unsubmitted_natural_horizon_tail_is_excluded():
    base, target, sub, valid, lower, upper = inputs()
    sub[16:] = valid[16:] = False; target[16:, :] = 100.
    report = support_bounds(base, target, sub, valid, lower, upper, arm_delta_bound=2.)
    assert report['submitted_plan']['slots'] == 16
    assert report['submitted_plan']['MSE_floor_physical14'] == 0.


@pytest.mark.parametrize('change', ['nonprefix', 'outside', 'nonfinite', 'shape', 'empty'])
def test_invalid_capture_rejected(change):
    base, target, sub, valid, lower, upper = inputs()
    if change == 'nonprefix': valid[1] = False
    elif change == 'outside': sub[16:] = False
    elif change == 'nonfinite': base[0, 0] = np.nan
    elif change == 'shape': target = target[:31]
    elif change == 'empty': valid[:] = False
    with pytest.raises(ValueError): support_bounds(base, target, sub, valid, lower, upper, arm_delta_bound=2.)


def test_aggregation_weights_label_slots_and_keeps_physical_groups():
    base, target, sub, valid, lower, upper = inputs(); target[:, 6] = 1.
    first = support_bounds(base, target, sub, valid, lower, upper, arm_delta_bound=2.)
    sub[1:] = valid[1:] = False; target[:] = 0.
    second = support_bounds(base, target, sub, valid, lower, upper, arm_delta_bound=2.)
    report = combine([first, second], 'submitted_plan')
    assert report['slots'] == 33
    assert report['MSE_floor_physical14'] == pytest.approx(32. / (33 * 14))
    assert report['MSE_floor_arms12'] == 0.


def test_plan_uses_captured_numeric_array_reference_and_exact_batch_shape():
    value = np.zeros((1, 32, 14), np.float32)
    assert physical_plan({'recorded/base': value}, {'base_actions': {'numeric_array': 'recorded/base'}}, 'base_actions').shape == (32, 14)
    with pytest.raises(ValueError): physical_plan({'bad': value[0]}, {'base_actions': {'numeric_array': 'bad'}}, 'base_actions')


def test_actual_small_bound_and_explicit_wider_counterfactual_have_same_gripper_floor():
    base, target, sub, valid, lower, upper = inputs()
    target[:, 0] = .2; target[:, 6] = .7
    actual = support_bounds(base, target, sub, valid, lower, upper, arm_delta_bound=.05)['submitted_plan']
    wider = support_bounds(base, target, sub, valid, lower, upper, arm_delta_bound=2.)['submitted_plan']
    assert actual['MSE_floor_arms12'] == pytest.approx(.15 ** 2 / 12)
    assert wider['MSE_floor_arms12'] == 0.
    assert actual['MSE_floor_grippers2'] == wider['MSE_floor_grippers2']
