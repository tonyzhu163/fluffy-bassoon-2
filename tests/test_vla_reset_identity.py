"""Opt-in source40 physical identity and one-LSB camera precision."""
from copy import deepcopy
import hashlib
import numpy as np
import pytest
from fluffy_bassoon.meta_rl.vla_reset_identity import RESET_IDENTITY, comparison
from fluffy_bassoon.meta_rl.vla_source40_reset_probe import compare_resets
from fluffy_bassoon.meta_rl.vla_rl2 import collect_rl2_trial
from test_vla_rl2 import small_policy, SIM, features, fake_episode


def reset():
    actual = np.arange(92, dtype=np.float64)
    return dict(initial_state_vector_sha256='same-nominal', initial_simulator_state=actual,
        initial_simulator_state_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
        initial_observation_sha256='recorded-hash', observation=dict(image=np.zeros((4, 4, 3), np.uint8),
            wrist=np.full((4, 4, 3), 64, np.uint8), state=np.zeros(8, np.float32)))


def test_only_camera_rounding_is_admitted_and_raw_values_are_unchanged():
    first, second = reset(), reset()
    second['observation']['wrist'][1, 2, :2] += 1
    before = second['observation']['wrist'].copy()
    metrics = comparison(first, second, camera_max_lsb=1)
    assert metrics['allowed'] and metrics['exact_actual_simulator_state'] and metrics['exact_proprioception']
    assert metrics['cameras']['wrist']['maximum_absolute_lsb'] == 1
    assert metrics['cameras']['wrist']['changed_channels'] == 2
    np.testing.assert_array_equal(second['observation']['wrist'], before)
    assert compare_resets([first, second, first, second], camera_max_lsb=1)['exact_actual_simulator_states']
    with pytest.raises(RuntimeError, match='exact public observation'):
        compare_resets([first, second, first, second])


@pytest.mark.parametrize('kind', ['physical', 'proprio', 'nominal', 'camera'])
def test_physical_proprio_or_larger_camera_changes_still_fail(kind):
    first, second = reset(), reset()
    if kind == 'physical':
        second['initial_simulator_state'][0] += 1e-12
        second['initial_simulator_state_sha256'] = hashlib.sha256(second['initial_simulator_state'].tobytes()).hexdigest()
    elif kind == 'proprio':
        second['observation']['state'][0] = np.float32(1e-8)
    elif kind == 'nominal':
        second['initial_state_vector_sha256'] = 'other'
    else:
        second['observation']['wrist'][1, 2, 0] += 2
    assert not comparison(first, second, camera_max_lsb=1)['allowed']


def test_generic_rl2_remains_byte_strict_and_source40_profile_is_explicit():
    count = 0
    def runner(*args, **kwargs):
        nonlocal count
        rows, summary = fake_episode(*args, **kwargs)
        value = reset()
        value['observation']['wrist'][1, 2, 0] += count % 2
        rows[0]['motor_observation'] = value['observation']
        summary.update({key: value[key] for key in ('initial_state_vector_sha256', 'initial_simulator_state', 'initial_simulator_state_sha256')})
        summary['initial_observation_sha256'] = str(count)
        count += 1
        return rows, summary
    arguments = dict(initial_state=1, scene_seed=3, attempts=2, chunk=1, episode_runner=runner)
    with pytest.raises(RuntimeError, match='changed its initial observation'):
        collect_rl2_trial(small_policy(), SIM, None, features, **arguments)
    count = 0
    _, receipt = collect_rl2_trial(small_policy(), SIM, None, features, reset_identity=RESET_IDENTITY, **arguments)
    assert receipt['attempts'][1]['reset_identity_metrics']['cameras']['wrist']['maximum_absolute_lsb'] == 1
