"""Core environment accounting; run on allocated Linux CPU resources."""
import pytest

from fluffy_bassoon.rl.robotwin_execution import chunk_execution


def measured(completed, partial, physics, *, requested=4):
    return dict(fb2_execution=dict(requested_targets=requested,
        completed_targets=completed, partial_targets=partial,
        delivered_targets=completed + partial, physics_steps=physics))


def test_terminal_partial_target_has_a_mask_and_actual_duration():
    # One target completed and the next partly moved before success. Neither
    # the four proposed targets nor the seven physics ticks is the action count.
    info = measured(1, 1, 7)
    result = chunk_execution(info, requested_targets=4, timestep_seconds=1 / 250)
    assert result.valid_target_mask == (True, True, False, False)
    assert result.duration_seconds == pytest.approx(.028)
    assert 'executed_action_count' not in info  # Do not alter native RLinf masks.


def test_already_finished_chunk_is_a_zero_duration_noop():
    result = chunk_execution(measured(0, 0, 0), requested_targets=4, timestep_seconds=.004)
    assert result.valid_target_mask == (False,) * 4
    assert result.duration_seconds == 0


def test_native_full_observation_count_must_agree():
    info = measured(4, 0, 3)
    info['executed_action_count'] = 4
    assert chunk_execution(info, requested_targets=4, timestep_seconds=.004).delivered_targets == 4
    info['executed_action_count'] = 3
    with pytest.raises(ValueError, match='disagrees'):
        chunk_execution(info, requested_targets=4, timestep_seconds=.004)


@pytest.mark.parametrize('info', [dict(executed_action_count=4), {},
    measured(4, 0, 0), measured(0, 1, 0), measured(0, 0, 1),
    measured(3, 2, 8), measured(5, 0, 8), measured(True, 0, 8)])
def test_missing_or_impossible_execution_is_never_used_as_a_transition(info):
    with pytest.raises(ValueError):
        chunk_execution(info, requested_targets=4, timestep_seconds=.004)


@pytest.mark.parametrize('dt', [0, -1, True, float('nan'), float('inf')])
def test_duration_uses_a_real_scene_timestep(dt):
    with pytest.raises(ValueError):
        chunk_execution(measured(4, 0, 20), requested_targets=4, timestep_seconds=dt)
