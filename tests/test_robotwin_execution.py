"""Core environment accounting; run on allocated Linux CPU resources."""
import pytest
from types import SimpleNamespace

from fluffy_bassoon.rl.robotwin_execution import (
    chunk_execution, install_accounted_controller, collated_chunk_executions)


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


class Task:
    def __init__(self, result):
        self.result = result
        self.scene = SimpleNamespace(get_timestep=lambda: .004000000189989805)

    def gen_sparse_reward_data(self, actions, *, collect_obs=False):
        raise AssertionError('The class controller must remain untouched')


CONTROLLER = compile('''def gen_sparse_reward_data(self, actions, *, collect_obs=False):
    self.physical_actions = actions
    self.full_observations = collect_obs
    return self.result
''', '<passively measured native boundary>', 'exec')


@pytest.mark.parametrize('collect_obs', [False, True])
def test_instance_accounting_preserves_commands_feedback_and_native_masks(collect_obs):
    info = measured(1, 1, 7)
    if collect_obs:
        info['executed_action_count'] = 2
    result = ([1.], [True], [False], info)
    task = Task(result)
    original = Task.gen_sparse_reward_data
    actions = [[.1] * 14 for _ in range(4)]
    install_accounted_controller(task, CONTROLLER, metadata=True)
    assert task.gen_sparse_reward_data(actions, collect_obs=collect_obs) is result
    assert task.physical_actions is actions and task.full_observations == collect_obs
    assert Task.gen_sparse_reward_data is original
    assert ('executed_action_count' in info) == collect_obs
    assert info['fb2_measured_valid_target_mask'] == (True, True, False, False)
    assert info['fb2_physics_duration_seconds'] == 7 * task.scene.get_timestep()
    assert info['fb2_native_feedback'] == dict(reward=[1.],terminated=[True],truncated=[False])
    result[0][0] = .5
    result[1][0] = False
    assert info['fb2_native_feedback']['reward'] == [1.]
    assert info['fb2_native_feedback']['terminated'] == [True]
    # Rebinding after a reset must use the original class globals each time.
    install_accounted_controller(task, CONTROLLER, metadata=True)
    assert task.gen_sparse_reward_data(actions) is result


def test_instance_accounting_keeps_terminal_noop_duration_zero():
    info = measured(0, 0, 0)
    task = Task(([0.], [True], [False], info))
    install_accounted_controller(task, CONTROLLER, metadata=True)
    task.gen_sparse_reward_data([[0.] * 14 for _ in range(4)])
    assert info['fb2_physics_duration_seconds'] == 0
    assert not any(info['fb2_measured_valid_target_mask'])
    assert 'executed_action_count' not in info


@pytest.mark.parametrize('auto_reset', [False, True])
def test_vector_physics_retains_partial_and_noop_lanes_through_auto_reset(auto_reset):
    packets = []
    for info in (measured(1, 1, 7), measured(0, 0, 0)):
        task = Task(([0.], [True], [False], info))
        install_accounted_controller(task, CONTROLLER, metadata=True)
        task.gen_sparse_reward_data([[0.] * 14 for _ in range(4)])
        packets.append(info)
    collated = {key: [packet[key] for packet in packets] for key in packets[0]}
    # Native eval auto-reset moves all terminal lanes to final_info, including
    # lanes which did not reset. The new reset info must never replace them.
    native = dict(final_info=collated, reset_state_ids=[100, 101]) if auto_reset else collated
    rows = collated_chunk_executions(native, num_envs=2, requested_targets=4)
    assert [row.delivered_targets for row in rows] == [2, 0]
    assert [row.physics_steps for row in rows] == [7, 0]
    assert [row.duration_seconds for row in rows] == [7 * .004000000189989805, 0]
    assert 'valid_action_mask' not in native and 'valid_action_mask' not in collated
    collated['fb2_measured_valid_target_mask'][0] = (True,) * 4
    with pytest.raises(ValueError, match='disagrees'):
        collated_chunk_executions(native, num_envs=2, requested_targets=4)


def test_vector_missing_lane_cannot_be_replaced_by_requested_count():
    with pytest.raises(ValueError, match='incomplete'):
        collated_chunk_executions(dict(fb2_execution=[measured(1, 1, 7)]),
                                  num_envs=2, requested_targets=4)
