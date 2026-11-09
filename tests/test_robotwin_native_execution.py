"""Numerical controller qualification against the unmodified pinned source.

Run on allocated Linux with ROBOTWIN_REFERENCE_ROOT set to the unmodified
dca9ec68 checkout and ROBOTWIN_ACCOUNTING_ROOT to a fresh patched checkout.
Only the actual native method is loaded; SAPIEN is not needed for CPU checks.
"""
import ast
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fluffy_bassoon.robotwin.execution import chunk_execution


@pytest.fixture(scope='module')
def controllers():
    functions = []
    for variable in ('ROBOTWIN_REFERENCE_ROOT', 'ROBOTWIN_ACCOUNTING_ROOT'):
        root = os.environ.get(variable)
        if root is None:
            pytest.skip(f'Allocated native controller qualification needs {variable}')
        path = Path(root) / 'envs/_base_task.py'
        module = ast.parse(path.read_text())
        methods = [node for node in ast.walk(module)
                   if isinstance(node, ast.FunctionDef) and node.name == 'gen_sparse_reward_data']
        assert len(methods) == 1
        namespace = dict(np=np)
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), 'exec'), namespace)
        functions.append(namespace['gen_sparse_reward_data'])
    return functions


class Task:
    """Deterministic planners with different left/right durations and real loop traces."""
    def __init__(self, success_step, *, already_success=False, at_limit=False, failed_planner=False):
        self.events = []
        self.steps = 0
        self.take_action_cnt = 100 if at_limit else 0
        self.step_lim = 100
        self.render_freq = 0
        self.eval_success = already_success
        self.success_step = success_step
        self.scene = SimpleNamespace(step=self.physics_step)
        self.robot = SimpleNamespace(
            get_left_arm_jointState=lambda: [0.] * 7,
            get_right_arm_jointState=lambda: [0.] * 7,
            left_mplib_planner=SimpleNamespace(TOPP=self.planner(8, failed_planner)),
            right_mplib_planner=SimpleNamespace(TOPP=self.planner(12, False)),
            set_arm_joints=lambda position, velocity, side: self.events.append(
                ('arm', side, tuple(position), tuple(velocity))),
            set_gripper=lambda value, side: self.events.append(('gripper', side, float(value))))

    @staticmethod
    def planner(length, fail):
        def topp(path, timestep, verbose):
            if fail:
                raise RuntimeError('Deterministic planner failure')
            position = np.arange(length * 6).reshape(length, 6) / 100
            return None, position, np.zeros_like(position), None, length * timestep
        return topp

    def physics_step(self):
        self.steps += 1
        self.events.append(('physics', self.steps))

    def _update_render(self):
        self.events.append(('render', self.steps))

    def get_obs(self):
        return dict(physics=self.steps)

    def check_success(self):
        return self.success_step is not None and self.steps >= self.success_step

    @staticmethod
    def compress_path(path):
        return path

    @staticmethod
    def downsample_trajectory(position, velocity):
        return position, velocity


@pytest.mark.parametrize('success_step', [None, 1, 2, 3, 4, 6, 11, 12, 50])
@pytest.mark.parametrize('initial', ['normal', 'success', 'limit', 'planner_failure'])
def test_passive_fast_and_full_paths_preserve_native_physics_and_results(controllers, success_step, initial):
    original, patched = controllers
    conditions = dict(already_success=initial == 'success', at_limit=initial == 'limit',
                      failed_planner=initial == 'planner_failure')
    actions = np.arange(4 * 14).reshape(4, 14) / 100
    reference_task, fast_task, full_task = [Task(success_step, **conditions) for _ in range(3)]
    reference = original(reference_task, actions, collect_obs=True)
    fast = patched(fast_task, actions, collect_obs=False)
    full = patched(full_task, actions, collect_obs=True)
    assert reference_task.events == fast_task.events == full_task.events
    assert reference_task.take_action_cnt == fast_task.take_action_cnt == full_task.take_action_cnt
    for index in (0, 1, 2):
        np.testing.assert_array_equal(reference[index], fast[index])
        np.testing.assert_array_equal(reference[index], full[index])
    assert reference[3] == {k: v for k, v in full[3].items() if k != 'fb2_execution'}
    assert reference[4] == full[4]
    assert 'executed_action_count' not in fast[3]
    assert fast[3]['success'] == reference[3]['success']
    execution = chunk_execution(fast[3], requested_targets=4, timestep_seconds=.004)
    assert execution.delivered_targets == reference[3]['executed_action_count']
    assert execution.physics_steps == fast_task.steps
    assert full[3]['fb2_execution'] == fast[3]['fb2_execution']


def test_waypoint_controller_synchronizes_waypoints_and_recovers_topp_failures(tmp_path):
    from fluffy_bassoon.robotwin.batch import accounted_controller_source
    from fluffy_bassoon.robotwin.execution import load_accounted_controller
    root = os.environ.get('ROBOTWIN_REFERENCE_ROOT')
    if root is None:
        fallback = Path(__file__).resolve().parents[1] / (
            'outputs/robotwin-sidney-deployment-diagnostic-20261106-r1/upstream/RoboTwin')
        if fallback.is_dir():
            root = str(fallback)
        else:
            pytest.skip('Waypoint controller test needs ROBOTWIN_REFERENCE_ROOT')
    meta = accounted_controller_source(root, tmp_path / 'waypoint', waypoint_controller=True)
    assert meta['waypoint_controller'] is True
    code = load_accounted_controller(meta['controller_source'], meta['controller_sha256'])
    namespace = dict(np=np)
    exec(code, namespace)
    waypoint_fn = namespace['gen_sparse_reward_data']

    # Normal 4-waypoint chunk: each waypoint plans a 2-point TOPP path, never
    # calls compress_path/downsample_trajectory, and skips intermediate renders
    # when collect_obs=False.
    task = Task(success_step=None, failed_planner=False)
    actions = np.arange(1, 4 * 14 + 1, dtype=np.float64).reshape(4, 14) / 100
    reward, term, trunc, info = waypoint_fn(task, actions, collect_obs=False)
    exec_info = chunk_execution(info, requested_targets=4, timestep_seconds=1 / 250)
    assert exec_info.completed_targets == 4 and exec_info.delivered_targets == 4
    assert exec_info.partial_targets == 0 and exec_info.physics_steps == 4 * 12
    # Render is called only at the start and end of the chunk when collect_obs=False.
    renders = [event for event in task.events if event[0] == 'render']
    assert len(renders) == 2

    # Non-stationary planner failure: fallback linear interpolation still drives
    # the left arm joints instead of freezing the arm across the chunk.
    failed_task = Task(success_step=None, failed_planner=True)
    waypoint_fn(failed_task, actions, collect_obs=False)
    left_arm_moves = [event for event in failed_task.events if event[:2] == ('arm', 'left')]
    assert len(left_arm_moves) == 4 * 50
    np.testing.assert_allclose(left_arm_moves[-1][2], actions[-1, :6])
