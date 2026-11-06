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

from fluffy_bassoon.rl.robotwin_execution import chunk_execution


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
