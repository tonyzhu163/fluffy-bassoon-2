"""Numerical reference for omitted-joint contamination; no simulator or CUDA."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import mplib
import numpy as np
import torch
from envs.robot.planner import MplibWrapperPlanner, MplibPlanner

from motus_planner_full_state import full_qpos, plan_path, plan_batch

ROOT = Path(__file__).resolve().parent
BASE = Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1')


def main():
    assert os.uname().sysname == 'Linux' and os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert not torch.cuda.is_initialized()
    code = ROOT.parents[1]; source = os.environ['TASK_SOURCE']
    assert subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip() == source
    out = BASE/'cpumotusplanner01'; out.mkdir()
    rows = []
    for side, arm in (('left', list(range(6))), ('right', list(range(8, 14)))):
        names = [f'joint-{i}' for i in range(16)]
        current = np.linspace(-.9, .9, 16)+.000003
        original = current.copy()
        stale = np.full(16, -1.7)
        articulation = SimpleNamespace(get_move_group_qpos_dim=lambda:6, get_qpos=lambda:stale.copy())
        native_planner = SimpleNamespace(robot=articulation, joint_limits=np.zeros((16, 2)))
        owner = object.__new__(MplibWrapperPlanner)
        owner.all_joints = names
        owner.active_joints_name = [names[i] for i in arm]
        # Use the actual installed SDK's padding function, not a rewritten reference.
        captures = []
        def solve(instance, *, now_qpos, **options):
            padded = mplib.Planner.pad_move_group_qpos(native_planner, now_qpos)
            captures.append(dict(input=now_qpos.copy(), padded=padded.copy()))
            return dict(status='Success', position=np.zeros((2, 6)), velocity=np.zeros((2, 6)))
        target = SimpleNamespace(p=[0., 0., .8], q=[1., 0., 0., 0.])
        with patch.object(MplibPlanner, 'plan_path', solve):
            MplibWrapperPlanner.plan_path(owner, current, target)
            old = captures.pop()
            plan_path(owner, current, target)
            new = captures.pop()
            batch = plan_batch(owner, current, [target, target])
            assert len(captures) == 2 and batch['status'].tolist() == ['Success', 'Success']
            for value in captures: np.testing.assert_array_equal(value['padded'], new['padded'])
        expected = original.copy(); expected[arm] = np.round(expected[arm], 5)
        np.testing.assert_array_equal(current, original)
        np.testing.assert_array_equal(new['input'], expected)
        np.testing.assert_array_equal(new['padded'], expected)
        assert len(old['input']) == 6 and len(new['input']) == 16
        assert not np.array_equal(old['padded'], expected)
        remaining = sorted(set(range(16))-set(arm))
        np.testing.assert_array_equal(new['padded'][remaining], original[remaining])
        rows.append(dict(side=side, upstream_input_DOFs=6, full_input_DOFs=16,
            upstream_reconstructed_state_differs_from_measured=True,
            full_input_reconstructed_state_exact=True, non_arm_measured_DOFs_exact=True,
            source_state_unmutated=True, batch_uses_same_full_state=True))
    assert not torch.cuda.is_initialized()
    report = dict(status='passed_native_MPLib_padding_and_wrapper_full_state_CPU_reference',
        source_commit=source, cases=rows, GPU_seconds=0, actual_simulators_constructed=False,
        actual_planner_solutions_computed=False, GPU_positive_controls_still_required=True,
        old_scene48_failures_causally_resolved=False,
        native_padding_source_sha256=hashlib.sha256(inspect.getsource(mplib.Planner.pad_move_group_qpos).encode()).hexdigest(),
        native_wrapper_source_sha256=hashlib.sha256(inspect.getsource(MplibWrapperPlanner).encode()).hexdigest())
    path = out/'RESULT.json'; path.write_text(json.dumps(report, indent=2)+'\n')
    print('PLANNER_CPU_REFERENCE', str(path), hashlib.sha256(path.read_bytes()).hexdigest(), flush=True)


if __name__ == '__main__': main()
