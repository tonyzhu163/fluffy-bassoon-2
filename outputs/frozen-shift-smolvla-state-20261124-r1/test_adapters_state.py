import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "frozen-shift-smolvla-20261124-r1")]
from adapters_state import SourceBranch, TargetBranch, other_branch  # noqa: E402

KINOVA = np.array([-0.21, 0.0, 1.17, -3.18, -0.05, 0.14, 0.04, -0.04], np.float32)
PANDA = np.array([-0.21, 0.0, 1.17, 3.14, 0.0, -0.09, 0.04, -0.04], np.float32)


def matrix(r):
    theta = np.linalg.norm(r)
    k = np.cross(np.eye(3), r / theta)
    return np.eye(3) + np.sin(theta) * k + (1 - np.cos(theta)) * k @ k


def same_rotation(a, b):
    return np.allclose(matrix(a[3:6].astype(np.float64)), matrix(b[3:6].astype(np.float64)), atol=1e-5)


def test_branch_change_keeps_rotation_and_other_channels():
    for state in (KINOVA, PANDA):
        moved = other_branch(state)
        assert same_rotation(state, moved) and np.sign(moved[3]) == -np.sign(state[3])
        assert np.array_equal(moved[[0, 1, 2, 6, 7]], state[[0, 1, 2, 6, 7]])


def test_each_adapter_acts_on_one_branch_only():
    assert SourceBranch().state(KINOVA, None, None)[3] > 3.0
    assert np.array_equal(SourceBranch().state(PANDA, None, None), PANDA)
    assert TargetBranch().state(PANDA, None, None)[3] < -3.0
    assert np.array_equal(TargetBranch().state(KINOVA, None, None), KINOVA)
