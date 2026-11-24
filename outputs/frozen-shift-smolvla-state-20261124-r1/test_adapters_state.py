import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "frozen-shift-smolvla-20261124-r1")]
from adapters_state import SourceBranch, TargetBranch, other_branch  # noqa: E402

KINOVA = np.array([-0.21, 0.0, 1.17, -3.18, -0.05, 0.14, 0.04, -0.04], np.float32)
PANDA = np.array([-0.21, 0.0, 1.17, 3.14, 0.0, -0.09, 0.04, -0.04], np.float32)


def same_rotation(a, b):
    return (Rotation.from_rotvec(a[3:6]) * Rotation.from_rotvec(b[3:6]).inv()).magnitude() < 1e-5


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
