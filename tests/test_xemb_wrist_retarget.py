import numpy as np

from fluffy_bassoon.xemb.embodiment import retarget_action_from_motor

LO, HI = np.full(7, -1, np.float32), np.full(7, 1, np.float32)


def _rotated_frame(action, deg):
    """ReGround's released rot_z shift: both command triplets rotate about world z."""
    t = np.radians(deg)
    rot = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1.0]])
    out = np.array(action, dtype=np.float64)
    out[..., 0:3] = out[..., 0:3] @ rot.T
    out[..., 3:6] = out[..., 3:6] @ rot.T
    return out


def test_wrist_rotation_inverts_a_rotated_action_frame():
    proposal = np.random.default_rng(0).uniform(-0.6, 0.6, (5, 7)).astype(np.float32)
    for deg in (90.0, 45.0, -135.0):
        issued = retarget_action_from_motor(proposal, deg, 1.0, 1, LO, HI, rotate_wrist=True)
        np.testing.assert_allclose(_rotated_frame(issued, deg), proposal, atol=1e-6)
        planar = retarget_action_from_motor(proposal, deg, 1.0, 1, LO, HI)
        np.testing.assert_array_equal(planar[:, 3:6], proposal[:, 3:6])
        np.testing.assert_allclose(_rotated_frame(planar, deg)[:, :3], proposal[:, :3], atol=1e-6)
