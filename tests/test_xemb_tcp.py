"""Unit tests for physical MuJoCo TCP offsets, orientation tracking, and compensation controller."""

import math
import numpy as np


def test_quat_and_world_tcp_offset_tracking():
    from fluffy_bassoon.xemb.tcp import quat_xyzw_to_mat, world_tcp_offset_at_quat

    q0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    np.testing.assert_allclose(quat_xyzw_to_mat(q0), np.eye(3), atol=1e-12)

    d0 = np.array([0.035, 0.0, 0.0], dtype=np.float64)
    np.testing.assert_allclose(world_tcp_offset_at_quat(q0, q0, d0), d0, atol=1e-12)

    # Rotate +90 deg around Z: [x, y, z, w] = [0, 0, sin(45), cos(45)]
    s = math.sqrt(0.5)
    q90 = np.array([0.0, 0.0, s, s], dtype=np.float64)
    d90 = world_tcp_offset_at_quat(q0, q90, d0)
    np.testing.assert_allclose(d90, np.array([0.0, 0.035, 0.0]), atol=1e-12)

    # Rotate 180 deg around Z: [0, 0, 1, 0]
    q180 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    d180 = world_tcp_offset_at_quat(q0, q180, d0)
    np.testing.assert_allclose(d180, np.array([-0.035, 0.0, 0.0]), atol=1e-12)

    # Non-identity reset quaternion rotated by +90 deg around Z
    q_init = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # 180 deg around X
    # Right-multiplying body rotation or world rotation: R_t = Rz(90) @ R_0
    r_z90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    # Construct q_t = q_z90 * q_init = [0, 0, s, s] * [1, 0, 0, 0] = [s, s, 0, 0]
    q_t = np.array([s, s, 0.0, 0.0], dtype=np.float64)
    np.testing.assert_allclose(quat_xyzw_to_mat(q_t), r_z90 @ quat_xyzw_to_mat(q_init), atol=1e-12)
    np.testing.assert_allclose(world_tcp_offset_at_quat(q_init, q_t, d0), np.array([0.0, 0.035, 0.0]), atol=1e-12)


def test_make_tcp_catalogue_and_compositional_split():
    from fluffy_bassoon.xemb.tcp import make_tcp_catalogue

    train_specs, eval_specs = make_tcp_catalogue(0.035)
    assert len(train_specs) == 5
    assert len(eval_specs) == 7

    # Zero diagonals in training.
    for spec in train_specs:
        assert spec.dx_m == 0.0 or spec.dy_m == 0.0

    comp = [s for s in eval_specs if s.regime == "compositional_ood"]
    assert len(comp) == 2
    for spec in comp:
        assert spec.dx_m != 0.0 and spec.dy_m != 0.0
        assert abs(math.hypot(spec.dx_m, spec.dy_m) - 0.035) < 2e-4


def test_tcp_controller_passthrough_sequence_and_rearm():
    from fluffy_bassoon.xemb.tcp import TcpCompensatedController

    q0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    nom_ctrl = TcpCompensatedController(q0, 0.0, 0.0, compensate_release=True)
    assert not nom_ctrl.active
    prop = np.array([0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 0.8], dtype=np.float32)
    out = nom_ctrl.next_action(np.zeros(3), q0, np.array([0.04, -0.04]), prop)
    np.testing.assert_array_equal(out, prop)
    assert nom_ctrl.needs_motor_proposal

    # Active controller under 90 deg wrist rotation: local +0.035 X becomes +0.035 Y in world!
    s = math.sqrt(0.5)
    q90 = np.array([0.0, 0.0, s, s], dtype=np.float64)
    ctrl = TcpCompensatedController(q0, 0.035, 0.0, compensate_release=True)
    pos = np.array([0.10, 0.20, 0.90], dtype=np.float64)
    jaw_open = np.array([0.04, -0.04], dtype=np.float32)
    jaw_closed = np.array([0.01, -0.01], dtype=np.float32)

    # Approach steps with negative grip don't trigger.
    approach = np.array([0.1, 0.0, -0.1, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    out0 = ctrl.next_action(pos, q90, jaw_open, approach)
    np.testing.assert_array_equal(out0, approach)
    assert ctrl.phase == "nominal"

    # Two consecutive positive grip commands trigger maneuver.
    close_prop = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    out1 = ctrl.next_action(pos, q90, jaw_open, close_prop)
    assert ctrl.phase == "nominal"
    np.testing.assert_array_equal(out1, close_prop)

    # On 2nd close command, transitions immediately into lift_clear.
    phases = []
    act = ctrl.next_action(pos, q90, jaw_open, close_prop)
    phases.append(ctrl.phase)
    assert ctrl.grasp_trigger_count == 1
    assert ctrl.phase == "lift_clear"
    assert not ctrl.needs_motor_proposal

    # Simulate simple kinematic plant during maneuver: pos += 0.015 * act[:3]
    for _ in range(80):
        pos = pos + 0.015 * act[:3].astype(np.float64)
        if ctrl.needs_motor_proposal:
            break
        act = ctrl.next_action(pos, q90, jaw_open)
        phases.append(ctrl.phase)

    assert ctrl.phase == "nominal"
    assert "shift_xy" in phases and "descend" in phases and "close_jaw" in phases and "lift_obj" in phases
    # Net XY position should have shifted by ~ -delta_t = [0.0, -0.035]
    np.testing.assert_allclose(pos[:2], np.array([0.10, 0.20 - 0.035]), atol=0.006)

    # Release compensation triggers after >=15 post-grasp steps when prop[6] < 0 and jaw < 0.055.
    for _ in range(16):
        ctrl.next_action(pos, q90, jaw_closed, close_prop)
    assert ctrl.release_trigger_count == 0
    open_prop = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    ctrl.next_action(pos, q90, jaw_closed, open_prop)
    act_rel = ctrl.next_action(pos, q90, jaw_closed, open_prop)
    assert ctrl.release_trigger_count == 1
    assert ctrl.phase == "release_shift"
    for _ in range(30):
        pos = pos + 0.015 * act_rel[:3].astype(np.float64)
        if ctrl.needs_motor_proposal:
            break
        act_rel = ctrl.next_action(pos, q90, jaw_closed)
    assert ctrl.phase == "nominal"
    # Release shifted back by -delta_t = [0.0, -0.035] relative to pre-release pos.
    np.testing.assert_allclose(pos[:2], np.array([0.10, 0.20 - 0.070]), atol=0.008)

    # Rearming after >= 30 open steps allows a second grasp trigger (e.g., 2-object LIBERO-10).
    for _ in range(32):
        ctrl.next_action(pos, q90, jaw_open, approach)
    ctrl.next_action(pos, q90, jaw_open, close_prop)
    ctrl.next_action(pos, q90, jaw_open, close_prop)
    assert ctrl.grasp_trigger_count == 2


def test_apply_mujoco_tcp_offset_handles_stale_kinematics():
    from types import SimpleNamespace
    from fluffy_bassoon.xemb.tcp import FINGER_COLLISION_SUFFIXES, apply_mujoco_tcp_offset

    names = [f"gripper0_{suffix}" for suffix in FINGER_COLLISION_SUFFIXES]
    # 90 deg rotation around Z for the finger bodies.
    rmat = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    body_xpos = np.array([[0.1, 0.2, 0.8], [0.1, 0.2, 0.8]], dtype=np.float64)
    geom_pos = np.array(
        [
            [0.0, 0.04, 0.01],
            [0.0, -0.04, 0.01],
            [0.0, 0.038, 0.02],
            [0.0, -0.038, 0.02],
        ],
        dtype=np.float64,
    )
    geom_bodyid = np.array([0, 1, 0, 1], dtype=np.int32)
    stale_lag = np.array([2e-5, -1e-5, 0.0], dtype=np.float64)

    class FakeModel:
        def __init__(self):
            self.ngeom = len(names)
            self.geom_bodyid = geom_bodyid
            self.geom_pos = geom_pos.copy()

        def geom_id2name(self, idx):
            return names[idx]

    class FakeSim:
        def __init__(self):
            self.model = FakeModel()
            body_xmat = np.stack([rmat.reshape(9), rmat.reshape(9)], axis=0)
            true_xpos = np.array([body_xpos[geom_bodyid[i]] + rmat @ self.model.geom_pos[i] for i in range(4)])
            self.data = SimpleNamespace(
                geom_xpos=true_xpos - stale_lag[None],
                body_xmat=body_xmat,
                ncon=12,
            )

        def forward(self):
            self.data.geom_xpos = np.array(
                [body_xpos[geom_bodyid[i]] + rmat @ self.model.geom_pos[i] for i in range(4)],
                dtype=np.float64,
            )

    world = SimpleNamespace(sim=FakeSim())
    meta = apply_mujoco_tcp_offset(world, 0.035, 0.0, 0.0)
    assert meta["max_shift_err_m"] < 1e-12
    assert meta["pad_gap_err_m"] < 1e-12
    assert meta["before_ncon"] == meta["after_ncon"] == 12
