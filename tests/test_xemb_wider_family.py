import numpy as np
import pytest

from fluffy_bassoon.xemb.wider_family import (
    FAMILIES,
    delay_commands,
    delay_keep_motion,
    interface_matrix,
    lagged,
    operator_error,
    procrustes_yaw_deg,
    relabel,
    ridge_linear_map,
    rot2_batch,
    sample_interfaces,
    soft_shape,
)


def _recorded(rng, n=6, k=3):
    yaw = rng.uniform(-180, 180, n)
    gain = rng.uniform(0.65, 1.5, n)
    grip = rng.choice([-1, 1], n)
    issued = rng.uniform(-1, 1, (n, k, 7)).astype(np.float32)
    world = np.einsum("bij,bkj->bki", gain[:, None, None] * rot2_batch(yaw), issued[..., :2])
    delta = np.zeros((n, k, 3), dtype=np.float32)
    delta[..., :2] = np.einsum("bij,bkj->bki", rot2_batch(-yaw), world)
    delta[..., 2] = gain[:, None] * issued[..., 2]
    batch = dict(
        issued_mean=issued,
        delta_eef=delta,
        valid=np.ones((n, k), dtype=bool),
        yaw_deg=yaw.astype(np.float32),
        xy_scale=gain.astype(np.float32),
        gripper_sign=grip.astype(np.int32),
    )
    return batch, world, gain[:, None] * issued[..., 2], grip[:, None] * issued[..., 6]


@pytest.mark.parametrize("family", FAMILIES)
def test_relabel_preserves_physical_motion(family):
    rng = np.random.default_rng(0)
    batch, world, world_z, jaw_cmd = _recorded(rng)
    target = sample_interfaces(rng, family, len(world))
    out = relabel(batch, target)

    replay = np.einsum("bij,bkj->bki", interface_matrix(target), out["issued_mean"][..., :2])
    np.testing.assert_allclose(replay, world, atol=1e-5)
    np.testing.assert_allclose(target["z_gain"][:, None] * out["issued_mean"][..., 2], world_z, atol=1e-5)
    np.testing.assert_allclose(target["gripper_sign"][:, None] * out["issued_mean"][..., 6], jaw_cmd, atol=1e-6)
    local = np.einsum("bij,bkj->bki", rot2_batch(-target["yaw_deg"]), world)
    np.testing.assert_allclose(out["delta_eef"][..., :2], local, atol=1e-5)
    np.testing.assert_allclose(out["delta_eef"][..., 2], batch["delta_eef"][..., 2])


def test_relabel_to_recorded_interface_is_identity():
    rng = np.random.default_rng(1)
    batch, *_ = _recorded(rng)
    same = dict(
        yaw_deg=batch["yaw_deg"].astype(np.float64),
        shape=batch["xy_scale"][:, None, None].astype(np.float64) * np.eye(2),
        z_gain=batch["xy_scale"].astype(np.float64),
        gripper_sign=batch["gripper_sign"],
    )
    out = relabel(batch, same)
    np.testing.assert_allclose(out["issued_mean"], batch["issued_mean"], atol=1e-5)
    np.testing.assert_allclose(out["delta_eef"], batch["delta_eef"], atol=1e-5)


def test_noiseless_solvers_recover_the_interface():
    rng = np.random.default_rng(2)
    batch, world, *_ = _recorded(rng)
    target = sample_interfaces(rng, "general", len(world))
    out = relabel(batch, target)
    cmd = out["issued_mean"][..., :2]
    ones = np.ones(cmd.shape[:2])

    yaw = procrustes_yaw_deg(out["delta_eef"][..., :2], world, ones)
    shape = ridge_linear_map(cmd, out["delta_eef"][..., :2], ones, np.zeros((len(world), 2, 2)), 0.0)
    est = rot2_batch(yaw) @ shape
    np.testing.assert_allclose(operator_error(interface_matrix(target), est), 0.0, atol=1e-4)
    # The paper's solver sees only a rotation and a gain, so it cannot represent these interfaces.
    paper = out["xy_scale"][:, None, None] * rot2_batch(procrustes_yaw_deg(cmd, world, ones))
    assert operator_error(interface_matrix(target), paper).min() > 0.05


def test_delay_pairs_each_motion_with_the_next_command():
    rng = np.random.default_rng(3)
    batch, *_ = _recorded(rng, k=4)
    out = delay_commands(batch, 1)
    np.testing.assert_array_equal(out["issued_mean"][:, :3], batch["issued_mean"][:, 1:])
    assert out["valid"][:, :3].all() and not out["valid"][:, 3].any()


def test_soft_shape_recovers_a_general_map_when_commands_span_the_plane():
    rng = np.random.default_rng(3)
    true = rng.uniform(-1.5, 1.5, (5, 2, 2))
    cmd = rng.uniform(-1, 1, (5, 3, 2))
    moved = np.einsum("bij,bkj->bki", true, cmd)
    shape, _, _, share = soft_shape(cmd, moved, np.ones((5, 3), dtype=bool), 1e-9)
    np.testing.assert_allclose(shape, true, atol=1e-5)
    assert share.max() < 1e-5


def test_soft_shape_completes_one_command_direction_with_a_rotation():
    rng = np.random.default_rng(4)
    swap = 1.2 * np.array([[0.0, 1.0], [1.0, 0.0]])
    direction = np.array([np.cos(0.4), np.sin(0.4)])
    cmd = rng.uniform(0.2, 1.0, (4, 3, 1)) * direction
    moved = cmd @ swap.T
    shape, resid, _, share = soft_shape(cmd, moved, np.ones((4, 3), dtype=bool), 0.3)
    np.testing.assert_allclose(shape @ direction, np.tile(swap @ direction, (4, 1)), atol=1e-6)
    np.testing.assert_allclose(np.linalg.det(shape), 1.2**2, rtol=1e-5)
    np.testing.assert_allclose(resid, 0.0, atol=1e-6)
    np.testing.assert_allclose(share, 1.0, atol=1e-5)


def test_delay_keep_motion_is_explained_by_the_one_chunk_lag():
    rng = np.random.default_rng(5)
    batch, *_ = _recorded(rng, n=8)
    gain = batch["xy_scale"].astype(np.float64)
    delayed = delay_keep_motion(batch, 1, rows=np.arange(8) < 6)
    np.testing.assert_array_equal(delayed["cmd_valid"][:6], [[True, True, False]] * 6)
    np.testing.assert_array_equal(delayed["cmd_valid"][6:], True)
    np.testing.assert_array_equal(delayed["issued_mean"][6:], batch["issued_mean"][6:])
    cmd = delayed["issued_mean"][..., :2].astype(np.float64)
    moved = delayed["delta_eef"][..., :2].astype(np.float64)
    for lag, exact in ((1, True), (0, False)):
        pair = delayed["valid"] & lagged(delayed["cmd_valid"], lag)
        shape, resid, _, _ = soft_shape(lagged(cmd, lag), moved, pair, 1e3)
        if exact:
            np.testing.assert_allclose(resid[:6], 0.0, atol=1e-5)
            np.testing.assert_allclose(shape[:6], gain[:6, None, None] * np.eye(2), atol=1e-3)
        else:
            assert resid[:6].min() > 1e-3
