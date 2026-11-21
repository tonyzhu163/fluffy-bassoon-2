"""Unit tests for V2 equivariant readers, SO(3) Procrustes, streaming, and competitive baselines."""

import math
import numpy as np
import pytest

from fluffy_bassoon.xemb.baselines import (
    broyden_uvs_yaw,
    reground_babbling_sysid,
)
from fluffy_bassoon.xemb.embodiment import circular_yaw_error_deg, rot2
from fluffy_bassoon.xemb.equiv_v2 import (
    ReaderV2Config,
    StreamingProcrustesEstimator,
    apply_reader_v2,
    extract_visual_vectors_v2,
    init_reader_v2,
    predict_embodiment_v2,
    procrustes_so3,
    reader_loss_v2,
)


def _synthetic_batch(bsz=4, k=6, h=16, w=16, seed=0):
    rng = np.random.default_rng(seed)
    return dict(
        img_before=rng.integers(0, 255, size=(bsz, k, h, w, 3), dtype=np.uint8),
        img_after=rng.integers(0, 255, size=(bsz, k, h, w, 3), dtype=np.uint8),
        dis_flow=rng.normal(scale=0.3, size=(bsz, k, h, w, 2)).astype(np.float32),
        issued_mean=rng.uniform(-0.8, 0.8, size=(bsz, k, 7)).astype(np.float32),
        delta_eef=rng.normal(scale=0.02, size=(bsz, k, 3)).astype(np.float32),
        delta_jaw=rng.normal(scale=0.005, size=(bsz, k, 2)).astype(np.float32),
        valid=np.ones((bsz, k), dtype=bool),
        yaw_deg=np.array([0.0, 45.0, -135.0, 180.0][:bsz], dtype=np.float32),
        xy_scale=np.array([1.0, 0.75, 1.35, 0.65][:bsz], dtype=np.float32),
        gripper_sign=np.array([1, -1, 1, -1][:bsz], dtype=np.int32),
    )


def _random_so3(rng):
    a = rng.normal(size=(3, 3))
    q, _ = np.linalg.qr(a)
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q.astype(np.float32)


def test_reground_babbling_recovers_world_frame_and_fails_rotated_local_frame():
    rng = np.random.default_rng(123)
    k = 200  # 1,000 controls at 5 steps/chunk (ReGround budget)
    true_yaw = 135.0
    true_scale = 0.75
    true_grip = -1
    nom_gain = 0.02
    iss = rng.uniform(-0.7, 0.7, size=(k, 7)).astype(np.float32)
    valid = np.ones((k,), dtype=bool)

    r = rot2(true_yaw)
    deef_world = np.zeros((k, 3), dtype=np.float32)
    deef_world[:, :2] = (iss[:, :2] @ r.T) * (nom_gain * true_scale)
    deef_world[:, 2] = iss[:, 2] * (nom_gain * true_scale)
    djaw = np.zeros((k, 2), dtype=np.float32)
    djaw[:, 0] = -true_grip * iss[:, 6] * 0.01

    # World-frame proprioception: ReGround recovers (yaw, scale, grip) accurately.
    yh, sh, gh = reground_babbling_sysid(iss, deef_world, djaw, valid, nominal_gain_m_per_cmd=nom_gain)
    assert circular_yaw_error_deg(yh, true_yaw) < 1e-2
    assert abs(sh - true_scale) < 1e-2
    assert gh == true_grip

    # Rotated local proprioception (delta_eef_local = R(-yaw) delta_eef_world): yaw-blind even at 1,000 steps!
    deef_local = deef_world.copy()
    deef_local[:, :2] = deef_world[:, :2] @ rot2(-true_yaw).T
    yh_loc, sh_loc, gh_loc = reground_babbling_sysid(iss, deef_local, djaw, valid, nominal_gain_m_per_cmd=nom_gain)
    assert circular_yaw_error_deg(yh_loc, 0.0) < 1e-2
    assert circular_yaw_error_deg(yh_loc, true_yaw) > 130.0
    assert abs(sh_loc - true_scale) < 1e-2
    assert gh_loc == true_grip


def test_broyden_uvs_recovers_planar_yaw_with_z_parallax():
    rng = np.random.default_rng(77)
    proj = np.array([[120.0, 15.0], [-20.0, 110.0], [35.0, -45.0]], dtype=np.float64)
    k = 12
    for true_yaw in (0.0, 45.0, -90.0, 135.0, 180.0):
        iss = rng.uniform(-0.8, 0.8, size=(k, 7))
        r = rot2(true_yaw).astype(np.float64)
        d_world = np.zeros((k, 3), dtype=np.float64)
        d_world[:, :2] = (iss[:, :2] @ r.T) * 0.03
        d_world[:, 2] = iss[:, 2] * 0.03
        mot = d_world @ proj
        valid = np.ones((k,), dtype=bool)
        yh = broyden_uvs_yaw(iss, mot, d_world[:, 2], valid, proj)
        assert circular_yaw_error_deg(yh, true_yaw) < 5.0


def test_streaming_procrustes_matches_batch_exact():
    rng = np.random.default_rng(9)
    a = rng.uniform(-1.0, 1.0, size=(8, 2))
    true_yaw = -67.5
    v = a @ rot2(true_yaw).T
    w = rng.uniform(0.5, 2.0, size=(8,))

    est = StreamingProcrustesEstimator(prior_weight=0.0)
    for k in range(8):
        est.update(a[k], v[k], float(w[k]), log_scale_est=math.log(1.35), grip_logit_est=-4.0)
    yh, sh, gh = est.estimate()
    assert circular_yaw_error_deg(yh, true_yaw) < 1e-4
    assert abs(sh - 1.35) < 1e-4
    assert gh == -1


def test_v2_so2_exact_equivariance_with_dz_parallax():
    jax = pytest.importorskip("jax")
    cfg = ReaderV2Config(model_type="factored_equiv_v2", visual_mode="dis_lumaflow", cnn_width=32, mlp_width=48, use_dz_parallax=True)
    params = init_reader_v2(jax.random.key(0), cfg)
    batch = _synthetic_batch(bsz=3, k=5, h=16, w=16, seed=11)

    yaw0, scale0, grip0 = predict_embodiment_v2(params, cfg, batch)
    v_planar, vis_w = extract_visual_vectors_v2(params, cfg, batch)
    assert np.isfinite(np.asarray(v_planar)).all()

    # Exact SO(2) equivariance: rotating issued XY by any continuous delta shifts yaw_hat by -delta
    # even with non-zero local_dz parallax!
    for delta in (37.5, -123.4, 161.8):
        rot_batch = dict(batch)
        iss_rot = batch["issued_mean"].copy()
        iss_rot[..., :2] = iss_rot[..., :2] @ rot2(delta).T
        rot_batch["issued_mean"] = iss_rot
        yaw1, _, _ = predict_embodiment_v2(params, cfg, rot_batch)
        for y0, y1 in zip(yaw0, yaw1, strict=True):
            assert circular_yaw_error_deg(float(y1), float(y0) - delta) < 1e-3

    # Cleared context must return exact nominal identity (0 deg, 1.0, +1).
    clr = dict(batch, valid=np.zeros_like(batch["valid"]))
    yc, sc, gc = predict_embodiment_v2(params, cfg, clr)
    assert np.allclose(yc, 0.0) and np.allclose(sc, 1.0) and np.all(gc == 1)


def test_v2_so3_exact_equivariance_and_finite_grads():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    rng = np.random.default_rng(42)
    r_true = _random_so3(rng)
    a = rng.normal(size=(2, 6, 3)).astype(np.float32)
    v = np.einsum("ij,bkj->bki", r_true, a)
    w = np.ones((2, 6), dtype=np.float32)
    r_hat = np.asarray(procrustes_so3(jnp.asarray(a), jnp.asarray(v), jnp.asarray(w), jnp.ones((2,), dtype=bool)))
    assert np.allclose(r_hat[0], r_true, atol=1e-4)

    # Test full SO(3) neural reader equivariance and finite gradients.
    cfg = ReaderV2Config(model_type="factored_equiv_so3", visual_mode="dis_lumaflow", cnn_width=24, mlp_width=32)
    params = init_reader_v2(jax.random.key(5), cfg)
    batch = _synthetic_batch(bsz=2, k=6, h=16, w=16, seed=5)
    batch["rot_mat"] = np.stack([r_true, _random_so3(rng)], axis=0)
    batch["xyz_scale"] = np.array([[0.8, 1.2, 0.9], [1.3, 0.7, 1.1]], dtype=np.float32)

    r0, _, _ = apply_reader_v2(params, cfg, batch)
    q = _random_so3(rng)
    b_rot = dict(batch)
    iss_rot = batch["issued_mean"].copy()
    iss_rot[..., :3] = np.einsum("ij,bkj->bki", q.T, iss_rot[..., :3])
    b_rot["issued_mean"] = iss_rot
    r1, _, _ = apply_reader_v2(params, cfg, b_rot)
    expected_r1 = np.einsum("bij,jk->bik", np.asarray(r0), q)
    assert np.allclose(np.asarray(r1), expected_r1, atol=1e-3)

    (loss, _), grads = jax.value_and_grad(lambda p: reader_loss_v2(p, cfg, batch), has_aux=True)(params)
    assert np.isfinite(float(loss))
    for leaf in jax.tree_util.tree_leaves(grads):
        assert np.isfinite(np.asarray(leaf)).all()


def test_perceiver_icba_cleared_and_grads():
    jax = pytest.importorskip("jax")
    cfg = ReaderV2Config(
        model_type="perceiver_icba",
        visual_mode="luminance_flow",
        cnn_width=24,
        perceiver_latents=4,
        perceiver_width=32,
        perceiver_layers=2,
        perceiver_heads=4,
    )
    params = init_reader_v2(jax.random.key(1), cfg)
    batch = _synthetic_batch(bsz=2, k=5, h=16, w=16, seed=2)
    (loss, _), grads = jax.value_and_grad(lambda p: reader_loss_v2(p, cfg, batch), has_aux=True)(params)
    assert np.isfinite(float(loss))
    for leaf in jax.tree_util.tree_leaves(grads):
        assert np.isfinite(np.asarray(leaf)).all()

    clr = dict(batch, valid=np.zeros_like(batch["valid"]))
    yc, sc, gc = predict_embodiment_v2(params, cfg, clr)
    assert np.allclose(yc, 0.0) and np.allclose(sc, 1.0) and np.all(gc == 1)


def test_equiv_flow_homography_exact_so2_and_zero_cardinal_quantization():
    jax = pytest.importorskip("jax")
    cfg = ReaderV2Config(
        model_type="equiv_flow_homography",
        visual_mode="dis_lumaflow",
        cnn_width=32,
        use_dz_parallax=True,
    )
    params = init_reader_v2(jax.random.key(7), cfg)
    batch = _synthetic_batch(bsz=3, k=5, h=16, w=16, seed=19)

    yaw0, _, _ = predict_embodiment_v2(params, cfg, batch)
    for delta in (45.0, -135.0, 29.3):
        b_rot = dict(batch)
        iss = batch["issued_mean"].copy()
        iss[..., :2] = iss[..., :2] @ rot2(delta).T
        b_rot["issued_mean"] = iss
        yaw1, _, _ = predict_embodiment_v2(params, cfg, b_rot)
        for y0, y1 in zip(yaw0, yaw1, strict=True):
            assert circular_yaw_error_deg(float(y1), float(y0) - delta) < 1e-3

    # Verify finite gradients through direction-blind mask CNN + homography + Procrustes.
    (loss, _), grads = jax.value_and_grad(lambda p: reader_loss_v2(p, cfg, batch), has_aux=True)(params)
    assert np.isfinite(float(loss))
    for leaf in jax.tree_util.tree_leaves(grads):
        assert np.isfinite(np.asarray(leaf)).all()

    clr = dict(batch, valid=np.zeros_like(batch["valid"]))
    yc, sc, gc = predict_embodiment_v2(params, cfg, clr)
    assert np.allclose(yc, 0.0) and np.allclose(sc, 1.0) and np.all(gc == 1)

