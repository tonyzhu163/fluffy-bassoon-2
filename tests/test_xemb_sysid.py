"""Unit tests for cross-embodiment coordinate transforms, analytic SysID, and JAX readers."""

import math
import numpy as np
import pytest


def test_embodiment_action_and_retarget_roundtrip():
    from fluffy_bassoon.xemb import (
        EVAL_EMBODIMENTS,
        TRAIN_EMBODIMENTS,
        EmbodimentSpec,
        apply_embodiment_action,
        retarget_action_from_motor,
        retarget_observation_for_motor,
        to_local_proprio,
    )

    assert len(TRAIN_EMBODIMENTS) == 7
    assert len(EVAL_EMBODIMENTS) == 12

    low = np.full(7, -1.0, dtype=np.float32)
    high = np.full(7, 1.0, dtype=np.float32)
    rng = np.random.default_rng(123)
    proposal = rng.uniform(-0.45, 0.45, size=(10, 7)).astype(np.float32)
    proposal[:, 6] = np.where(rng.uniform(size=10) > 0.5, 0.8, -0.8)

    test_specs = (
        EmbodimentSpec("arb1", -137.0, 1.0, -1, "test"),
        EmbodimentSpec("arb2", 45.0, 1.55, 1, "test"),
        EmbodimentSpec("arb3", 180.0, 1.25, -1, "test"),
    )
    for spec in test_specs:
        issued = retarget_action_from_motor(
            proposal, spec.yaw_deg, spec.xy_scale, spec.gripper_sign, low, high
        )
        physical = apply_embodiment_action(issued, spec, low, high)
        np.testing.assert_allclose(physical, proposal, atol=1e-5)

    # Frame-0 local proprioception is bitwise identical across all embodiments when eef == anchor.
    anchor = np.array([0.12, -0.05, 0.91], dtype=np.float32)
    raw0 = {
        "robot0_eef_pos": anchor.copy(),
        "robot0_eef_quat_xyzw": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.04, -0.04], dtype=np.float32),
        "agentview_image": np.zeros((256, 256, 3), dtype=np.uint8),
    }
    locals0 = [to_local_proprio(raw0, anchor, s)["robot0_eef_pos"] for s in EVAL_EMBODIMENTS]
    for arr in locals0:
        assert np.array_equal(arr, anchor)

    # Observation retargeting recovers canonical EEF coordinates after motion.
    moved = dict(raw0, robot0_eef_pos=anchor + np.array([0.09, -0.04, 0.03], dtype=np.float32))
    for spec in (*TRAIN_EMBODIMENTS, *EVAL_EMBODIMENTS):
        pub = to_local_proprio(moved, anchor, spec)
        recovered = retarget_observation_for_motor(pub, anchor, spec.yaw_deg)
        np.testing.assert_allclose(recovered["robot0_eef_pos"], moved["robot0_eef_pos"], atol=1e-5)


def test_local_proprioception_is_yaw_invariant():
    from fluffy_bassoon.xemb import (
        ACTIVE_PROBE_BURSTS,
        EmbodimentSpec,
        analytic_proprio_sysid,
        apply_embodiment_action,
        circular_yaw_error_deg,
        fit_nominal_gain,
        to_local_proprio,
    )

    low = np.full(7, -1.0, dtype=np.float32)
    high = np.full(7, 1.0, dtype=np.float32)
    gain = 0.02
    anchor = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    issued_chunks = ACTIVE_PROBE_BURSTS[:, 0, :]  # [3, 7]

    # Calibrate gain from nominal trace.
    nom_spec = EmbodimentSpec("nom", 0.0, 1.0, 1, "test")
    nom_phys = apply_embodiment_action(issued_chunks, nom_spec, low, high)
    nom_deef = gain * nom_phys[:, :3]
    fitted_gain = fit_nominal_gain(issued_chunks[None], nom_deef[None], np.ones((1, 3), dtype=bool))
    assert math.isclose(fitted_gain, gain, rel_tol=1e-5)

    ref_delta_local = None
    for yaw_deg in (0.0, 45.0, 90.0, -135.0, 180.0):
        for xy_scale in (0.65, 1.0, 1.55):
            for gsign in (-1, 1):
                spec = EmbodimentSpec("test", yaw_deg, xy_scale, gsign, "test")
                phys = apply_embodiment_action(issued_chunks, spec, low, high)
                world_deef = gain * phys[:, :3]
                local_deef = []
                for step_d in world_deef:
                    raw = {"robot0_eef_pos": anchor + step_d}
                    pub = to_local_proprio(raw, anchor, spec)
                    local_deef.append(pub["robot0_eef_pos"] - anchor)
                local_deef = np.stack(local_deef)
                if math.isclose(xy_scale, 1.0):
                    if ref_delta_local is None:
                        ref_delta_local = local_deef
                    else:
                        np.testing.assert_allclose(local_deef, ref_delta_local, atol=1e-6)

                delta_jaw = np.stack(
                    (
                        -0.02 * phys[:, 6],
                        0.04 - 0.02 * phys[:, 6],
                    ),
                    axis=-1,
                )
                yaw_hat, scale_hat, gsign_hat = analytic_proprio_sysid(
                    issued_chunks,
                    local_deef,
                    delta_jaw,
                    np.ones((3,), dtype=bool),
                    nominal_gain_m_per_cmd=fitted_gain,
                )
                # Local proprioception keeps OLS yaw estimate at 0 deg regardless of true yaw.
                assert circular_yaw_error_deg(yaw_hat, 0.0) < 1e-3
                assert math.isclose(scale_hat, xy_scale, rel_tol=1e-3)
                assert gsign_hat == gsign


def test_procrustes_so2_equivariance_and_cleared_invariance():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from fluffy_bassoon.xemb import (
        MODEL_NAMES,
        ReaderConfig,
        apply_reader,
        circular_yaw_error_deg,
        init_reader,
        predict_embodiment,
        reader_loss,
        rot2,
    )
    from fluffy_bassoon.xemb.readers import _normalize_cs

    # 1. Exact continuous SO(2) Procrustes recovery across arbitrary angles and scales.
    rng = np.random.default_rng(7)
    issued_xy = np.array([[0.55, 0.0], [0.0, 0.55], [-0.35, -0.35]], dtype=np.float32)
    weights = np.array([1.2, 0.8, 1.5], dtype=np.float32)
    for true_yaw in (-173.4, -135.0, -45.0, 0.0, 33.7, 90.0, 135.0, 179.5):
        for scale in (0.4, 1.0, 2.3):
            v_hat = scale * (issued_xy @ rot2(true_yaw).T)
            c_raw = np.sum(weights * (issued_xy[:, 0] * v_hat[:, 0] + issued_xy[:, 1] * v_hat[:, 1]))
            s_raw = np.sum(weights * (issued_xy[:, 0] * v_hat[:, 1] - issued_xy[:, 1] * v_hat[:, 0]))
            cos_h, sin_h = _normalize_cs(jnp.array([c_raw]), jnp.array([s_raw]), jnp.array([True]))
            pred_deg = float(np.degrees(np.arctan2(np.asarray(sin_h)[0], np.asarray(cos_h)[0])))
            assert circular_yaw_error_deg(pred_deg, true_yaw) < 1e-4

    # 2. Cleared invariance and finite backward gradients across all 4 neural models.
    b, k = 3, 4
    batch = {
        "img_before": rng.uniform(0.0, 1.0, size=(b, k, 64, 64, 3)).astype(np.float32),
        "img_after": rng.uniform(0.0, 1.0, size=(b, k, 64, 64, 3)).astype(np.float32),
        "issued_mean": rng.uniform(-0.5, 0.5, size=(b, k, 7)).astype(np.float32),
        "delta_eef": rng.uniform(-0.05, 0.05, size=(b, k, 3)).astype(np.float32),
        "delta_jaw": rng.uniform(-0.05, 0.05, size=(b, k, 2)).astype(np.float32),
        "valid": np.array(
            [
                [True, True, False, False],
                [True, True, True, True],
                [False, False, False, False],  # fully cleared row in same batch!
            ],
            dtype=bool,
        ),
        "frame0_img": rng.uniform(0.0, 1.0, size=(b, 64, 64, 3)).astype(np.float32),
        "frame0_state": rng.uniform(-0.2, 0.2, size=(b, 8)).astype(np.float32),
        "yaw_deg": np.array([45.0, -90.0, 0.0], dtype=np.float32),
        "xy_scale": np.array([1.0, 0.65, 1.0], dtype=np.float32),
        "gripper_sign": np.array([1, -1, 1], dtype=np.int32),
    }
    mutated = dict(batch)
    mutated["img_before"] = batch["img_before"] * 3.0 + 0.5
    mutated["issued_mean"] = -batch["issued_mean"] * 4.0
    mutated["valid"] = np.zeros((b, k), dtype=bool)

    for idx, model_type in enumerate(MODEL_NAMES):
        cfg = ReaderConfig(model_type=model_type, max_chunks=8)
        params = init_reader(jax.random.key(41 + idx), cfg)

        # Forward and finite gradient check including an all-False row.
        (loss, metrics), grads = jax.value_and_grad(
            lambda p: reader_loss(p, cfg, batch), has_aux=True
        )(params)
        assert jnp.isfinite(loss)
        assert all(jnp.isfinite(v) for v in metrics.values())
        assert all(bool(jnp.isfinite(g).all()) for g in jax.tree.leaves(grads))

        if model_type != "frame0_probe":
            # Cleared input (valid=False everywhere) must yield exact nominal regardless of payload.
            cleared_a = dict(batch, valid=np.zeros((b, k), dtype=bool))
            y_a, s_a, g_a = predict_embodiment(params, cfg, cleared_a)
            y_b, s_b, g_b = predict_embodiment(params, cfg, mutated)
            np.testing.assert_allclose(y_a, 0.0, atol=1e-6)
            np.testing.assert_allclose(s_a, 1.0, atol=1e-6)
            np.testing.assert_array_equal(g_a, np.ones((b,), dtype=np.int32))
            np.testing.assert_allclose(y_a, y_b, atol=1e-6)
            np.testing.assert_allclose(s_a, s_b, atol=1e-6)
            np.testing.assert_array_equal(g_a, g_b)
        else:
            cos_h, sin_h, log_s, grip_l = apply_reader(params, cfg, batch)
            assert cos_h.shape == (b,) and jnp.isfinite(log_s).all() and jnp.isfinite(grip_l).all()
