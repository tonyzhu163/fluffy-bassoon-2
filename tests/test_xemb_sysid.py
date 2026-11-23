"""Unit tests for cross-embodiment coordinate transforms, analytic SysID, and JAX readers."""

from functools import partial
import math
import os
from pathlib import Path
import numpy as np
import pytest


def reader_jax():
    if "OPENPI_DATA_HOME" in os.environ:
        import jax  # A missing reader dependency must fail Linux qualification.
        return jax
    return pytest.importorskip("jax")


def require_flax():
    """`eval_live` imports flax at module load; skip off the qualified Linux runtime."""
    if "OPENPI_DATA_HOME" not in os.environ:
        pytest.importorskip("flax")


@pytest.fixture
def run_scripts(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "outputs" / "libero-xemb-sysid-20261119-r1"
    monkeypatch.syspath_prepend(str(path))


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
    jax = reader_jax()
    import jax.numpy as jnp
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
    from fluffy_bassoon.xemb.readers import _procrustes_yaw

    # 1. Exact continuous SO(2) Procrustes recovery across arbitrary angles and scales.
    rng = np.random.default_rng(7)
    issued_xy = np.array([[0.55, 0.0], [0.0, 0.55], [-0.35, -0.35]], dtype=np.float32)
    weights = np.array([1.2, 0.8, 1.5], dtype=np.float32)
    for true_yaw in (-173.4, -135.0, -45.0, 0.0, 33.7, 90.0, 135.0, 179.5):
        for scale in (0.4, 1.0, 2.3):
            v_hat = scale * (issued_xy @ rot2(true_yaw).T)
            cos_h, sin_h = _procrustes_yaw(
                jnp.asarray(issued_xy)[None], jnp.asarray(v_hat)[None], jnp.asarray(weights)[None], jnp.array([True])
            )
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

    byte_batch = dict(batch)
    for field in ("img_before", "img_after", "frame0_img"):
        byte_batch[field] = (batch[field] * 255).astype(np.uint8)
        batch[field] = byte_batch[field].astype(np.float32) / 255.0
    padded = dict(batch)
    for field in ("img_before", "img_after", "issued_mean", "delta_eef", "delta_jaw"):
        padded[field] = batch[field].copy()
        padded[field][~batch["valid"]] = np.nan

    for idx, model_type in enumerate(MODEL_NAMES):
        cfg = ReaderConfig(model_type=model_type, max_chunks=8)
        params = init_reader(jax.random.key(41 + idx), cfg)

        expected = apply_reader(params, cfg, batch)
        # On-disk uint8 and the unit-float motor callback encode the same RGB.
        np.testing.assert_allclose(apply_reader(params, cfg, byte_batch), expected, atol=1e-5)
        np.testing.assert_allclose(apply_reader(params, cfg, padded), expected, atol=1e-5)
        if model_type == "frame0_probe":
            frame0_only = {field: batch[field] for field in ("frame0_img", "frame0_state")}
            np.testing.assert_array_equal(apply_reader(params, cfg, frame0_only), expected)

        # Forward and finite gradient check including an all-False row.
        (loss, metrics), grads = jax.value_and_grad(
            partial(reader_loss, cfg=cfg, batch=padded), has_aux=True
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


def test_shifted_images_and_calibrated_visual_yaw():
    from fluffy_bassoon.xemb import rot2
    from fluffy_bassoon.xemb.readers import (
        analytic_proprio_sysid, analytic_visual_yaw, fit_visual_projection, image_motion,
    )

    yy, xx = np.mgrid[:64, :64]

    def texture(x, y):
        return (np.cos(x * 0.10) + np.sin(y * 0.12))[..., None]

    valid = np.array([[True, False]])
    before = np.stack([texture(xx, yy), texture(xx, yy)])[None]
    after = np.stack([texture(xx - 0.8, yy + 0.6), np.full_like(before[0, 0], np.nan)])[None]
    motion = image_motion(before, after, valid)
    # Finite differences approximate the continuous subpixel shift.
    np.testing.assert_allclose(motion[0, 0], [0.8, -0.6], atol=0.02)
    np.testing.assert_array_equal(motion[0, 1], [0, 0])

    camera = np.array([[-3., 2.], [1., 4.], [1.5, -1.]])
    xyz = np.array([[1., 0., 0.], [0., 1., 0.], [0.5, -0.2, 1.]])
    observed = np.ones(3, dtype=bool)
    projection = fit_visual_projection(xyz @ camera, xyz, observed)
    np.testing.assert_allclose(projection, camera)
    local = xyz.copy()
    local[:, :2] = xyz[:, :2] @ rot2(-57.0).T
    image_displacements = local @ camera
    y = analytic_visual_yaw(xyz, image_displacements, local[:, 2], observed, projection)
    assert abs(y + 57.0) < 1e-4
    with pytest.raises(ValueError, match="projection"):
        fit_visual_projection(np.zeros((3, 2)), xyz, observed)

    # Observed cross-axis motion must actually change the proprioceptive estimate.
    iss = np.pad(xyz, ((0, 0), (0, 4)))
    yaw, _, _ = analytic_proprio_sysid(iss, local, np.zeros((3, 2)), observed, 1.0)
    assert abs(yaw + 57.0) < 1e-4


def test_short_terminal_prefix_and_legacy_passive_cap(run_scripts, tmp_path):
    from collect import chunk_counts, frame0_identity, issued_exposure, read_support

    def save(n, controls):
        images = np.zeros((n, 64, 64, 3), dtype=np.uint8)
        tape = dict(img_before=images, img_after=images, issued_mean=np.ones((n, 7), dtype=np.float32),
                    delta_eef=np.zeros((n, 3), dtype=np.float32), delta_jaw=np.zeros((n, 2), dtype=np.float32),
                    valid=np.arange(n) * 5 < controls, frame0_img=images[0], frame0_state=np.zeros(8, np.float32))
        np.savez(tmp_path / "tape.npz", **tape)
        return dict(index=0, passive_path="tape.npz", passive_controls=controls, **frame0_identity("reset", tape))

    short = read_support(tmp_path, save(3, 7), "passive_attempt1")
    np.testing.assert_array_equal(short["valid"], [True, True, False])
    np.testing.assert_allclose(issued_exposure(short)[:, 0], [1., 0.4, 0.])
    long_row = save(24, 120)
    long_tape = read_support(tmp_path, long_row, "passive_attempt1")
    assert long_tape["chunk_steps"].sum() == 15
    assert len(long_tape["img_after"]) == 3
    with pytest.raises(ValueError, match="Reset mismatch"):
        read_support(tmp_path, dict(long_row, frame0_state_sha256="different"), "passive_attempt1")
    with pytest.raises(ValueError, match="prefix"):
        chunk_counts(np.array([True, False, True]), 7)


def test_checkpoint_restores_lists_and_preserves_predictions(run_scripts, tmp_path):
    jax = reader_jax()
    require_flax()
    from eval_live import load_reader_checkpoint
    from fluffy_bassoon.xemb import ReaderConfig, apply_reader, init_reader
    from train_readers import save_checkpoint

    if os.environ.get("JAX_PLATFORMS") == "cpu":
        assert jax.default_backend() == "cpu"
    if "OPENPI_DATA_HOME" in os.environ:
        import flax
        import optax

        for module in (jax, flax, optax, np):
            assert "robocasa365-native-20261101-r1" not in Path(module.__file__).parts, module.__file__

    cfg = ReaderConfig(model_type="joint_transformer", cnn_width=16, transformer_width=16,
                       transformer_layers=1, max_chunks=1)
    params = init_reader(jax.random.key(41), cfg)
    path = tmp_path / "active_probe-joint_transformer-seed41.msgpack"
    save_checkpoint(path, params, {}, 1, 41, "active_probe", cfg, 1.0, "archive-a", np.random.default_rng(41))
    args = (tmp_path, "active_probe", cfg.model_type, 41)
    restored, cfg2 = load_reader_checkpoint(*args, "archive-a", 1)
    tape = dict(img_before=np.zeros((1, 1, 64, 64, 3), np.uint8), img_after=np.ones((1, 1, 64, 64, 3), np.uint8),
                issued_mean=np.zeros((1, 1, 7)), delta_eef=np.zeros((1, 1, 3)), delta_jaw=np.zeros((1, 1, 2)),
                valid=np.ones((1, 1), bool))
    np.testing.assert_array_equal(apply_reader(restored, cfg2, tape), apply_reader(params, cfg, tape))
    with pytest.raises(ValueError, match="archive"):
        load_reader_checkpoint(*args, "archive-b", 1)


def test_panel_integrity_donors_and_live_gate(run_scripts):
    reader_jax()
    require_flax()
    from collect import build_schedule, validate_archive
    from eval_live import control_account, query_seed, select_panel
    from train_readers import make_donor_permutation, offline_gate

    schedule = build_schedule("full")
    assert len(schedule) == 224 + 96
    assert set(build_schedule("poc")).issubset(schedule)
    rows = []
    for i, (split, suite, task, init, seed, spec) in enumerate(schedule):
        rows.append(dict(index=i, split=split, suite=suite, task_id=task, init_state_id=init, seed=seed,
                         embodiment=spec.name, regime=spec.regime, yaw_deg=spec.yaw_deg, xy_scale=spec.xy_scale,
                         gripper_sign=spec.gripper_sign, active_controls=15, active_path=f"{i}.npz",
                         physics="physics", frame0_img_sha256="rgb", frame0_state_sha256="state"))
    archive = dict(panel="full", trials=rows, modes=["active_probe"], passive_max_controls=15)
    assert validate_archive(archive) == 40
    for field, wrong in (("split", "dev"), ("seed", -1), ("frame0_state_sha256", "leak")):
        with pytest.raises(ValueError):
            validate_archive(dict(archive, trials=[dict(rows[0], **{field: wrong}), *rows[1:]]))
    with pytest.raises(ValueError, match="Incomplete"):
        validate_archive(dict(archive, trials=rows[:-1]))

    dev = [r for r in rows if r["split"] == "dev"]
    donors = make_donor_permutation(dev, np.random.default_rng(41))
    assert sorted(donors) == list(range(len(dev)))
    suites = {r["suite"] for r in dev}
    seeds = set()
    for row, donor in zip(dev, donors, strict=True):
        assert (row["suite"], row["task_id"], row["seed"]) == (dev[donor]["suite"], dev[donor]["task_id"], dev[donor]["seed"])
        assert row["embodiment"] != dev[donor]["embodiment"]
        seeds.add(query_seed(suites, row["suite"], row["task_id"], 13))
    assert len(seeds) == 8
    metric = {"retained": {"continuous_angle_ood": {"yaw_err_deg": {"mean": 20.0}}}}
    report = dict(complete=True, steps=1500, seeds=[41, 42, 43], panel="full", archive_sha256="archive",
                  modes={"active_probe": {"models": {"factored_equiv": metric}}}, offline_gate_passed=True)
    assert offline_gate({"modes": {}}, 20.)[0] is False
    with pytest.raises(ValueError, match="gate"):
        select_panel(archive, report, "development", 13, "archive", 41)
    metric["retained"]["continuous_angle_ood"]["yaw_err_deg"]["mean"] = 19.9
    assert len(select_panel(archive, report, "development", 13, "archive", 41)) == 96
    with pytest.raises(ValueError, match="Pilot"):
        select_panel(archive, report, "poc", 13, "archive", 41)
    pilot_rows = [rows[schedule.index(entry)] for entry in build_schedule("poc")]
    poc = dict(archive, panel="poc", trials=pilot_rows)
    assert validate_archive(poc) == 5
    poc_report = dict(report, panel="poc", steps=1500, seeds=[41], offline_gate_passed=None)
    assert len(select_panel(poc, poc_report, "poc", 13, "archive", 41)) == 3
    with pytest.raises(ValueError, match="completed fit"):
        select_panel(archive, dict(report, complete=False), "poc", 13, "archive", 41)

    record = dict(controls=9, reset={"horizon": 100})
    assert control_account("ours_online_prefix", {}, {}, record)["query_motor"] == 0
    assert control_account("ours_retained_passive", {"passive_controls": 7}, {}, record)["total_executed"] == 16
    assert control_account("ours_cleared", {"active_controls": 15}, {}, record)["support_observed"] == 0
