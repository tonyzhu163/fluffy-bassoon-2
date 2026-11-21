#!/usr/bin/env python3
"""Stage 2D CPU qualification. Run from a clone containing REV. JSON on stdout."""
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from unittest.mock import patch

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True

REV = "437da7f677632e9495c813aaa52071e207ba834e"
ROOT = Path(__file__).resolve().parents[2]
REPORT = {"revision": REV, "python": platform.python_version(), "checks": {}, "metrics": {}}
REF_TOL = 1e-10
JAX_TOL = 2e-5  # max matrix entry, float32 SVD on this well-conditioned fixture


def check(name, passed, **metrics):
    REPORT["checks"][name] = bool(passed)
    REPORT["metrics"][name] = metrics


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def verify_source():
    for name in ("embodiment", "readers", "equiv_v2"):
        path = f"src/fluffy_bassoon/xemb/{name}.py"
        pinned = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{REV}:{path}"])
        if (ROOT / path).read_bytes() != pinned:
            raise ValueError(f"Check out the pinned reader source before running: {path}")


def fixture_rotation():
    roll, pitch, yaw = np.deg2rad([23., -31., 47.])
    cr, cp, cy = np.cos([roll, pitch, yaw])
    sr, sp, sy = np.sin([roll, pitch, yaw])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def forward(u, rotation, gains, sign):
    out = u.copy()
    out[..., :3] = (u[..., :3] * gains) @ rotation.T
    out[..., 6] *= sign
    return out


def inverse(v, rotation, gains, sign):
    out = v.copy()
    out[..., :3] = (v[..., :3] @ rotation) / gains
    out[..., 6] *= sign
    return out


def error(a, b):
    return float(np.max(np.abs(a - b)))


def angle_deg(a, b):
    rel = a.T @ b
    skew = rel - rel.T
    sine = .5 * np.linalg.norm(skew[[2, 0, 1], [1, 2, 0]])
    return float(np.degrees(np.arctan2(sine, .5 * (np.trace(rel) - 1))))


def excitation(label, xyz):
    singular = np.linalg.svd(xyz, compute_uv=False)
    rank = int(np.linalg.matrix_rank(xyz))
    check(label, rank == 3, rank=rank, singular_values=singular.tolist(),
          condition=float(singular[0] / singular[-1]) if rank == 3 else None)


def geometry(bursts, r, g, s):
    controls = bursts.reshape(15, 7)
    means = bursts.mean(axis=1)
    xyz = controls[:, :3]
    excitation("15_control_xyz", xyz)
    excitation("3_chunk_mean_xyz", means[:, :3])
    prefix = [int(np.linalg.matrix_rank(xyz[:n])) for n in (5, 10, 15)]
    jaw_signs = np.unique(controls[:, 6])
    check("prefix_and_jaw_excitation", prefix == [1, 2, 3] and np.array_equal(jaw_signs, [-1, 1]),
          prefix_ranks=prefix, jaw_levels=jaw_signs.tolist(),
          joint_xyz_jaw_rank=int(np.linalg.matrix_rank(controls[:, [0, 1, 2, 6]])))

    # Paired held-out canonical commands, with nonzero wrist and both jaw polarities.
    target = np.random.default_rng(2).uniform(-.2, .2, (32, 7))
    applied = forward(target, r, g, s)
    oracle = inverse(target, r, g, s)
    returned = forward(oracle, r, g, s)
    clean = (np.eye(3), np.ones(3), 1)
    check("clean_identity", np.array_equal(forward(target, *clean), target)
          and np.array_equal(inverse(target, *clean), target))
    xyz_nom = error(applied[:, :3], target[:, :3])
    jaw_nom = error(applied[:, 6], target[:, 6])
    oracle_error = error(returned, target)
    issue_error = error(inverse(applied, r, g, s), target)
    check("oracle_vs_nominal", max(oracle_error, issue_error) < REF_TOL
          and min(xyz_nom, jaw_nom) > .01 and np.max(np.abs(oracle)) < 1,
          oracle_max_error=oracle_error, inverse_after_forward_error=issue_error,
          nominal_xyz_max_error=xyz_nom, nominal_jaw_max_error=jaw_nom,
          max_abs_issued=float(np.max(np.abs(oracle))))
    wrong = oracle.copy()
    wrong[:, :3] = (target[:, :3] / g) @ r
    order_error = error(forward(wrong, r, g, s)[:, :3], target[:, :3])
    check("inverse_order_negative_control", order_error > .01, xyz_max_error=order_error)
    check("wrist_passthrough", np.array_equal(applied[:, 3:6], target[:, 3:6])
          and np.array_equal(oracle[:, 3:6], target[:, 3:6]))

    # Independent full-map solve from noiseless pairs; column norms give positive gains.
    world = forward(controls, r, g, s)[:, :3]
    matrix = np.linalg.lstsq(xyz, world, rcond=None)[0].T
    fitted_g = np.linalg.norm(matrix, axis=0)
    fitted_r = matrix / fitted_g
    jaw_width_change = -.01 * forward(controls, r, g, s)[:, 6]
    fitted_s = int(np.sign(-controls[:, 6] @ jaw_width_change))
    fit_error = error(forward(inverse(target, fitted_r, fitted_g, fitted_s), r, g, s), target)
    check("identified_map_on_held_out_commands", max(error(fitted_r, r), error(fitted_g, g),
          fit_error, abs(np.linalg.det(fitted_r) - 1)) < REF_TOL and fitted_s == s,
          fitted_rotation=fitted_r.tolist(), fitted_gains=fitted_g.tolist(),
          fitted_jaw_sign=fitted_s, rotation_error_deg=angle_deg(fitted_r, r),
          held_out_max_error=fit_error)


def reader_gate(mod, means, r, g, s):
    import jax
    import jax.numpy as jnp

    REPORT["jax"] = jax.__version__
    check("cpu_backend", jax.default_backend() == "cpu", backend=jax.default_backend())
    # Rows: compound, clean, and cleared compound (same observations, all masked).
    valid = np.ones((3, len(means)), dtype=bool)
    valid[-1] = False
    issued = np.broadcast_to(means, (3, *means.shape)).copy()
    rotations = np.stack([r, np.eye(3), r])
    gains = np.stack([g, np.ones(3), g])
    signs = np.array([s, 1, s])
    local = issued[..., :3] * gains[:, None]
    world = np.einsum("bij,bkj->bki", rotations, local)
    expected = np.stack([r, np.eye(3), np.eye(3)])
    jaw_change = -.01 * signs[:, None] * issued[..., 6]
    images = np.zeros((*valid.shape, 8, 8, 3), dtype=np.uint8)
    batch = dict(issued_mean=issued, delta_eef=.01 * local,
                 delta_jaw=np.stack([jaw_change, .05 + jaw_change.cumsum(axis=1)], axis=-1),
                 valid=valid, img_before=images, img_after=images)
    vectors = jnp.asarray(world, dtype=jnp.float32)
    weights = jnp.asarray(valid, dtype=jnp.float32)
    has_valid = jnp.asarray(valid.any(axis=1))

    def align(commands):
        return np.asarray(mod.procrustes_so3(jnp.asarray(commands, dtype=jnp.float32),
                                             vectors, weights, has_valid))

    raw = align(issued[..., :3])
    corrected = align(local)
    # Compare the committed contraction with an independent float64 polar factor.
    cross = world[0].T @ issued[0, :, :3]
    u, _, vt = np.linalg.svd(cross)
    polar = u @ np.diag([1., 1., np.sign(np.linalg.det(u @ vt))]) @ vt
    check("raw_kernel_matches_polar_factor", error(raw[0], polar) < JAX_TOL,
          max_matrix_error=error(raw[0], polar), polar_to_truth_deg=angle_deg(polar, r))
    check("gain_aware_kernel_reference", error(corrected, expected) < JAX_TOL,
          max_matrix_error=error(corrected, expected), compound_error_deg=angle_deg(corrected[0], r))
    check("raw_kernel_recovers_compound_rotation", error(raw[0], r) < JAX_TOL,
          max_matrix_error=error(raw[0], r), rotation_error_deg=angle_deg(raw[0], r))

    cfg = mod.ReaderV2Config(model_type="factored_equiv_so3", visual_mode="luminance_flow",
                             cnn_width=24, mlp_width=32, max_chunks=len(means))
    params = mod.init_reader_v2(jax.random.key(5), cfg)
    native = tuple(map(np.asarray, mod.apply_reader_v2(params, cfg, batch)))
    check("unmodified_reader_forward", [x.shape for x in native] == [(3, 3, 3), (3, 3), (3,)]
          and all(np.isfinite(x).all() for x in native),
          input="constant images, noiseless local proprioception, initialization seed 5")
    native_mlp = mod._apply_mlp

    def exact_visual_head(layers, features):
        if layers is params["vis_vec3"]:
            return jnp.concatenate([vectors, jnp.zeros_like(vectors[..., :1])], axis=-1)
        return native_mlp(layers, features)

    # Only replace the learned visual vector/confidence output. All subsequent
    # reader operations, gain/jaw parameters and public decoder run as committed.
    with patch.object(mod, "_apply_mlp", exact_visual_head):
        rot, log_g, jaw_logit = map(np.asarray, mod.apply_reader_v2(params, cfg, batch))
        check("reader_output_contract", rot.shape == (3, 3, 3) and log_g.shape == (3, 3)
              and jaw_logit.shape == (3,) and all(np.isfinite(x).all() for x in (rot, log_g, jaw_logit)),
              shapes=[list(x.shape) for x in (rot, log_g, jaw_logit)])
        check("reader_matches_raw_kernel", error(rot, raw) < JAX_TOL, max_matrix_error=error(rot, raw))
        check("reader_compound_rotation", error(rot[0], r) < JAX_TOL,
              rotation_error_deg=angle_deg(rot[0], r), max_matrix_error=error(rot[0], r))
        check("reader_clean_rotation", error(rot[1], np.eye(3)) < JAX_TOL)
        check("reader_cleared_identity", error(rot[2], np.eye(3)) < JAX_TOL
              and np.array_equal(log_g[2], np.zeros(3)) and jaw_logit[2] > 0)
        REPORT["untrained_heads"] = {"log_gains": log_g.tolist(), "target_log_gains": np.log(gains[:2]).tolist(),
                                     "jaw_logits": jaw_logit.tolist(), "target_jaw_signs": signs[:2].tolist(),
                                     "rows": ["compound", "clean", "cleared"]}
        try:
            dr, dg, ds = map(np.asarray, mod.predict_embodiment_v2(params, cfg, batch))
            ok = (dr.shape == rot.shape and dg.shape == log_g.shape and ds.shape == jaw_logit.shape
                  and np.allclose(dr, rot, atol=JAX_TOL)
                  and np.allclose(dg, np.exp(np.clip(log_g, np.log(.25), np.log(4))), atol=JAX_TOL)
                  and np.array_equal(ds, np.where(jaw_logit >= 0, 1, -1)))
            check("public_so3_decoder", ok, shapes=[list(x.shape) for x in (dr, dg, ds)])
        except Exception as exc:
            check("public_so3_decoder", False, exception=f"{type(exc).__name__}: {exc}")


def main():
    global np
    import numpy as np
    REPORT["numpy"] = np.__version__
    verify_source()
    sys.path.insert(0, str(ROOT / "src"))
    from fluffy_bassoon.xemb import embodiment, equiv_v2

    bursts = embodiment.ACTIVE_PROBE_BURSTS.astype(np.float64)
    if bursts.shape != (3, 5, 7):
        raise ValueError(f"unexpected probe dimensions {bursts.shape}")
    r, g, s = fixture_rotation(), np.array([.65, 1.25, 1.55]), -1
    REPORT["fixture"] = dict(roll_pitch_yaw_deg=[23, -31, 47], xyz_gains=g.tolist(),
                             jaw_sign=s, order="Rz Ry Rx diag(g)", probe_shape=list(bursts.shape))
    geometry(bursts, r, g, s)
    reader_gate(equiv_v2, bursts.mean(axis=1), r, g, s)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        check("execution", False, exception=f"{type(exc).__name__}: {exc}")
    failures = [name for name, passed in REPORT["checks"].items() if not passed]
    REPORT["failures"] = failures
    REPORT["status"] = "ERROR" if "execution" in failures else ("FAIL" if failures else "PASS")
    print(json.dumps(json_safe(REPORT), indent=2, allow_nan=False))
    sys.exit(2 if REPORT["status"] == "ERROR" else int(bool(failures)))
