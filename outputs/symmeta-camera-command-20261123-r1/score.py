"""Score a frozen passive reader and source-calibrated brightness flow on paired camera rolls."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

from fluffy_bassoon.xemb.baselines import compute_dis_flow_maps
from fluffy_bassoon.xemb.embodiment import EVAL_EMBODIMENTS, get_embodiment_spec
from fluffy_bassoon.xemb.equiv_v2 import (
    ReaderV2Config, extract_visual_vectors_v2, init_reader_v2, predict_embodiment_v2,
)
from fluffy_bassoon.xemb.readers import _procrustes_yaw, analytic_visual_yaw, image_motion


CODE = Path(__file__).resolve().parents[2]
SCALE = Path("outputs/libero-xemb-ur5e-scale-20261120-r1")
CAMPAIGN = Path("outputs/xemb-icml-campaign-20261121-r1")
SPLITS = CODE / "outputs/libero-meta-learning-20261117-r1/SPLITS.json"
CHECKPOINT = CAMPAIGN / "gpu03v2/stage1/checkpoints/equiv_v2_dis_dz-passive_attempt1-seed41.msgpack"
AUDIT = SCALE / "gpu01/train/offline-audit.json"
CHECKPOINT_SHA = "c12bace5cb391499d6abd7cc5767f94b1519de7d77940efb0cfcb71ca0f686a4"
AUDIT_SHA = "af0d1933bb23d4c040b0a4488279cdf500f88e69f906e9612523dd2385763456"
ARCHIVE_SHA = "37f062958cb63b962ceefb760f0592fdb120da966369ee308b509e2d82449076"
SUPPORT_SET_SHA = "8729fa68343e2a7a0cc6471ab25e7c707555a866e682a3dc999079a7b9dc3554"
REFERENCE_SET_SHA = "52fa154a6becd989d0bdaa19246694289e9ac33d1fbfe1b299b870c81cf179ae"
NATIVE_ARMS = {"model_native": "ours_equiv_v2_passive", "hs_native": "analytic_visual_hs_passive"}
METHODS = (*NATIVE_ARMS, "model_matched", "hs_matched")
IMAGES = ("frame0_img", "img_before", "img_after")
# Archived yaw_hat is rounded to two decimals. Report the raw residual as well.
PUBLISHED_ROUNDING_DEG = 0.005
FLOAT32_SLACK_DEG = 0.0001


def read_file(path, expected=None):
    raw = path.read_bytes()
    digest = sha256(raw).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {digest}, expected {expected}")
    return raw, {"path": str(path), "sha256": digest}


def set_digest(digests):
    return sha256("".join(f"{d}\n" for d in digests).encode("ascii")).hexdigest()


def frozen_inputs(root):
    import jax
    from flax import serialization

    raw, checkpoint = read_file(root / CHECKPOINT, CHECKPOINT_SHA)
    meta = serialization.msgpack_restore(raw)
    cfg = ReaderV2Config(**meta["reader_v2_config"])
    signature = (meta["tag"], meta["mode"], meta["seed"], meta["archive_sha256"],
                 cfg.model_type, cfg.visual_mode, cfg.use_dz_parallax)
    if signature != ("equiv_v2_dis_dz", "passive_attempt1", 41, ARCHIVE_SHA,
                     "factored_equiv_v2", "dis_lumaflow", True):
        raise ValueError(f"Unexpected checkpoint metadata: {signature}")
    params = serialization.from_state_dict(init_reader_v2(jax.random.key(0), cfg), meta["params"])
    raw, calibration = read_file(root / AUDIT, AUDIT_SHA)
    audit = json.loads(raw)
    if not audit["complete"] or audit["archive_sha256"] != ARCHIVE_SHA:
        raise ValueError("Checkpoint and calibration source archive differ")
    modes = audit["modes"]
    projection = np.array(modes["active_probe"]["analytic_visual_sysid"]["projection"])
    clean = modes["passive_attempt1"]["analytic_visual_sysid_clean_proj"]["projection"]
    if projection.shape != (3, 2) or not np.array_equal(projection, clean):
        raise ValueError("Expected the frozen active-nominal projection used for passive evaluation")
    projection.setflags(write=False)
    checkpoint.update(tag=meta["tag"], mode=meta["mode"], seed=meta["seed"], steps=meta["steps"],
                      config=asdict(cfg), archive_sha256=ARCHIVE_SHA)
    calibration.update(projection=projection.tolist(), archive_sha256=ARCHIVE_SHA,
                       key="modes.active_probe.analytic_visual_sysid.projection")
    return params, cfg, projection, {"checkpoint": checkpoint, "calibration": calibration}


def official_panel(root):
    raw, split_file = read_file(SPLITS)
    test = json.loads(raw)["test"]
    tasks = [(suite, tid) for suite in sorted(test) for tid in test[suite]]
    if len(set(tasks)) != 8 or len(tasks) != 8 or len(EVAL_EMBODIMENTS) != 12:
        raise ValueError("Expected eight official tasks and twelve command transforms")
    cases, references, manifests = [], [], []
    for shard, (suite, tid) in enumerate(tasks):
        source = root / SCALE / f"gpu09test/test_live/shard-{shard}"
        raw, manifest = read_file(source / "support-meta.json")
        support = json.loads(raw)
        manifests.append(manifest)
        attempt = "gpu04s2agoal6" if shard == 3 else "gpu03s2a"
        raw, reference = read_file(root / CAMPAIGN / attempt / f"live/shard-{shard}/live-shard.json")
        published = json.loads(raw)
        references.append(reference)
        if published["shard_idx"] != shard:
            raise ValueError(f"Wrong reference shard in {reference}")
        for local, (spec, tape, ref) in enumerate(zip(EVAL_EMBODIMENTS, support, published["rows"], strict=True)):
            index = len(cases)
            identity = (index, suite, tid, spec.name, spec.regime, spec.yaw_deg, spec.xy_scale, spec.gripper_sign)
            keys = ("index", "suite", "task_id", "embodiment", "regime", "true_yaw", "true_scale", "true_grip")
            basename = f"{local:02d}-{spec.name}-passive.npz"
            declared = Path(tape["passive_npz"])
            if (tuple(ref[k] for k in keys) != identity or tape["embodiment"] != spec.name
                    or declared.parts[-3:] != (f"shard-{shard}", "support", basename)):
                raise ValueError(f"Official row mismatch on shard {shard}, slot {local}")
            cases.append(dict(id=f"{suite}/{tid}/{spec.name}", index=index, shard=shard, suite=suite,
                              task_id=tid, embodiment=spec.name, passive_npz=str(source / "support" / basename),
                              physics=tape["physics"], support_init_state_id=11, query_init_state_id=13,
                              support_seed=31800000 + 10000 * sorted(test).index(suite) + 100 * tid + 11,
                              reference_yaw_deg={m: ref["arms"][arm]["yaw_hat"] for m, arm in NATIVE_ARMS.items()}))
    if set_digest(f["sha256"] for f in references) != REFERENCE_SET_SHA:
        raise ValueError("Published native reference file set differs from the locked set")
    return cases, dict(name="official_UR5e_pi05_passive", splits=split_file, support_manifests=manifests,
                       reference_shards=references, reference_set_sha256=REFERENCE_SET_SHA)


def descriptor_panel(path):
    raw, identity = read_file(path)
    desc = json.loads(raw)
    cases = []
    for row in desc["rows"]:
        case = {k: row[k] for k in ("id", "suite", "task_id", "embodiment")}
        # A named transform is essential for an actual issued-interface truth.
        get_embodiment_spec(case["embodiment"])
        case["passive_npz"] = str((path.parent / row["passive_npz"]).resolve())
        for k in ("index", "shard", "camera_condition", "physics", "chunk_steps", "sha256"):
            if k in row:
                case[k] = row[k]
        cases.append(case)
    ids = [c["id"] for c in cases]
    if not ids or any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("Descriptor needs nonempty, unique string row IDs")
    return cases, dict(name=desc["name"], descriptor=identity)


def load_prefix(case, official):
    import io

    raw, identity = read_file(Path(case["passive_npz"]), case.get("sha256"))
    with np.load(io.BytesIO(raw), allow_pickle=False) as tape:
        valid = tape["valid"]
        if valid.ndim != 1 or valid.dtype != np.bool_:
            raise ValueError("Expected a 1-D boolean valid mask")
        indices = np.flatnonzero(valid)[:3]
        if len(indices) != 3 or (official and not np.array_equal(valid, [True] * 3)):
            raise ValueError(f"{case['id']}: invalid 15-control prefix in {valid}")
        if "chunk_steps" in tape:
            steps = tape["chunk_steps"]
            if "chunk_steps" in case and not np.array_equal(steps, case["chunk_steps"]):
                raise ValueError(f"{case['id']}: descriptor and tape disagree on step counts")
        elif "chunk_steps" in case:
            steps = np.asarray(case["chunk_steps"])
        elif official:
            steps = np.where(valid, 5, 0)  # Published loader contract for these fingerprinted tapes.
        else:
            raise ValueError(f"{case['id']}: descriptor or tape must supply chunk_steps")
        if steps.shape != valid.shape or not np.all(steps[indices] == 5):
            raise ValueError(f"{case['id']}: expected three five-control chunks")
        frame = tape["frame0_img"]
        if frame.ndim != 3 or frame.shape[0] != frame.shape[1] or frame.shape[-1] != 3 or frame.dtype != np.uint8:
            raise ValueError("Expected a square HWC uint8 reset image")
        batch = {"frame0_img": frame[None], "valid": valid[indices][None]}
        fields = {"img_before": frame.shape, "img_after": frame.shape, "issued_mean": (7,),
                  "delta_eef": (3,), "delta_jaw": (2,)}
        for name, shape in fields.items():
            values = tape[name]
            dtype = np.uint8 if name in IMAGES else np.float32
            if values.shape != (len(valid), *shape) or values.dtype != dtype:
                raise ValueError(f"{case['id']}: invalid {name} shape or dtype")
            batch[name] = values[indices][None]
            if not np.isfinite(batch[name]).all():
                raise ValueError(f"{case['id']}: nonfinite {name} in selected support")
    for value in batch.values():
        value.setflags(write=False)
    return batch, identity, indices.tolist()


def roll_pixels(batch, degrees):
    result = dict(batch)
    for key in IMAGES:
        result[key] = np.ascontiguousarray(np.rot90(batch[key], degrees // 90, axes=(-3, -2)))
    return result


def unit_vectors(vectors):
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0), norms[:, 0]


def matched_yaw(commands, vectors, weights):
    dot = np.sum(weights * np.sum(commands * vectors, axis=1))
    cross = np.sum(weights * (commands[:, 0] * vectors[:, 1] - commands[:, 1] * vectors[:, 0]))
    size = float(np.hypot(dot, cross))
    count = int(np.sum(weights))
    identified = size > 8 * np.finfo(np.float64).eps * count
    yaw = float(np.degrees(np.arctan2(cross, dot))) if identified else 0.0
    return yaw, {"resultant_per_vote": size / count if count else 0.0, "identified": bool(identified)}


def score_view(batch, params, cfg, projection, *, check_native=False):
    before, after, valid = (batch[k] for k in ("img_before", "img_after", "valid"))
    motion = image_motion(before, after, valid)
    dz = batch["delta_eef"][..., 2]
    issued = batch["issued_mean"]
    # The selected commands all span a full five steps: issued_exposure == issued_mean.
    hs_native = float(analytic_visual_yaw(issued, motion, dz, valid, projection)[0])
    hs_xy = ((motion - dz[..., None] * projection[2]) @ np.linalg.pinv(projection[:2], rcond=1e-5))[0]
    inputs = dict(batch, dis_flow=compute_dis_flow_maps(before, after, valid))
    model_xy, confidence = extract_visual_vectors_v2(params, cfg, inputs)
    c, s = _procrustes_yaw(issued[..., :2], model_xy, confidence, np.any(valid, axis=1))
    model_native = float(np.degrees(np.arctan2(np.asarray(s), np.asarray(c)))[0])
    direct_yaw = None
    if check_native:
        direct_yaw = float(predict_embodiment_v2(params, cfg, inputs)[0][0])
    model_xy = np.asarray(model_xy)[0]
    a, a_size = unit_vectors(issued[0, :, :2])
    h, h_size = unit_vectors(hs_xy)
    v, v_size = unit_vectors(model_xy)
    weights = ((a_size > 0) & (h_size > 0) & (v_size > 0)).astype(float)
    model_matched, model_info = matched_yaw(a, v, weights)
    hs_matched, hs_info = matched_yaw(a, h, weights)
    predictions = dict(model_native=model_native, hs_native=hs_native,
                       model_matched=model_matched, hs_matched=hs_matched)
    details = dict(hs_image_motion_px=motion[0].tolist(), hs_projected_xy=hs_xy.tolist(),
                   model_visual_xy=model_xy.tolist(), model_native_weights=np.asarray(confidence)[0].tolist(),
                   matched_weights=weights.tolist(), model_matched=model_info, hs_matched=hs_info)
    return predictions, details, direct_yaw


def wrap(degrees):
    return (degrees + 180.0) % 360.0 - 180.0


def score_case(case, batch, params, cfg, projection, rolls):
    spec = get_embodiment_spec(case["embodiment"])
    rows = []
    parity = {}
    for angle in rolls:
        rotated = roll_pixels(batch, angle)
        pred, vectors, direct = score_view(rotated, params, cfg, projection, check_native=(angle == 0))
        inverse = None
        if angle == 0:
            zero = pred
            parity["direct_native_model_yaw_deg"] = direct
            parity["direct_model_abs_diff_deg"] = abs(wrap(pred["model_native"] - direct))
            for m, archived in case.get("reference_yaw_deg", {}).items():
                diff = abs(wrap(pred[m] - archived))
                parity[m] = dict(published_yaw_deg=archived, abs_diff_deg=diff,
                                 within_rounding=diff <= PUBLISHED_ROUNDING_DEG + FLOAT32_SLACK_DEG)
        else:
            restored = roll_pixels(rotated, -angle)
            if not all(np.array_equal(restored[k], batch[k]) for k in IMAGES):
                raise ValueError(f"{case['id']}: inverse pixel check failed at {angle}")
            inverse_pred = score_view(restored, params, cfg, projection)[0]
            inverse = dict(pixels_equal=True, yaw_deg=inverse_pred,
                           abs_diff_from_zero_deg={m: abs(wrap(inverse_pred[m] - zero[m])) for m in METHODS})
        err = {m: abs(wrap(p - spec.yaw_deg)) for m, p in pred.items()}
        rows.append(dict(id=case["id"], index=case.get("index"), suite=case["suite"], task_id=case["task_id"],
                         embodiment=spec.name, regime=spec.regime, true_yaw_deg=spec.yaw_deg,
                         true_scale=spec.xy_scale, true_grip=spec.gripper_sign, roll_deg=angle,
                         yaw_deg=pred, abs_error_deg=err,
                         shift_from_zero_deg={m: wrap(pred[m] - zero[m]) for m in METHODS},
                         error_change_from_zero_deg={m: err[m] - abs(wrap(zero[m] - spec.yaw_deg)) for m in METHODS},
                         matched_model_minus_hs_error_deg=err["model_matched"] - err["hs_matched"],
                         vectors=vectors, inverse=inverse))
    return rows, parity


def summarize(rows):
    groups = {}
    for row in rows:
        task = f"{row['suite']}/{row['task_id']}"
        labels = ["all", row["regime"], task]
        if row["regime"] in ("compositional_ood", "continuous_angle_ood"):
            labels.extend(["ood", f"{task}/ood"])
        for label in labels:
            groups.setdefault((label, row["roll_deg"]), []).append(row)
    summary = {}
    for (label, angle), subset in groups.items():
        info = dict(n=len(subset))
        for method in METHODS:
            shift = np.array([r["shift_from_zero_deg"][method] for r in subset])
            info[method] = dict(mae_deg=float(np.mean([r["abs_error_deg"][method] for r in subset])),
                                mean_error_change_deg=float(np.mean([r["error_change_from_zero_deg"][method]
                                                                     for r in subset])),
                                mean_signed_shift_deg=float(shift.mean()), mean_abs_shift_deg=float(abs(shift).mean()))
        info["mean_matched_model_minus_hs_error_deg"] = float(np.mean([r["matched_model_minus_hs_error_deg"]
                                                                       for r in subset]))
        for method in ("model_matched", "hs_matched"):
            info[method]["unidentified_count"] = sum(not r["vectors"][method]["identified"] for r in subset)
        summary.setdefault(label, {})[str(angle)] = info
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/mowenjie/fluffy-bassoon-2"),
                        help="Shared root containing the pinned artifacts (independent of the code checkout).")
    parser.add_argument("--descriptor", type=Path, help="Score the specified replay panel; schema in SCORING.md.")
    parser.add_argument("--rolls", type=int, nargs="+", default=[0, 90, -90], help="Image rolls in degrees, zero first.")
    parser.add_argument("--output", required=True, type=Path, help="Fresh JSON report path.")
    args = parser.parse_args()
    if (not args.rolls or args.rolls[0] != 0 or len(set(args.rolls)) != len(args.rolls)
            or not set(args.rolls) <= {0, 90, -90}):
        parser.error("--rolls must start with 0 and use distinct angles from 0, 90 and -90")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import cv2
    import flax
    import jax

    started = time.monotonic()
    cv2.setNumThreads(1)
    if jax.default_backend() != "cpu":
        raise RuntimeError("CPU JAX required")
    params, cfg, projection, sources = frozen_inputs(args.data_root)
    official = args.descriptor is None
    cases, panel = official_panel(args.data_root) if official else descriptor_panel(args.descriptor)
    supports, rows, parity = [], [], {}
    for case in cases:
        batch, identity, indices = load_prefix(case, official)
        case_rows, parity[case["id"]] = score_case(case, batch, params, cfg, projection, args.rolls)
        for row in case_rows:
            row.update(support_sha256=identity["sha256"], checkpoint_sha256=CHECKPOINT_SHA)
        rows.extend(case_rows)
        supports.append(dict(case=case, npz=identity, selected_chunk_indices=indices, controls=15,
                             issued_mean=batch["issued_mean"][0].tolist(), delta_eef=batch["delta_eef"][0].tolist()))
        if len(supports) % 12 == 0:
            print(f"Scored {len(supports)}/{len(cases)} supports", flush=True)
    digest = set_digest(s["npz"]["sha256"] for s in supports)
    if official and digest != SUPPORT_SET_SHA:
        raise ValueError(f"Official passive NPZ set differs from the frozen set: {digest}")
    checks = {"direct_native_max_abs_diff_deg": max(p["direct_model_abs_diff_deg"] for p in parity.values()),
              "inverse_max_abs_diff_deg": max((d for r in rows if r["inverse"] for d in
                                               r["inverse"]["abs_diff_from_zero_deg"].values()), default=0.0)}
    if official:
        for method in NATIVE_ARMS:
            records = [p[method] for p in parity.values()]
            checks[method] = dict(n=len(records), within_rounding=sum(p["within_rounding"] for p in records),
                                  max_abs_diff_deg=max(p["abs_diff_deg"] for p in records))
    checks["passed"] = (checks["direct_native_max_abs_diff_deg"] <= FLOAT32_SLACK_DEG
                        and checks["inverse_max_abs_diff_deg"] <= FLOAT32_SLACK_DEG
                        and all(checks[m]["within_rounding"] == len(cases) for m in NATIVE_ARMS if official))
    code_paths = [Path(__file__).resolve()] + [CODE / f"src/fluffy_bassoon/xemb/{module}.py"
                                               for module in ("readers", "baselines", "embodiment", "equiv_v2")]
    revision = subprocess.check_output(["git", "-C", str(CODE), "rev-parse", "HEAD"], text=True).strip()
    report = dict(schema_version=1, panel=panel, support_set_sha256=digest, rolls_deg=args.rolls,
                  sources=sources, revision=revision, code_files=[read_file(p)[1] for p in code_paths],
                  runtime=dict(python=platform.python_version(), jax=jax.__version__, backend=jax.default_backend(),
                               numpy=np.__version__, flax=flax.__version__, opencv=cv2.__version__,
                               seconds=time.monotonic() - started),
                  supports=supports, rows=rows, parity_by_id=parity, checks=checks, summary=summarize(rows))
    encoded = json.dumps(report, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(encoded + "\n")
    print(json.dumps(checks), flush=True)
    if not checks["passed"]:
        raise SystemExit(f"Parity check failed; inspect per-query residuals in {args.output}")


if __name__ == "__main__":
    main()
