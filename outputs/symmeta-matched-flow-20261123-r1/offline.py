"""Fit on the original passive source traces, or evaluate a previously frozen fit."""
import argparse
from collections import Counter
from hashlib import sha256
import gzip
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import cv2
import numpy as np
import scipy
from scipy.optimize import minimize

from fluffy_bassoon.xemb.baselines import dense_flow_image_motion
from fluffy_bassoon.xemb.embodiment import TRAIN_EMBODIMENTS, get_embodiment_spec

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
CAMERA = HERE.parent / "symmeta-camera-command-20261123-r1"
sys.path.insert(0, str(CAMERA))
import score as historical

ROLL_SHA = "be4190952f62bc352e92ec99e00cb407b04acb2e8eb0597964cc964f9d0b8850"
KEYS = ("index", "suite", "task_id", "embodiment")
SOURCE_KEYS = ("index", "split", "arm", "suite", "task_id", "init_state_id", "seed",
               "embodiment", "yaw_deg", "passive_controls", "passive_path")
OOD = {"compositional_ood", "continuous_angle_ood"}
BOOTSTRAP_SEED = 20261123
BOOTSTRAP_REPS = 100000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save(path, obj):
    raw = (json.dumps(obj, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if path.suffix == ".gz":
        raw = gzip.compress(raw, mtime=0)
    with path.open("xb") as f:
        f.write(raw)
    digest = sha256(raw).hexdigest()
    print(f"wrote {path.name} sha256={digest}", flush=True)
    return digest


def readout_features(before, after, valid, dz):
    flow = dense_flow_image_motion(before, after, valid, method="dis")
    features = np.concatenate([flow, np.asarray(dz, dtype=np.float64)[..., None]], axis=-1)
    require(np.isfinite(features).all(), "Nonfinite motion features")
    return features


def unit(x):
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.divide(x, norm, out=np.zeros_like(x), where=norm > 0)


def dot_cross(a, v):
    return np.stack([np.sum(a * v, axis=-1), a[..., 0] * v[..., 1] - a[..., 1] * v[..., 0]], axis=-1)


def fit_projection(x, commands, yaw):
    # No intercept: zero flow and zero vertical displacement represent zero motion.
    rms = np.sqrt(np.mean(x ** 2, axis=(0, 1)))
    require((rms > 0).all(), "Unobserved source feature")
    z = x / rms
    c = np.cos(np.radians(yaw))[:, None]
    s = np.sin(np.radians(yaw))[:, None]
    cx, cy = commands[..., 0], commands[..., 1]
    rotated = np.stack([c * cx - s * cy, s * cx + c * cy], axis=-1)
    initial, _, rank, _ = np.linalg.lstsq(z.reshape(-1, 3), rotated.reshape(-1, 2), rcond=None)
    require(rank == 3, "Source features do not span three dimensions")
    command_units = unit(commands)
    truth = np.concatenate([c, s], axis=-1)

    def loss(flat):
        votes = dot_cross(command_units, unit(z @ flat.reshape(3, 2)))
        sums = votes.sum(axis=1)
        counts = (np.linalg.norm(votes, axis=-1) > 0).sum(axis=1)
        identified = np.linalg.norm(sums, axis=-1) > 8 * np.finfo(float).eps * counts
        estimates = np.where(identified[:, None], unit(sums), [1., 0.])
        return np.mean(1 - (estimates * truth).sum(axis=1))

    fit = minimize(loss, initial.ravel(), method="BFGS", jac="3-point",
                   options={"gtol": 1e-7, "maxiter": 500})
    require(fit.success and np.isfinite(fit.x).all(), f"Source optimizer failed: {fit.message}")
    w = fit.x.reshape(3, 2) / rms[:, None]
    # Check the vectorized training objective against the common evaluation solver.
    pred = []
    for cmd, vector in zip(commands, x @ w, strict=True):
        a, an = historical.unit_vectors(cmd)
        v, vn = historical.unit_vectors(vector)
        pred.append(historical.matched_yaw(a, v, ((an > 0) & (vn > 0)).astype(float))[0])
    source_error = historical.wrap(np.array(pred) - yaw)
    np.testing.assert_allclose(np.mean(1 - np.cos(np.radians(source_error))), fit.fun, atol=1e-12, rtol=0)
    return dict(matrix=w.tolist(), feature_order=["dis_u_px", "dis_v_px", "local_dz_m"],
                intercept=False, source_rms=rms.tolist(), standardized_initial=initial.tolist(),
                initialization="OLS on R(trace yaw) @ issued_mean_xy",
                objective="trace mean 1 - cos(equal-vote yaw estimate - trace yaw label)",
                optimizer="scipy BFGS, 3-point finite differences, gtol=1e-7, maxiter=500",
                iterations=fit.nit, evaluations=fit.nfev, optimizer_message=str(fit.message),
                initial_loss=float(loss(initial.ravel())), final_loss=float(fit.fun),
                source_in_sample_mae_deg=float(np.mean(np.abs(source_error))))


def fit_source(data, out, pin):
    archive_path = data / historical.SCALE / "gpu01/archive/archive.json"
    raw, archive = historical.read_file(archive_path, historical.ARCHIVE_SHA)
    rows = [r for r in json.loads(raw)["trials"] if r["split"] == "train"]
    train = json.loads(historical.SPLITS.read_bytes())["train"]
    expected = {(arm, suite, task, reset, spec.name)
                for arm in ("Panda", "Sawyer") for suite, tasks in train.items() for task in tasks
                for reset in range(4) for spec in TRAIN_EMBODIMENTS}
    require(len(expected) == len(rows) == 1344, "Unexpected source panel")
    inputs, commands, labels, membership = [], [], [], []
    for row in rows:
        key = tuple(row[k] for k in ("arm", "suite", "task_id", "init_state_id", "embodiment"))
        expected.remove(key)
        require(row["yaw_deg"] == get_embodiment_spec(row["embodiment"]).yaw_deg and
                row["passive_controls"] == 15, f"Source label or exposure mismatch: {key}")
        raw, ident = historical.read_file(archive_path.parent / row["passive_path"])
        with np.load(io.BytesIO(raw), allow_pickle=False) as tape:
            valid = tape["valid"]
            require(valid.dtype == bool and np.array_equal(valid, [True] * 3), f"Invalid chunks: {key}")
            xy = tape["issued_mean"][:, :2].astype(float)
            require(xy.shape == (3, 2) and np.isfinite(xy).all(), f"Invalid commands: {key}")
            features = readout_features(tape["img_before"], tape["img_after"], valid, tape["delta_eef"][:, 2])
        inputs.append(features)
        commands.append(xy)
        labels.append(row["yaw_deg"])
        membership.append({**{k: row[k] for k in SOURCE_KEYS}, "sha256": ident["sha256"]})
        if len(inputs) % 192 == 0:
            print(f"read {len(inputs)}/1344 source traces", flush=True)
    require(not expected and len(set(r["passive_path"] for r in membership)) == 1344, "Source identity mismatch")
    result = fit_projection(np.array(inputs), np.array(commands), np.array(labels))
    source_hash = save(out / "source-membership.json.gz", membership)
    save(out / "fit.json", dict(revision=pin, archive=archive, source_count=len(rows),
                                source_chunks=3 * len(rows), by_arm=dict(Counter(r["arm"] for r in rows)),
                                source_membership_sha256=source_hash,
                                ordered_source_tape_set_sha256=historical.set_digest(r["sha256"] for r in membership),
                                **result))


def summarize(rows):
    groups = {"all": rows, "held_out": [r for r in rows if r["regime"] in OOD]}
    for regime in sorted({r["regime"] for r in rows}):
        groups[regime] = [r for r in rows if r["regime"] == regime]
    require(len(groups["held_out"]) == 64 and len(rows) == 96, "Wrong target panel size")
    tasks = sorted({(r["suite"], r["task_id"]) for r in rows})
    require(len(tasks) == 8, "Expected eight target tasks")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(len(tasks), size=(BOOTSTRAP_REPS, len(tasks)))
    results = {}
    fields = ("learned_mae_deg", "linear_mae_deg", "linear_minus_learned_mae_deg")
    for label, group in groups.items():
        means, per_task = [], []
        for suite, tid in tasks:
            part = [r for r in group if (r["suite"], r["task_id"]) == (suite, tid)]
            require(len(part) * 8 == len(group), f"Unbalanced task clusters in {label}")
            err = np.array([[r["learned_abs_error_deg"], r["linear_abs_error_deg"]] for r in part])
            learned, linear = err.mean(axis=0)
            means.append([learned, linear, linear - learned])
            per_task.append(dict(suite=suite, task_id=tid, n=len(part), **dict(zip(fields, means[-1]))))
        matrix = np.array(means)
        lo, hi = np.quantile(matrix[draws].mean(axis=1), [0.025, 0.975], axis=0)
        results[label] = dict(n=len(group), per_task=per_task, **{
            field: dict(mean=mean, ci95=[lower, upper])
            for field, mean, lower, upper in zip(fields, matrix.mean(axis=0), lo, hi, strict=True)})
    return results


def score_target(data, out, pin, fit_sha):
    raw, fit_identity = historical.read_file(out / "fit.json", fit_sha)
    fit = json.loads(raw)
    require(fit["revision"] == pin and fit["archive"]["sha256"] == historical.ARCHIVE_SHA, "Wrong fit pin")
    historical.read_file(out / "source-membership.json.gz", fit["source_membership_sha256"])
    w = np.array(fit["matrix"])
    raw, roll_identity = historical.read_file(CAMERA / "evidence/roll.json.gz", ROLL_SHA)
    roll = json.loads(gzip.decompress(raw))
    checkpoint = roll["sources"]["checkpoint"]
    historical.read_file(data / historical.CHECKPOINT, historical.CHECKPOINT_SHA)
    require(checkpoint["sha256"] == historical.CHECKPOINT_SHA and checkpoint["seed"] == 41
            and checkpoint["archive_sha256"] == fit["archive"]["sha256"], "Unmatched learned reader")
    upright = [r for r in roll["rows"] if r["roll_deg"] == 0]
    cases, panel = historical.official_panel(data)
    predictions, details = [], []
    for case, archived, support in zip(cases, upright, roll["supports"], strict=True):
        identity = {k: case[k] for k in KEYS}
        require(all(identity[k] == archived[k] == support["case"][k] for k in KEYS)
                and case["id"] == archived["id"] == support["case"]["id"], "Target join mismatch")
        batch, tape, indices = historical.load_prefix(case, official=True)
        require(tape["sha256"] == archived["support_sha256"] == support["npz"]["sha256"] and
                archived["checkpoint_sha256"] == checkpoint["sha256"] and
                indices == support["selected_chunk_indices"] and support["controls"] == 15, "Wrong support or checkpoint")
        np.testing.assert_array_equal(support["issued_mean"], batch["issued_mean"][0])
        features = readout_features(batch["img_before"][0], batch["img_after"][0], batch["valid"][0],
                                    batch["delta_eef"][0, :, 2])
        linear_xy = features @ w
        learned_xy = np.array(archived["vectors"]["model_visual_xy"])
        require(learned_xy.shape == linear_xy.shape == (3, 2) and np.isfinite(learned_xy).all(), "Invalid vectors")
        a, an = historical.unit_vectors(batch["issued_mean"][0, :, :2])
        l, ln = historical.unit_vectors(linear_xy)
        v, vn = historical.unit_vectors(learned_xy)
        weights = ((an > 0) & (ln > 0) & (vn > 0)).astype(float)
        linear, linear_diag = historical.matched_yaw(a, l, weights)
        learned, learned_diag = historical.matched_yaw(a, v, weights)
        if np.array_equal(weights, archived["vectors"]["matched_weights"]):
            np.testing.assert_allclose(learned, archived["yaw_deg"]["model_matched"], rtol=0, atol=1e-12)
        row = dict(**identity, support_sha256=tape["sha256"], learned_yaw_deg=learned, linear_yaw_deg=linear)
        predictions.append(row)
        spec = get_embodiment_spec(case["embodiment"])
        require(archived["true_yaw_deg"] == spec.yaw_deg and archived["regime"] == spec.regime, "Wrong label")
        details.append(dict(**row, true_yaw_deg=spec.yaw_deg, regime=spec.regime, support_path=tape["path"],
                            support_seed=case["support_seed"], support_init_state_id=case["support_init_state_id"],
                            query_init_state_id=case["query_init_state_id"], chunk_indices=indices,
                            features=features.tolist(), command_unit=a.tolist(), learned_xy=learned_xy.tolist(),
                            linear_xy=linear_xy.tolist(), weights=weights.tolist(), learned_diagnostics=learned_diag,
                            linear_diagnostics=linear_diag, learned_abs_error_deg=abs(historical.wrap(learned-spec.yaw_deg)),
                            linear_abs_error_deg=abs(historical.wrap(linear-spec.yaw_deg))))
    require([r["index"] for r in predictions] == list(range(96)), "Incomplete target indices")
    digest = historical.set_digest(r["support_sha256"] for r in predictions)
    require(digest == historical.SUPPORT_SET_SHA == roll["support_set_sha256"], "Support set SHA mismatch")
    metrics = summarize(details)
    prediction_hash = save(out / "predictions.json", {"rows": predictions})
    details_hash = save(out / "target-details.json.gz", details)
    save(out / "metrics.json", dict(groups=metrics, bootstrap=dict(cluster="suite/task_id", paired=True,
         draws=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED, interval="percentile 95%, numpy linear quantiles")))
    save(out / "prediction-provenance.json", dict(revision=pin, fit=fit_identity, historical_roll=roll_identity,
         historical_score_revision=roll["revision"], upright_vectors_key="rows[roll_deg=0].vectors.model_visual_xy",
         checkpoint=checkpoint, panel=panel, ordered_support_set_sha256=digest,
         predictions_sha256=prediction_hash, target_details_sha256=details_hash))
    print(json.dumps({k: g["linear_minus_learned_mae_deg"] for k, g in metrics.items()}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("fit", "score"))
    p.add_argument("--pin", required=True)
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--fit-sha256")
    args = p.parse_args()
    revision = subprocess.check_output(["git", "-C", CODE, "rev-parse", "HEAD"], text=True).strip()
    require(revision == args.pin and not subprocess.check_output(["git", "-C", CODE, "status", "--porcelain"]),
            "Require the clean, pinned checkout")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and os.environ.get("JAX_PLATFORMS") == "cpu",
            "Require CPU-only environment")
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    started = time.monotonic()
    print(f"{args.phase} {revision} host={platform.node()} python={platform.python_version()} "
          f"numpy={np.__version__} scipy={scipy.__version__} cv2={cv2.__version__} "
          f"opencv_threads={cv2.getNumThreads()} opencl={cv2.ocl.useOpenCL()} CPU only", flush=True)
    if args.phase == "fit":
        fit_source(args.data_root, args.out, revision)
    else:
        require(args.fit_sha256 is not None, "Scoring requires a frozen fit digest")
        score_target(args.data_root, args.out, revision, args.fit_sha256)
    print(f"{args.phase} completed in {time.monotonic() - started:.2f}s", flush=True)


if __name__ == "__main__":
    main()
