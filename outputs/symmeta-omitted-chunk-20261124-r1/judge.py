"""Fit an independent flow judge, calibrate on separate tasks, freeze unlabeled predictions."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fluffy_bassoon.xemb.baselines import dense_flow_image_motion

ID = ("case_id", "physical_trace_id", "task", "arm")
SEEDS = (41, 42, 43)
CONSTANTS = np.arange(101) / 100.
SLOPES = np.array([0., .125, .25, .5, 1., 2.])
ROLES = ("fit", "cal", "eval")
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
TASK_IDS = dict(fit=((2, 9), (5, 7), (2, 6), (2, 4)), cal=((6,), (8,), (5,), (6,)),
                eval=((4, 7), (4, 9), (4, 9), (5, 9)),
                reader_fit=((1, 2, 9), (2, 5, 7), (2, 3, 6), (2, 3, 4)))
EXPECTED = {role: sorted(f"{suite}/{tid}" for suite, tasks in zip(SUITES, groups, strict=True)
                         for tid in tasks) for role, groups in TASK_IDS.items()}


def require(check, message):
    if not check:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json(path, obj):
    with path.open("x") as f:
        json.dump(obj, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def save_arrays(path, arrays):
    with path.open("xb") as f:
        np.savez_compressed(f, **arrays)


def load_arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def execution(pin):
    require(platform.system() == "Linux" and os.getenv("CUDA_VISIBLE_DEVICES") == "",
            "Use Linux with CUDA_VISIBLE_DEVICES=''")
    require(os.getenv("OPENBLAS_NUM_THREADS") == "1", "Use one BLAS thread")
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    paths = [HERE / k for k in ("judge.py", "score.py", "SCORE.md")]
    paths.append(ROOT / "src/fluffy_bassoon/xemb/baselines.py")
    revision = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
    require(revision == pin, "Checkout differs from expected Git revision")
    for path in paths:
        binary = subprocess.check_output(["git", "-C", ROOT, "show", f"{pin}:{path.relative_to(ROOT)}"])
        require(hashlib.sha256(binary).hexdigest() == digest(path), f"Unpinned file {path}")
    return dict(revision=pin, host=platform.node(), python=platform.python_version(),
                numpy=np.__version__, opencv=cv2.__version__, opencv_threads=cv2.getNumThreads(),
                opencl=cv2.ocl.useOpenCL(), code_sha256={p.name: digest(p) for p in paths})


def wrap(deg):
    return (np.asarray(deg, dtype=np.float64) + 180.) % 360. - 180.


def unit(vec):
    vec = np.asarray(vec, dtype=np.float64)
    norm = np.linalg.norm(vec, axis=-1)
    ok = np.isfinite(norm) & (norm > 0.)
    return np.divide(vec, norm[..., None], where=ok[..., None], out=np.zeros_like(vec)), ok


def rotate(v, deg):
    angle = np.deg2rad(deg)
    cs, sn = np.cos(angle), np.sin(angle)
    return np.stack([v[..., 0] * cs - v[..., 1] * sn, v[..., 0] * sn + v[..., 1] * cs], axis=-1)


def heading(v):
    return np.rad2deg(np.arctan2(v[..., 1], v[..., 0]))


def row_index(keys, order):
    require(len(keys) == len(set(keys)), "Duplicate case IDs")
    by_id = {k: i for i, k in enumerate(keys)}
    require(set(by_id) == set(order), "Case membership mismatch")
    return np.array([by_id[k] for k in order])


def equal_units(data):
    """Task codes and row weights, summing to 1 inside each task."""
    task, ti = np.unique(data["task"], return_inverse=True)
    ids = list(zip(data["task"], data["physical_trace_id"], data["case_id"], strict=True))
    repetitions = Counter(ids)
    cases = Counter((t, p) for t, p, c in repetitions)
    traces = Counter(t for t, p in cases)
    u = np.array([1. / (traces[t] * cases[t, p] * repetitions[t, p, c]) for t, p, c in ids])
    np.testing.assert_allclose(np.bincount(ti, weights=u), 1.)
    return task, ti, u


def source_arrays(path, role):
    with np.load(path, allow_pickle=False) as z:
        if role != "fit":
            require(not {"true_yaw", "true_gain", "true_sign", "world_cmd"}.intersection(z.files),
                    f"Labeled public {role} data")
        data = {k: z[k] for k in (*ID, "split", "cmd", "valid")}
        n = len(data["case_id"])
        require(all(data[k].shape == (n,) and data[k].dtype.kind == "U" for k in (*ID, "split")),
                "Expected 1D unicode IDs")
        require(set(data["split"]) == {role} and len(set(data["case_id"])) == n, f"Bad {role} IDs")
        require(set(data["task"]) == set(EXPECTED[role]), f"{role} task split changed")
        # The registered panel is physical, with no relabels. Check per-task/arm cells.
        require(len(set(data["physical_trace_id"])) == n and
                Counter(zip(data["task"], data["arm"], strict=True)) ==
                Counter({(t, a): (14 if role == "fit" else 28) for t in EXPECTED[role]
                         for a in ("Panda", "Sawyer")}), f"Incomplete physical {role} panel")
        before, after = z["img_before"], z["img_after"]
        cmd, valid = data["cmd"], data["valid"]
        require(before.ndim == 5 and before.shape[:2] == (n, 3) and before.shape[-1] == 3 and
                before.shape == after.shape and before.dtype == after.dtype == np.uint8, "RGB")
        require(cmd.shape == (n, 3, 2) and np.isfinite(cmd).all() and valid.shape == (n, 3)
                and valid.dtype == bool, "Command or availability contract")
        if role == "fit":
            yaw, gain = z["true_yaw"], z["true_gain"]
            require(yaw.shape == gain.shape == (n,) and np.isfinite(yaw).all() and
                    np.isfinite(gain).all() and (gain > 0.).all(), "Fit label contract")
            world = rotate(cmd, yaw[:, None]) * gain[:, None, None]
            np.testing.assert_allclose(z["world_cmd"], world, atol=2e-6, rtol=2e-6)
            data["true_yaw"] = yaw.astype(float)
        data["flow"] = np.stack([dense_flow_image_motion(a, b, v, method="dis") for a, b, v in
                                 zip(before, after, valid, strict=True)])
    return data


def read_predictions(path):
    fields = {*ID, "split", "seed", "full_yaw", "loo_yaw", "full_valid", "loo_valid",
              "cmd", "valid", "tuned_gain", "tuned_sign"}
    with np.load(path, allow_pickle=False) as z:
        require(set(z.files) == fields, "Reader public array schema changed")
        data = {k: z[k] for k in fields}
    require(set(data["split"]) == {"cal", "eval"}, "Predictions must cover both held-out roles")
    return data


def join_predictions(source, readers, role):
    take = readers["split"] == role
    p = {k: v[take] for k, v in readers.items()}
    n = len(p["case_id"])
    require(len(set(zip(p["case_id"], p["seed"], strict=True))) == n ==
            len(source["case_id"]) * len(SEEDS) and set(p["seed"]) == set(SEEDS),
            "Missing seeds or duplicate predictions")
    order = np.lexsort((p["seed"], p["case_id"]))
    p = {k: v[order] for k, v in p.items()}
    idx = row_index(source["case_id"], p["case_id"])
    for k in (*ID, "cmd", "valid"):
        np.testing.assert_array_equal(p[k], source[k][idx], err_msg=f"Reader/source {k}")
    require(p["full_yaw"].shape == p["full_valid"].shape == (n, 2)
            and p["loo_yaw"].shape == p["loo_valid"].shape == (n, 3, 2)
            and p["full_valid"].dtype == p["loo_valid"].dtype == bool, "Reader axis/validity contract")
    for k in ("full", "loo"):
        require(np.isfinite(p[k + "_yaw"][p[k + "_valid"]]).all(), "Nonfinite declared-valid decoder")
    require(np.isfinite(p["full_yaw"][:, 0]).all(), "Baseline decoded yaw unavailable")
    require(p["tuned_gain"].shape == p["tuned_sign"].shape == (n,) and np.isfinite(p["tuned_gain"]).all()
            and (p["tuned_gain"] > 0.).all() and np.isin(p["tuned_sign"], [-1, 1]).all(), "Gain/sign")
    p["flow"] = source["flow"][idx]
    return p


def labels(path, data):
    z = load_arrays(path)
    ix = row_index(z["case_id"], data["case_id"])
    out = {k: z[k][ix].astype(float) for k in ("true_yaw", "true_gain", "true_sign")}
    require(all(v.shape == ix.shape and np.isfinite(v).all() for v in out.values()) and
            (out["true_gain"] > 0.).all() and np.isin(out["true_sign"], [-1., 1.]).all(),
            "Label contract")
    return out


def fit_terms(fit):
    _, task, u = equal_units(fit)
    x, flow_ok = unit(fit["flow"])
    cmd, cmd_ok = unit(fit["cmd"])
    y = rotate(cmd, fit["true_yaw"][:, None])
    ok = fit["valid"] & flow_ok & cmd_ok
    mass = u[:, None] * ok / np.maximum(1, ok.sum(axis=1))[:, None]
    def cross(b):
        sample = np.einsum("nki,nkj,nk->nij", x, b, mass)
        return np.stack([sample[task == t].sum(axis=0) for t in np.unique(task)])
    return cross(x), cross(y), ok


def projection(terms, counts):
    xx, xy = [np.tensordot(counts, a, axes=(0, 0)) for a in terms[:2]]
    if np.linalg.matrix_rank(xx) < 2:
        return np.zeros((2, 2))
    return np.linalg.solve(xx, xy)


def residuals(data, matrix):
    visual, ok_flow = unit(data["flow"])
    reference, ok_ref = unit(visual @ matrix)
    cmd, ok_cmd = unit(data["cmd"])
    valid = (data["valid"] & ok_flow & ok_ref & ok_cmd)[..., None] & data["loo_valid"]
    # A fold uses its own held-out command/image pair and the decoder which OMITTED that pair.
    off = heading(reference) - heading(cmd)
    errors = np.abs(wrap(data["loo_yaw"] - off[..., None]))
    errors = np.where(valid, errors, np.nan)
    contrast = np.mean(errors[..., 0] - errors[..., 1], axis=1)
    return errors, contrast


def endpoints(data):
    tuned, geom = np.asarray(data["full_yaw"], float).T
    arc = wrap(geom - tuned)
    ok = data["full_valid"].all(axis=1) & np.isfinite(arc) & (np.abs(arc) != 180.)
    return wrap(tuned), np.where(ok, arc, 0.), ok


def mix(tuned, arc, w):
    return wrap(tuned + arc * w)


def choose_weight(c, b, scale, d):
    if b == 0. or scale == 0.:
        return np.full(len(d), c, float)
    return np.clip(c + b * np.where(np.isfinite(d), d, 0.) / scale, 0., 1.)


def cal_cache(cal):
    t, a, _ = endpoints(cal)
    err = wrap(t - cal["true_yaw"])
    losses = np.abs(wrap(err[:, None] + a[:, None] * CONSTANTS))
    ok = cal["full_valid"].all(axis=1)
    raw_err = np.abs(wrap(cal["full_yaw"] - cal["true_yaw"][:, None]))
    unique = ok & (raw_err[:, 0] != raw_err[:, 1])
    win_g = raw_err[:, 1] < raw_err[:, 0]
    return a, err, losses, unique, win_g


def calibrate(cache, d, mass):
    arc, err, const_loss, unique, win_g = cache
    c = float(CONSTANTS[np.argmin(mass @ const_loss)])
    valid = np.isfinite(d)
    m = mass[valid]
    scale = float(np.sqrt(m @ d[valid]**2 / m.sum())) if m.sum() else 0.
    losses = [mass @ np.abs(wrap(err + arc * choose_weight(c, b, scale, d))) for b in SLOPES]
    b = float(SLOPES[np.argmin(losses)])
    hard = int((mass @ (unique & win_g)) > (mass @ (unique & ~win_g)))
    return dict(c=c, b=b, s=scale, hard_default=hard)


def decide(data, d, gate):
    tuned, arc, blend_ok = endpoints(data)
    c, b, s, default = (gate[k] for k in ("c", "b", "s", "hard_default"))
    usable = np.isfinite(d) & blend_ok
    # Preserve exact c as the reported weight for missing fold evidence.
    w = choose_weight(c, b, s, np.where(usable, d, np.nan))
    w = np.where(blend_ok, w, 0.)
    hard = np.where(np.isfinite(d) & (d != 0.), d > 0., default).astype(int)
    hard = np.where(data["full_valid"].all(axis=1), hard, 0)
    geom = np.where(data["full_valid"][:, 1], wrap(data["full_yaw"][:, 1]), tuned)
    return dict(tuned=tuned, geom=geom, midpoint=mix(tuned, arc, .5),
                constant=mix(tuned, arc, c), adaptive=mix(tuned, arc, w),
                hard_yaw=np.where(hard, geom, tuned), hard=hard, weight=w, blend_ok=blend_ok,
                tuned_gain=data["tuned_gain"], tuned_sign=data["tuned_sign"])


def diagnostics(data, matrix, gate, errors, d, pred):
    bad_cmd = rotate(data["cmd"], 90.)
    variants = dict(rotated_command={"cmd": bad_cmd}, next_chunk_pixels={"flow": np.roll(data["flow"], -1, axis=1)},
                    ignored_flow={"flow": np.zeros_like(data["flow"])},
                    ignored_command={"cmd": np.zeros_like(data["cmd"])})
    report = {}
    for name, overwrite in variants.items():
        e1, d1 = residuals({**data, **overwrite}, matrix)
        p1 = decide(data, d1, gate)
        def delta(a, b, angle=False):
            ok = np.isfinite(a) & np.isfinite(b)
            dif = b[ok] - a[ok]
            diff = np.abs(wrap(dif) if angle else dif)
            return dict(n_common=int(ok.sum()), n_validity_changed=int(np.sum(np.isfinite(a) != np.isfinite(b))),
                        n_changed_gt_1e_10=int((diff > 1e-10).sum()),
                        mean_absolute_change=float(diff.mean()) if len(diff) else None)
        report[name] = dict(fold_errors=delta(errors, e1), advantage=delta(d, d1),
                            weight=delta(pred["weight"], p1["weight"]),
                            emitted_yaw=delta(pred["adaptive"], p1["adaptive"], angle=True))
    return report


def assert_disjoint(parts):
    for i, a in enumerate(ROLES):
        for b in ROLES[i + 1:]:
            for field in ("task", "physical_trace_id", "case_id"):
                require(not set(parts[a][field]).intersection(parts[b][field]), f"{a}/{b} overlap in {field}")
    require(not set(EXPECTED["reader_fit"]).intersection(set(EXPECTED["cal"]) | set(EXPECTED["eval"])),
            "Reader-fit overlap")


def freeze(data_dir, reader_path, out, runtime):
    out.mkdir(parents=True, exist_ok=False)
    source_paths = {role: data_dir / f"{role}.npz" for role in ROLES}
    cal_labels = data_dir / "labels-cal.npz"
    input_sha = {str(p.resolve()): digest(p) for p in (*source_paths.values(), cal_labels, reader_path)}
    source_seal = data_dir / "source-seal.json"
    source = json.loads(source_seal.read_bytes())
    receipt_path = reader_path.with_name("prediction-receipt.json")
    receipt = json.loads(receipt_path.read_bytes())
    require(receipt["reader_sha256"] == input_sha[str(reader_path.resolve())] and
            receipt["data_seal"] == digest(source_seal) and receipt["revision"] == source["revision"],
            "Reader predictions and source populations do not share provenance")
    require({(r["seed"], r["model"]) for r in receipt["masks"]} ==
            {(s, m) for s in SEEDS for m in ("tuned", "geom")} and
            all(r["native_full_exact"] and [f["omit"] for f in r["folds"]] == [0, 1, 2] and
                all(f["poison_exact"] for f in r["folds"]) for r in receipt["masks"]),
            "Missing reader omission audit")
    input_sha[str(receipt_path.resolve())] = digest(receipt_path)
    for role, key in (("fit", "judge_fit_tasks"), ("reader_fit", "readerfit"), ("cal", "cal"), ("eval", "eval")):
        member = source[key] if role == "fit" else source["roles"][key]
        require(sorted(member) == EXPECTED[role], f"Unexpected registered {role} tasks")
    for path in (*source_paths.values(), cal_labels):
        require(source["sha256"][path.name] == input_sha[str(path.resolve())], f"Source seal mismatch: {path}")
    input_sha[str(source_seal.resolve())] = digest(source_seal)
    parts = {r: source_arrays(p, r) for r, p in source_paths.items()}
    assert_disjoint(parts)
    reader = read_predictions(reader_path)
    for r in ("cal", "eval"):
        parts[r] = join_predictions(parts[r], reader, r)
    parts["cal"].update(labels(cal_labels, parts["cal"]))
    terms = fit_terms(parts["fit"])
    matrix = projection(terms, np.ones(len(terms[0])))
    require(np.any(matrix), "Fit direction design is rank deficient")
    _, dcal = residuals(parts["cal"], matrix)
    _, _, mass = equal_units(parts["cal"])
    gate = calibrate(cal_cache(parts["cal"]), dcal, mass)
    e, d = residuals(parts["eval"], matrix)
    pred = decide(parts["eval"], d, gate)
    # Independent complex-angle calculation on the real omitted pairs before any test label opens.
    ev = parts["eval"]
    xy = ev["cmd"][..., 0] + 1j * ev["cmd"][..., 1]
    projected = ev["flow"] @ matrix
    visual = projected[..., 0] + 1j * projected[..., 1]
    predicted = xy[..., None] * np.exp(1j * np.deg2rad(ev["loo_yaw"].astype(float)))
    ref = np.abs(np.angle(predicted * np.conj(visual[..., None]), deg=True))
    ok = np.isfinite(e)
    np.testing.assert_allclose(e[ok], ref[ok], rtol=0., atol=1e-9)
    independent_fold_difference = float(np.max(np.abs(e[ok] - ref[ok]), initial=0.))
    for name, role_arrays in parts.items():
        save_arrays(out / f"compact-{name}.npz", role_arrays)
    save_arrays(out / "predictions.npz", {**{k: parts["eval"][k] for k in (*ID, "seed")},
                **pred, "fold_error": e, "advantage": d})
    hashes = {p.name: digest(p) for p in sorted(out.glob("*.npz"))}
    seal = dict(expected_eval_label_sha256=source["sha256"]["labels-eval.npz"],
                runtime=runtime, inputs=input_sha, outputs=hashes, registered_tasks=EXPECTED,
                matrix=matrix.tolist(), gate=gate, complex_check_valid_folds=int(ok.sum()),
                complex_check_max_difference_deg=independent_fold_difference, eligible_fit_chunks=int(terms[2].sum()),
                total_fit_chunks=terms[2].size, calibration_scoreable=int(np.isfinite(dcal).sum()),
                probes=diagnostics(parts["eval"], matrix, gate, e, d, pred))
    write_json(out / "seal.json", seal)
    return digest(out / "seal.json")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pin", required=True)
    p.add_argument("--data", type=Path, required=True, help="directory with role NPZs and labels-cal.npz")
    p.add_argument("--readers", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path, help="new directory")
    args = p.parse_args()
    checksum = freeze(args.data, args.readers, args.out, execution(args.pin))
    print("Frozen pre-label seal SHA-256", checksum, flush=True)


if __name__ == "__main__":
    main()
