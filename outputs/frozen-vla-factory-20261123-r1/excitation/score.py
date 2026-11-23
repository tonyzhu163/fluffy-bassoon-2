"""Score fresh simulated supports and archive-only passive supports separately."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from inputs import BASE, HERE, OUTPUTS, PI0, RELEASE, SEED, WORLD, load, rot, spread, supports

sys.path.insert(0, str(OUTPUTS.parent / "src"))
from fluffy_bassoon.xemb.embodiment import get_embodiment_spec
from fluffy_bassoon.xemb.wider_family import operator_error


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def features(x):
    lag = np.vstack([np.zeros(2), x[:-1]])
    return np.stack((x, lag), axis=-1)


def calibrate():
    src = OUTPUTS / "libero-xemb-joint-sysid-20261119-r1/cpu01"
    paths = [src / f"sensors-{i:02d}.npz" for i in (0, 7, 14, 21)]
    xx, yy = [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as a:
            np.testing.assert_allclose(a["issued"].reshape(3, 5, 7).mean(1), a["issued_mean"])
            xx.append(features(a["issued_mean"][:, :2]).reshape(-1, 2))
            yy.append(a["delta_eef"][:, :2].ravel())
    x, y = np.concatenate(xx), np.concatenate(yy)
    coef = np.linalg.lstsq(x, y, rcond=None)[0]
    sigma = np.sqrt(np.mean((x @ coef - y) ** 2))
    return coef, sigma, {p.name: sha(p) for p in paths}


def estimates(cmd, motion, coef, sigma):
    x = features(cmd) @ coef
    total = np.sum(x * x)
    dot = np.sum(x * motion) / total
    cross = (x[:, 0] @ motion[:, 1] - x[:, 1] @ motion[:, 0]) / total
    sim = np.array([[dot, -cross], [cross, dot]])
    cov = x.T @ x + sigma**2 * np.eye(2)
    ridge = sim + np.linalg.solve(cov, x.T @ (motion - x @ sim.T)).T
    return {"full": ridge, "similarity": sim, "minnorm": np.linalg.lstsq(x, motion, rcond=None)[0].T,
            "identity": np.eye(2)}


def scores(cmd, motion, coef, sigma, truth):
    pred = estimates(cmd, motion, coef, sigma)
    return {model: dict(forward=float(np.linalg.norm(m - truth) / np.linalg.norm(truth)),
                        residual=float(operator_error(truth, m)))
            for model, m in pred.items()}


def paired(rows, left, right):
    # Each row contains both contrasts on a single template, map and noise realization.
    deltas = defaultdict(list)
    for r in rows:
        deltas[r["task"]].append(r["scores"][left] - r["scores"][right])
    by_task = np.array([np.mean(deltas[t]) for t in sorted(deltas)])
    assert len(by_task) == 8
    sample = np.random.default_rng(SEED + 1).choice(by_task, size=(10000, 8)).mean(axis=1)
    return {"mean": float(by_task.mean()), "task_boot_95": np.quantile(sample, [.025, .975]).tolist(),
            "task_deltas": dict(zip(sorted(deltas), by_task.tolist()))}


def summarize_groups(rows):
    result = {}
    for group in ["all", "motor", "amplitude", "template", "sign"]:
        values = ["all"] if group == "all" else sorted({r[group] for r in rows})
        for v in values:
            selected = rows if group == "all" else [r for r in rows if r[group] == v]
            saved = dict(n_comparisons=len(selected), scores={})
            for k in selected[0]["scores"]:
                values_k = np.array([r["scores"][k] for r in selected])
                saved["scores"][k] = dict(mean=float(values_k.mean()), median=float(np.median(values_k)),
                                          p95=float(np.quantile(values_k, .95)))
            saved["differences"] = {
                name: paired(selected, "two/full/forward", right)
                for name, right in [("two_minus_one", "one/full/forward"),
                                    ("two_minus_passive", "passive/full/forward"),
                                    ("full_minus_similarity_two", "two/similarity/forward")]}
            result[group if group == "all" else f"{group}={v}"] = saved
    return result


def inspect_trials(collected, archives, coef, sigma):
    source_by_sha = {r["tape_sha"]: r for r in archives}
    assert collected["base"] == BASE and collected["release"] == RELEASE
    assert len(collected["tasks"]) == 8 and len(collected["rows"]) == 384
    grouped = defaultdict(dict)
    for r in collected["rows"]:
        source = source_by_sha[r["tape_sha"]]
        assert (source["task"], source["motor"], source["index"]) == (r["task"], r["motor"], r["index"])
        key = r["task"], r["motor"], r["template"], r["sign"], r["amplitude"]
        assert r["design"] not in grouped[key]
        grouped[key][r["design"]] = r
    rng = np.random.default_rng(SEED + 2)
    banks = {"observed": [], "matched_noise": []}
    coverage = defaultdict(list)
    for key, triplet in sorted(grouped.items()):
        task, motor, template, sign, amplitude = key
        assert set(triplet) == {"one", "passive", "two"}
        truth = np.array(triplet["one"]["truth"])
        np.testing.assert_allclose(np.linalg.svd(truth, compute_uv=False), [1.45, .70])
        assert np.sign(np.linalg.det(truth)) == sign
        # Common additive endpoint noise. Adjacent displacement errors share an endpoint.
        jitter = np.diff(rng.normal(0, sigma / np.sqrt(2), size=(4, 2)), axis=0)
        for noise, perturb in (("observed", 0), ("matched_noise", jitter)):
            measured = {}
            for name, r in triplet.items():
                np.testing.assert_array_equal(truth, r["truth"])
                x, y = np.array(r["x"]), np.array(r["y"])
                assert np.isfinite(y).all() and x.shape == y.shape == (3, 2)
                np.testing.assert_allclose(np.linalg.norm(x, axis=1), amplitude)
                np.testing.assert_allclose(x[0], triplet["one"]["x"][0])
                if noise == "observed":
                    coverage[f"{name}/{template}"].append(spread(x))
                metrics = scores(x, y + perturb, coef, sigma, truth)
                for model, fields in metrics.items():
                    for field, val in fields.items():
                        measured[f"{name}/{model}/{field}"] = val
            banks[noise].append(dict(task=task, motor=motor, template=template, sign=sign,
                                     amplitude=amplitude, scores=measured))
    assert len(grouped) == 128
    return ({k: summarize_groups(rows) for k, rows in banks.items()},
            {k: float(np.median(v)) for k, v in coverage.items()})


def recorded(rows, coef, sigma):
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["task"], r["motor"])].append(r)
    buckets = defaultdict(list)
    for (_, motor), group in grouped.items():
        assert len(group) == 12
        for i, r in enumerate(sorted(group, key=lambda r: spread(r["x"]))):
            spec = get_embodiment_spec(r["label"])
            truth = spec.xy_scale * rot(np.deg2rad(spec.yaw_deg))
            err = scores(r["x"], r["y"], coef, sigma, truth)
            data = dict(task=r["task"], scores={f"{m}/{f}": v for m, d in err.items()
                                                for f, v in d.items()}, spread=spread(r["x"]))
            for k in ("all", motor, "lower_half" if i < 6 else "upper_half"):
                buckets[k].append(data)
    return {k: dict(n=len(v), median_spread=float(np.median([r["spread"] for r in v])),
                    mean={f: float(np.mean([r["scores"][f] for r in v])) for f in v[0]["scores"]},
                    full_minus_similarity=paired(v, "full/forward", "similarity/forward"))
            for k, v in buckets.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supports", type=Path, default=HERE / "supports.json.gz")
    parser.add_argument("--out", type=Path, default=HERE / "metrics.json")
    args = parser.parse_args()
    coef, sigma, source_shas = calibrate()
    archives, _ = supports()
    collected = load(args.supports)
    sims, coverage = inspect_trials(collected, archives, coef, sigma)
    metrics = dict(base=BASE, release=RELEASE, seed=SEED, gpu_hours=0, fresh_supports=384,
                   support_controls_each=15, physical_controls=5760, archive_supports=192,
                   task_clusters=8, map_singular_values=[1.45, .70], unit_command_spread=coverage,
                   source_lag0_lag1_m_per_cmd=coef.tolist(), source_residual_rms_m=float(sigma),
                   source_npz_sha256=source_shas, input_sha256={p.name: sha(p) for p in (WORLD, PI0, args.supports)},
                   archived_passive=recorded(archives, coef, sigma), simulated=sims)
    args.out.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
