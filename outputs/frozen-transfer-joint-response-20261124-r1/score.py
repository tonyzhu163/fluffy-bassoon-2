"""Source-only response fitting and paired, disjoint one-control forecasts."""
import csv
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

MODELS = ("pd", "pooled", "joint")
WEIGHTS = (.1, 1., 5., 20.)
FIELDS = ("mm", "deg", "joint_mrad")


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False, default=lambda x: x.tolist())+"\n")


def load(path):
    with np.load(path, allow_pickle=False) as a:
        x = np.stack((.05*a["actions"][:, :7], .05*a["v"][:-1]), axis=-1)
        true_rotation = Rotation.from_matrix(a["rot"][1:] @ a["rot"][:-1].transpose(0, 2, 1))
        return dict(x=x, y=np.diff(a["q"], axis=0), jac=a["J"], dp=np.diff(a["xyz"], axis=0),
                    dr=true_rotation.as_rotvec(), hit=a["external_contacts"] > 0,
                    sat=np.any(abs(a["actions"][:, :7]) == 1, axis=1))


def fit(train):
    x = np.concatenate([d["x"] for d in train])
    y = np.concatenate([d["y"] for d in train])
    w = np.sqrt(50.)*.05
    fixed = dict(pd=np.tile([1-(1+w)*np.exp(-w), np.exp(-w)], (7, 1)),
                 geom=np.tile([1., 0.], (7, 1)), hold=np.zeros((7, 2)))
    common = np.linalg.lstsq(x.reshape(-1, 2), y.ravel(), rcond=None)[0]
    fixed["pooled"] = np.tile(common, (7, 1))
    fixed["joint"] = np.array([np.linalg.lstsq(x[:, j], y[:, j], rcond=None)[0] for j in range(7)])
    v = x[..., 1]
    fixed["velocity"] = np.tile([0., np.sum(v*y)/np.sum(v*v)], (7, 1))
    return fixed, np.sqrt(np.mean(x*x, axis=(0, 1)))


def update(center, scale, x, y, weight):
    # Each two-parameter fit is independent; one five-transition support per joint.
    z = (x/scale).transpose(1, 0, 2)
    residual = (y - np.sum(x*center, axis=-1)).T
    zt = z.transpose(0, 2, 1)
    dx = np.linalg.solve(zt @ z + weight*np.eye(2), zt @ residual[..., None])[..., 0]
    return center + dx / scale


def queries(d):
    return range(10, len(d["y"]), 11)


def displace(d, i, center):
    dq = (d["x"][i]*center).sum(axis=-1)
    return dq, d["jac"][i] @ dq


def metric(d, i, center):
    _, p = displace(d, i, center)
    return 1000*np.linalg.norm(p[:3] - d["dp"][i])


def source_selection(train):
    fixed_scores = {name: [] for name in MODELS}
    folds = []
    for k, val in enumerate(train):
        fixed, scale = fit(train[:k] + train[k+1:])
        folds.append((val, fixed, scale))
        for name in MODELS:
            fixed_scores[name].append(np.mean([metric(val, i, fixed[name]) for i in queries(val)]))
    means = {n: np.mean(v) for n, v in fixed_scores.items()}
    selected = min(MODELS, key=means.get)
    history_scores = {}
    for weight in WEIGHTS:
        fold_means = []
        for d, fixed, scale in folds:
            errs = []
            for i in queries(d):
                h = slice(i-5, i)
                local = update(fixed[selected], scale, d["x"][h], d["y"][h], weight)
                errs.append(metric(d, i, local))
            fold_means.append(np.mean(errs))
        history_scores[weight] = np.mean(fold_means)
    return selected, min(WEIGHTS, key=history_scores.get), means, history_scores


def forecast(d, key, arm, fixed, scale, selected, weight):
    rows = []
    for i in queries(d):
        estimators = dict(**fixed, source=fixed[selected])
        for method, h in (("recent", slice(i-5, i)), ("stale", slice(i-10, i-5))):
            estimators[method] = update(fixed[selected], scale, d["x"][h], d["y"][h], weight)
        for name, model in estimators.items():
            dq, p = displace(d, i, model)
            rotation_err = (Rotation.from_rotvec(p[3:]) * Rotation.from_rotvec(d["dr"][i]).inv()).magnitude()
            rows.append(dict(key=key, arm=arm, zero_based_control=i, method=name,
                             mm=metric(d, i, model), deg=np.rad2deg(rotation_err),
                             joint_mrad=1000*np.sqrt(np.mean((dq-d["y"][i])**2)),
                             actual_mm=1000*np.linalg.norm(d["dp"][i]), contact=int(d["hit"][i]),
                             saturated=int(d["sat"][i]), pred_x_m=p[0], pred_y_m=p[1], pred_z_m=p[2]))
    return rows


def compare(rows, keys):
    means = {}
    for key in keys:
        ds = [r for r in rows if r["key"] == key]
        methods = sorted({r["method"] for r in ds})
        means[key] = {name: {f: np.mean([r[f] for r in ds if r["method"] == name])
                             for f in FIELDS} for name in methods}
    overall = {m: {f: np.mean([v[m][f] for v in means.values()]) for f in FIELDS} for m in methods}
    tasks = sorted({k.rsplit("-",1)[0] for k in keys})
    deltas = {}
    for base in ("source", "stale", "pd", "geom", "pooled", "joint", "velocity", "hold"):
        dif = {k: means[k]["recent"]["mm"]-means[k][base]["mm"] for k in keys}
        by_task = {t: [d for k, d in dif.items() if k.rsplit("-", 1)[0] == t] for t in tasks}
        # Enumerate the exact empirical bootstrap of 3 trajectories with replacement per goal.
        groups = [np.array([np.mean(v) for v in itertools.product(x, repeat=len(x))]) for x in by_task.values()]
        sampling = [np.mean(combo) for combo in itertools.product(*groups)]
        deltas[base] = dict(recent_minus_baseline_mm=np.mean(list(dif.values())), by_trajectory=dif,
                            task_means={t: np.mean(v) for t, v in by_task.items()},
                            conditional_percentile95=np.percentile(sampling, [2.5, 97.5]))
    return dict(overall=overall, trajectories=means, differences=deltas)


def main(out):
    keys = [f"{suite}-{layout}" for layout in (0, 2, 3) for suite in ("libero_goal", "libero_spatial")]
    source = {key: load(out/f"{key}-Panda.npz") for key in keys}
    result, rows = {}, []
    for layout in (0, 2, 3):
        suffix = "-"+str(layout)
        training_keys = [key for key in keys if not key.endswith(suffix)]
        train = [source[key] for key in training_keys]
        name, weight, cv, hcv = source_selection(train)
        fixed, scale = fit(train)
        result[layout] = dict(source_keys=training_keys, model=name, weight=weight, source_cv_mm=cv,
                              history_cv_mm=hcv, coefficients=fixed, feature_rms_rad=scale)
    save(out/"source-fits.json", result)
    for key in keys:
        r = result[int(key.rsplit("-",1)[1])]
        for arm in ("Panda", "Kinova3"):
            d = source[key] if arm == "Panda" else load(out/f"{key}-{arm}.npz")
            rows.extend(forecast(d, key, arm, r["coefficients"], r["feature_rms_rad"], r["model"], r["weight"]))
    with (out/"forecasts.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    save(out/"results.json", {a: compare([r for r in rows if r["arm"] == a], keys) for a in ("Panda", "Kinova3")})


if __name__ == "__main__":
    main(Path(sys.argv[1]))
