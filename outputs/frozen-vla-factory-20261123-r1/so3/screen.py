"""CPU-only, offline compound-axis identification on recorded frozen π₀.₅ motions."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parents[1]
SOURCE = OUTPUTS / "libero-xemb-friction-dev-20261119-r1" / "cpu01"
TARGET = OUTPUTS / "symmeta-camera-command-20261123-r1" / "evidence" / "world.json.gz"
LABELS = OUTPUTS / "xemb-live-wave-20261121-r1" / "evidence" / "pi05-rows.jsonl"
PIN = "757a44cd4374833cad790852df8b87ddf412e638"
SOURCE_SEED, TEST_SEED, BOOT_SEED = 231101, 231102, 231103
DRAWS = 32
GRID = (0., 0.001, 0.01, 0.1, 1., 10.)
I = np.eye(3)


def transpose(a):
    return np.swapaxes(a, -1, -2)


def proper(a):
    u, _, vh = np.linalg.svd(a)
    u[..., :, -1] *= np.linalg.det(u @ vh)[..., None]
    return u @ vh


def axis_rotation(axis, angles):
    """Right-handed rotation for column vectors."""
    rot = np.broadcast_to(I, (*angles.shape, 3, 3)).copy()
    j, k = (axis + 1) % 3, (axis + 2) % 3
    rot[..., j, j] = rot[..., k, k] = np.cos(angles)
    rot[..., k, j] = np.sin(angles)
    rot[..., j, k] = -np.sin(angles)
    return rot


def read_source():
    x, y, tasks = [], [], []
    for path in sorted(SOURCE.glob("*-default-ordinary.npz")):
        with np.load(path) as tape:
            commands = tape["issued"][:, :3]
            position = tape["robot0_eef_pos"].astype(float)
        assert len(position) == len(commands) + 1
        for start in range(0, len(commands) - 14, 15):
            x.append(commands[start:start+15].reshape(3, 5, 3).mean(1))
            y.append(np.diff(position[start:start+16:5], axis=0))
            tasks.append(path.name.removesuffix("-default-ordinary.npz"))
    assert len(set(tasks)) == 8
    return np.stack(x), np.stack(y), np.array(tasks)


def read_target():
    with gzip.open(TARGET, "rt") as f:
        record = json.load(f)
    assert record["status"] == "complete" and record["method"] == "same_archived_passive_joint_fk"
    rows = sorted(record["rows"], key=lambda r: r["shard"] * 12 + r["index"] % 12)
    x = np.array([r["issued_mean"] for r in rows])[:, :, :3]
    y = np.array([r["delta_world_m"] for r in rows])
    tasks = np.array([f"{r['suite']}-{r['task_id']}" for r in rows])
    keys = [(r["suite"], r["task_id"], r["embodiment"]) for r in rows]
    assert x.shape == y.shape == (96, 3, 3) and len(set(keys)) == 96
    assert np.all(np.unique(tasks, return_counts=True)[1] == 12)
    return x, y, tasks, keys


def native_labels(keys):
    records = [json.loads(line) for line in LABELS.read_text().splitlines()]
    by_key = {(r["suite"], r["task_id"], r["embodiment"]): r for r in records}
    assert set(keys) == set(by_key)
    yaw = np.array([by_key[k]["true_yaw"] for k in keys])
    scale = np.array([by_key[k]["true_scale"] for k in keys])
    return axis_rotation(2, np.deg2rad(yaw)), scale


def draw_frames(n, seed, compound):
    rng = np.random.default_rng(seed)
    size = (n, DRAWS)
    yaw = rng.uniform(-np.pi, np.pi, size)
    if compound:
        angles = np.deg2rad(rng.uniform(15, 50, (2, *size))) * rng.choice([-1, 1], (2, *size))
        q = axis_rotation(2, yaw) @ axis_rotation(1, angles[0]) @ axis_rotation(0, angles[1])
    else:
        axes = rng.integers(0, 3, size)
        q = np.broadcast_to(I, (*size, 3, 3)).copy()
        for axis in range(3):
            q[axes == axis] = axis_rotation(axis, yaw[axes == axis])
    gain = np.exp(rng.uniform(np.log(0.65), np.log(1.4), (*size, 3)))
    return q, gain


def support_inputs(x, y, q, gain, meters):
    # Row notation: (u @ Q) / g, then (u_new * g) @ Q.T restores u.
    issued = (x[:, None] @ q) / gain[..., None, :]
    assert np.allclose((issued * gain[..., None, :]) @ transpose(q), x[:, None])
    observed = np.broadcast_to(y[:, None], issued.shape) / meters
    return issued.reshape(-1, 3, 3), observed.reshape(-1, 3, 3)


def fit_head(name, x, y, weight=0.):
    """Only the three XYZ command and motion pairs reach the estimator."""
    cross = transpose(y) @ x
    energy = np.square(x).sum(axis=1)
    strength = weight * energy.mean(axis=1)
    if name == "least_squares":
        return transpose(np.linalg.pinv(x) @ y)
    raw = proper(cross)
    projected = np.sum(x * (y @ raw), axis=1)
    scalar = np.clip(projected.sum(1) / energy.sum(1), .25, 4.)
    raw_map = raw * scalar[:, None, None]
    if name == "raw_polar":
        return raw_map
    if name in ("ridge_zero", "ridge_polar"):
        if weight == 0:
            return transpose(np.linalg.pinv(x) @ y)
        anchor = raw_map if name == "ridge_polar" else np.zeros_like(raw)
        gram = transpose(x) @ x + strength[:, None, None] * I
        rhs = cross + strength[:, None, None] * anchor
        return transpose(np.linalg.solve(gram, transpose(rhs)))
    assert name == "gain_so3"
    # Alternating proper Kabsch and separable box-constrained gain minimization.
    # The source-selected penalty is lambda * ||X||²/3 * ||g - 1||².
    gains = np.ones_like(energy)
    for _ in range(96):
        rot = proper(cross * gains[:, None, :])
        numerator = np.sum(x * (y @ rot), axis=1) + strength[:, None]
        gains = np.clip(numerator / np.maximum(energy + strength[:, None], 1e-15), .25, 4.)
    rot = proper(cross * gains[:, None, :])
    return rot * gains[:, None, :]


def angles(est, truth):
    cosine = ((proper(est) * truth).sum(axis=(-1, -2)) - 1) / 2
    return np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))


def scores(est, rot, gains):
    truth = rot * gains[:, None, :]
    angle = angles(est, rot)
    # Same inverse singular-value limits for all heads, as in the native gain adapter.
    u, s, vh = np.linalg.svd(est)
    inverse = (transpose(vh) / np.clip(s[:, None, :], .25, 4.)) @ transpose(u)
    inverse_rms = np.linalg.norm(truth @ inverse - I, axis=(-1, -2)) / np.sqrt(3)
    return angle, inverse_rms


def task_means(values, case_tasks, mask):
    # Even partially in-box supports get one vote within their task.
    count = mask.reshape(-1, DRAWS).sum(1)
    case_mean = (values * mask).reshape(-1, DRAWS).sum(1) / np.maximum(count, 1)
    return np.array([case_mean[(count > 0) & (case_tasks == t)].mean() for t in np.unique(case_tasks)])


def tune(x, y, tasks):
    # One scalar physical units conversion, equally weighting source episodes.
    per_task_scale = [np.sum(x[tasks == t] * y[tasks == t]) / np.sum(x[tasks == t]**2)
                      for t in np.unique(tasks)]
    meters = float(np.mean(per_task_scale))
    q, gain = draw_frames(len(x), SOURCE_SEED, compound=False)
    xx, yy = support_inputs(x, y, q, gain, meters)
    params, curves = {}, {}
    for name in ("gain_so3", "raw_polar", "least_squares", "ridge_zero", "ridge_polar"):
        curve = {}
        for w in (GRID if name in ("gain_so3", "ridge_zero", "ridge_polar") else (0.,)):
            err = angles(fit_head(name, xx, yy, w), q.reshape(-1, 3, 3)).reshape(len(x), DRAWS)
            curve[w] = float(np.mean([err[tasks == t].mean() for t in np.unique(tasks)]))
        params[name] = min(curve, key=curve.get)
        curves[name] = curve
    return meters, params, curves


def summarize(all_scores, tasks, mask, boot_ids):
    summary, target_tasks = {}, np.unique(tasks)
    for name, (angle, residual) in all_scores.items():
        per_task = task_means(angle, tasks, mask)
        delta = task_means(angle - all_scores["gain_so3"][0], tasks, mask)
        delta_rms = task_means(residual - all_scores["gain_so3"][1], tasks, mask)
        summary[name] = {
            "mean_deg": per_task.mean(), "pooled_p90_deg": np.percentile(angle[mask], 90),
            "mean_inverse_rms": task_means(residual, tasks, mask).mean(),
            "control_minus_gain_deg": delta.mean(),
            "paired_task_bootstrap95_deg": np.percentile(delta[boot_ids].mean(1), [2.5, 97.5]).tolist(),
            "control_minus_gain_inverse_rms": delta_rms.mean(),
            "paired_task_bootstrap95_inverse_rms": np.percentile(delta_rms[boot_ids].mean(1), [2.5, 97.5]).tolist(),
            "gain_win_fraction": task_means(all_scores["gain_so3"][0] < angle, tasks, mask).mean(),
            "mean_deg_by_task": dict(zip(target_tasks, per_task)),
        }
    return summary


def checks():
    rng = np.random.default_rng(67)
    x = rng.normal(size=(8, 3, 3))
    rot = axis_rotation(0, np.linspace(-.6, 1.2, 8)) @ axis_rotation(1, np.linspace(.9, -.6, 8))
    g = np.array([.8, 1.4, .7])
    true_map = rot * g
    clean = x @ transpose(true_map)
    assert np.allclose(fit_head("least_squares", x, clean), true_map)
    # Full-rank orthogonal excitation separates the three gains exactly.
    balanced = np.tile(I, (8, 1, 1))
    assert np.allclose(fit_head("gain_so3", balanced, balanced @ transpose(true_map)), true_map)
    angle, inverse_rms = scores(true_map, rot, np.tile(g, (8, 1)))
    assert angle.max() < 1e-5 and inverse_rms.max() < 1e-12


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "metrics.json")
    args = parser.parse_args()
    checks()
    xs, ys, source_tasks = read_source()
    meters, params, curves = tune(xs, ys, source_tasks)
    x, y, task_names, keys = read_target()
    assert not set(task_names).intersection(source_tasks)
    rank_ratio = 1 / np.linalg.cond(x)
    weak = np.zeros(len(x), bool)
    for task in np.unique(task_names):
        ids = np.flatnonzero(task_names == task)
        weak[ids[np.argsort(rank_ratio[ids])[:len(ids)//2]]] = True
    task = np.repeat(task_names, DRAWS)
    masks = {"all": np.ones(len(task), bool), "weaker": np.repeat(weak, DRAWS),
             "stronger": np.repeat(~weak, DRAWS)}
    q, random_gain = draw_frames(len(x), TEST_SEED, compound=True)
    boot = np.random.default_rng(BOOT_SEED).integers(8, size=(10000, 8))
    report = {
        "revision": PIN, "resource": "local CPU, 0 GPUh", "numpy": np.__version__,
        "source_tasks_windows": {t: int(sum(source_tasks == t)) for t in np.unique(source_tasks)},
        "source_mean_m_per_command": meters, "source_geodesic_grid_deg": curves,
        "source_selected_shrinkage": params, "seeds_source_target_boot": [SOURCE_SEED, TEST_SEED, BOOT_SEED],
        "draws_per_trace": DRAWS, "target_supports": len(x), "target_tasks": len(np.unique(task_names)),
        "original_s3_s1_quantiles_min_q1_median_q3_max": np.quantile(rank_ratio, [0,.25,.5,.75,1]).tolist(),
        "median_original_s3_s1_by_half": {"weaker": np.median(rank_ratio[weak]), "stronger": np.median(rank_ratio[~weak])},
        "results": {},
    }
    for condition, gain in [("anisotropic", random_gain), ("isotropic", np.ones_like(random_gain))]:
        xx, yy = support_inputs(x, y, q, gain, meters)
        estimates = {name: fit_head(name, xx, yy, w) for name, w in params.items()}
        # Native target labels enter only here, after the complete set of estimates.
        r_native, g_native = native_labels(keys)
        truth = (r_native[:, None] @ q).reshape(-1, 3, 3)
        g_total = (gain * g_native[:, None, None]).reshape(-1, 3)
        all_scores = {name: scores(est, truth, g_total) for name, est in estimates.items()}
        in_box = (abs(xx) <= 1).all(axis=(1, 2))
        strata = dict(masks, in_box=in_box, weaker_in_box=in_box & masks["weaker"])
        assert all(set(task[mask]) == set(task) for mask in strata.values())
        report["results"][condition] = {
            "cases": len(task), "outside_unit_box": int(sum(~in_box)),
            "pairs_by_stratum": {name: int(mask.sum()) for name, mask in strata.items()},
            "strata": {name: summarize(all_scores, task_names, mask, boot) for name, mask in strata.items()},
        }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
