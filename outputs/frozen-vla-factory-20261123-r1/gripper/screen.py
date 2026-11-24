"""Offline forward prediction of public jaw width. Run from any working directory."""
import gzip
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1]
SOURCE = OUT / "libero-xemb-friction-dev-20261119-r1/cpu01"
PIN = "757a44cd4374833cad790852df8b87ddf412e638"
MODELS = ("persist", "motion_only", "signed_gain", "sign", "analog", "dead_zone", "velocity", "memory")


@dataclass
class Episode:
    name: str
    task: str
    u: np.ndarray
    w: np.ndarray
    ordinary: bool

    def __post_init__(self):
        assert self.w.shape == (len(self.u) + 1,)
        assert np.isfinite(self.u).all() and np.isfinite(self.w).all()


def source():
    episodes = []
    for p in sorted(SOURCE.glob("*.npz")):
        suite, task, _, material = p.stem.split("-")
        with np.load(p) as d:
            q = d["robot0_gripper_qpos"].astype(float)
            u = d["issued"][:, 6].astype(float)
        episodes.append(Episode(p.stem, f"{suite}-{task}", u, 1000 * (q[:, 0] - q[:, 1]),
                                material == "ordinary"))
    assert len(episodes) == 32
    return episodes


def arm_probes():
    for arm in ("UR5e", "IIWA", "Kinova3"):
        p = OUT / f"xemb-stage2c-cpu-20261121-r1/evidence/{arm}-BASE/cell.json"
        d = json.loads(p.read_text())
        assert d["status"] == "passed" and d["support"]["model"]["gripper"] == "PandaGripper"
        u = np.array([r["issued"][6] for r in d["controls"]])
        w = [d["support"]["tool"]["jaw_opening"]] + [r["jaw_opening"] for r in d["controls"]]
        yield Episode(arm, "libero_goal-1", u, 1000 * np.array(w), True)


def command_memory(u, w0, limits, fraction):
    lo, hi = limits
    target = w0
    out = np.empty(len(u))
    for i, direction in enumerate(np.sign(u)):
        target = min(hi, max(lo, target - direction * fraction * (hi - lo)))
        out[i] = target
    return out


def features(model, u, w, v, target, thresholds):
    # Jaw widths and past velocities enter in mm. Positive u closes the source jaw.
    if model == "motion_only":
        return v[:, None]
    if model == "signed_gain":
        return u[:, None]
    if model in ("analog", "dead_zone"):
        a, b = thresholds
        close = np.maximum(0, u - a) / (1 - a)
        open_ = np.maximum(0, -u - b) / (1 - b)
    else:
        close, open_ = u > 0, u < 0
    cols = [close, open_, close * w, open_ * w]
    if model in ("velocity", "memory"):
        cols.append(v)
    if model == "memory":
        cols.append(target - w)
    return np.column_stack(cols)


@dataclass
class Fit:
    model: str
    beta: np.ndarray
    limits: tuple
    thresholds: tuple = (0., 0.)
    fraction: float = 0.

    def target(self, u, w0):
        if self.model == "memory":
            return command_memory(u, w0, self.limits, self.fraction)
        return np.zeros(len(u))

    def predict_change(self, u, w, v, target):
        if self.model == "persist":
            return np.zeros_like(w)
        return features(self.model, u, w, v, target, self.thresholds) @ self.beta


def fit(training, model):
    limits = (min(min(e.w) for e in training), max(max(e.w) for e in training))
    if model == "persist":
        return Fit(model, np.array([]), limits)
    # Fit equally weighted episodes on the ordinary material only. Every grid
    # decision, column and coefficient uses source episodes exclusively.
    thresholds = itertools.product((0., .25, .5, .75), repeat=2) if model == "dead_zone" else [(0., 0.)]
    fractions = (1/32, 1/16, 1/8, 1/4, 1/2, 1.) if model == "memory" else [0.]
    best_loss = np.inf
    for cut, fraction in itertools.product(thresholds, fractions):
        f = Fit(model, np.empty(0), limits, cut, fraction)
        rows, responses = [], []
        for e in training:
            u = np.clip(e.u, -1, 1)
            v = np.diff(e.w, prepend=e.w[0])[:-1]
            x = features(model, u, e.w[:-1], v, f.target(u, e.w[0]), cut)
            rows.append(x / np.sqrt(len(u)))
            responses.append(np.diff(e.w) / np.sqrt(len(u)))
        x, y = np.concatenate(rows), np.concatenate(responses)
        f.beta = np.linalg.lstsq(x, y, rcond=None)[0]
        loss = np.sum((x @ f.beta - y) ** 2)
        if loss < best_loss:
            best_loss, best = loss, f
    return best


def forecast(f, u, start_width, start_v, target, starts, horizon):
    """Advance independent blocks without any observed width inside a block."""
    w, v = start_width.copy(), start_v.copy()
    for k in range(horizon):
        ids = starts + k
        v = f.predict_change(u[ids], w, v, target[ids])
        w = w + v
    return w


def score(e, f):
    u = np.clip(e.u, -1, 1)
    target = f.target(u, e.w[0])
    v = np.diff(e.w, prepend=e.w[0])[:-1]
    out = {}
    for h in (1, 5):
        starts = np.arange(0, len(u) - h + 1, h)
        pred = forecast(f, u, e.w[starts], v[starts], target, starts, h)
        err = np.abs(pred - e.w[starts + h])
        out[f"h{h}"] = float(err.mean())
        if h == 1:
            switches = np.flatnonzero(np.diff(np.sign(u))) + 1
            if len(switches):
                out["switch"] = float(err[switches].mean())
    return out


def smoke(fits):
    # Only endpoint widths are committed. Ignore the first chunk in scoring
    # and use the causal mean velocity from the preceding 5-control segment.
    p = OUT / "xemb-pi0-source-smoke-20261122-r1/evidence/result.json"
    out = {}
    for key, row in json.loads(p.read_text())["outcomes"].items():
        c = row["first_chunks"]
        jaw = np.array(c["delta_jaw"]) * 1000
        u = np.clip(np.array([p["first_five_actions"] for p in row["proposals"]])[:, :, 6], -1, 1)
        np.testing.assert_allclose(u.mean(axis=1), np.array(c["issued_mean"])[:, 6], atol=1e-6)
        # All 25 commands open. Initialize command state at the first public
        # endpoint and use only the four subsequent blocks for prediction.
        starts = np.arange(0, 20, 5)
        u = u[1:].ravel()
        out[key] = {}
        for f in fits:
            target = f.target(u, jaw[0, 1])
            pred = forecast(f, u, jaw[:-1, 1], jaw[:-1, 0] / 5, target, starts, 5)
            out[key][f.model] = float(np.mean(np.abs(pred - jaw[1:, 1])))
    return out


def polarity():
    # Native replay of a synthetic wire inversion. Estimate solely on the
    # first two opposite pulses and score on the final nonoverlapping burst.
    src = OUT / "libero-xemb-joint-sysid-20261119-r1/cpu01"
    rows = []
    for idx in (28, 29, 30):
        with np.load(src / f"sensors-{idx:02}.npz") as d:
            u = d["issued_mean"][:, -1]
            w = (d["fingers"][:, 0] - d["fingers"][:, 1]) * 1000
        dw = np.diff(w)
        sign = int(-np.sign(u[:2] @ dw[:2]))
        rows.append(dict(index=idx, estimated_sign=sign, third_dw_mm=float(dw[2]),
                         movement_direction_correct=bool(-sign * np.sign(u[2]) == np.sign(dw[2]))))
    return rows


def historical_sign():
    # A scoring-only audit of archived sign predictions on the four 96-case
    # panels. IDs and injected signs never enter the response fits.
    proposals = OUT / "symmeta-seven-schedule-20261123-r1/evidence/gpu01/results/paired-cases-41.json.gz"
    with gzip.open(proposals, "rt") as fp:
        reference = json.load(fp)
    labels = {r["embodiment"]: r["gripper_sign"] for r in reference}
    reader = OUT / "symmeta-fk-ur5e-20261123-r1/evidence/reader.json"
    ur5e = json.loads(reader.read_text())["rows"]
    assert len(ur5e) == 96
    panels = {"UR5e/pi05": sum(r["grip_hat"] == labels[r["id"].split("/")[-1]] for r in ur5e)}
    for p in sorted((OUT / "symmeta-fk-crossarm-20261123-r1/evidence").glob("*-cases.json.gz")):
        with gzip.open(p, "rt") as fp:
            d = json.load(fp)
        rows = d["rows"]
        panels[f"{d['arm']}/{d['policy']}"] = sum(r["historical"]["grip_hat"] == labels[r["embodiment"]] for r in rows)
        assert len(rows) == 96
    return panels


def main():
    episodes = source()
    tasks = sorted({e.task for e in episodes})
    per_task, config, external = {}, {}, {}
    task_switch_counts = {}
    for task in tasks:
        train = [e for e in episodes if e.ordinary and e.task != task]
        test = [e for e in episodes if e.task == task]
        assert len(train) == 14 and len(test) == 4
        task_switch_counts[task] = int(sum(np.count_nonzero(np.diff(np.sign(e.u))) for e in test))
        fits = [fit(train, m) for m in MODELS]
        config[task] = {"memory_fraction": fits[-1].fraction,
                        "dead_zone_thresholds": next(f.thresholds for f in fits if f.model == "dead_zone"),
                        "fit_controls": sum(len(e.u) for e in train)}
        per_task[task] = {}
        for f in fits:
            scores = [score(e, f) for e in test]
            per_task[task][f.model] = {k: float(np.mean([s[k] for s in scores if k in s]))
                                       for k in sorted(set.union(*(set(s) for s in scores)))}
        if task == "libero_goal-1":
            external["active_cross_arm_mm"] = {e.name: {f.model: score(e, f) for f in fits}
                                               for e in arm_probes()}
            external["frozen_opening_h5_mm"] = smoke(fits)
    rng = np.random.default_rng(20261123)
    summary, differences = {}, {}
    for metric in ("h1", "h5", "switch"):
        selected = [r for r in per_task.values() if metric in r["persist"]]
        x = np.array([[r[m][metric] for m in MODELS] for r in selected])
        indices = rng.integers(len(x), size=(20000, len(x)))
        summary[metric] = dict(zip(MODELS, x.mean(axis=0)))
        differences[metric] = {}
        for a, b in (("persist", "memory"), ("motion_only", "memory"), ("sign", "memory"), ("velocity", "memory"),
                     ("sign", "dead_zone"), ("sign", "analog")):
            delta = x[:, MODELS.index(a)] - x[:, MODELS.index(b)]
            ci = np.quantile(delta[indices].mean(axis=1), [.025, .975])
            differences[metric][f"{a}_minus_{b}"] = [delta.mean(), *ci]
    u = np.concatenate([e.u for e in episodes])
    report = dict(revision=PIN, episodes=len(episodes), controls=len(u), tasks=tasks,
                  raw_abs_u_range=[abs(u).min(), abs(u).max()],
                  controls_with_abs_u_below_0_98=int(np.sum(abs(u) < .98)),
                  switch_counts=task_switch_counts, support=config,
                  mean_error_mm=summary, paired_task_mean_and_percentile95_mm=differences,
                  task_means_mm=per_task, **external, inverted_probe=polarity(),
                  historical_sign_correct_of_96=historical_sign())
    (HERE / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
