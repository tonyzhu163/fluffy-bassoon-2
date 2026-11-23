"""Score complete frozen-policy branch tables. Run with Python and NumPy only."""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parents[1]
MENU = OUTPUTS / "libero-decision-pretraining-20261119-r1/reader01-summary.json"
PREFIX = OUTPUTS / "libero-first-controls-20261119-r1/prefix-readout.json"
SPLIT = OUTPUTS / "libero-meta-learning-20261117-r1/SPLITS.json"
FRICTION = OUTPUTS / "libero-xemb-friction-dev-20261119-r1/cpu01"
MODES = ("default", "paced", "short_wait", "gripper_wait")
PIN = "757a44cd4374833cad790852df8b87ddf412e638"


def read(path):
    return json.loads(path.read_text())


@dataclass
class Panel:
    tasks: np.ndarray
    regimes: np.ndarray
    motion: np.ndarray
    lag: np.ndarray
    hits: np.ndarray


def panel(menu, prefix, task_split):
    # Support IDs 0,3 correspond to query poses 1,4 on a fresh reset.
    indexed = {}
    for row in menu["detail"]:
        suite, task, start, regime = row["case"]
        key = (f"{suite}/{task}", start + 1, regime)
        assert key not in indexed
        indexed[key] = row
    expected = {(f"{suite}/{task}", pose, regime)
                for suite, tasks in task_split.items() for task in tasks
                for pose in (1, 4) for regime in ("BASE", "S20")}
    keys = [(r["task"], r["pose"], r["timing"]) for r in prefix]
    assert set(indexed) == set(keys) == expected and len(keys) == len(expected)
    motion, lag, hits = [], [], []
    for key, row in zip(keys, prefix, strict=True):
        record = indexed[key]
        outcomes = [record["success"][mode] for mode in MODES]
        assert all(type(x) is bool for x in outcomes) and row["hits"] == outcomes
        assert len(row["current"]) == 5 and np.isfinite(row["current"]).all()
        motion.append(row["current"][-1])
        # reader.py writes legacy_lag directly from the public-motion estimator.
        decision = record["choices"]["legacy_lag"]
        assert decision in ("default", "paced")
        lag.append(int(decision == "paced"))
        hits.append(outcomes)
    return Panel(np.array([k[0] for k in keys]), np.array([k[2] for k in keys]),
                 np.array(motion), np.array(lag), np.array(hits, dtype=int))


@dataclass
class Readout:
    mean: float
    scale: float
    center: np.ndarray
    bias: np.ndarray
    weights: np.ndarray

    @staticmethod
    def design(z, lag):
        return z[:, None] if lag is None else np.column_stack((z, lag, z * lag))

    @classmethod
    def fit(cls, motion, lag, outcomes):
        mean, scale = motion.mean(), motion.std()
        assert scale > 0
        features = cls.design((motion - mean) / scale, lag)
        target = outcomes - outcomes[:, :1]
        center, bias = features.mean(axis=0), target.mean(axis=0)
        x = features - center
        weights = np.linalg.solve(x.T @ x + np.eye(x.shape[1]), x.T @ (target - bias))
        return cls(mean, scale, center, bias, weights)

    def scores(self, motion, lag=None):
        features = self.design((motion - self.mean) / self.scale, lag)
        return (features - self.center) @ self.weights + self.bias


def contrast(candidate, control, tasks):
    delta = candidate.astype(int) - control.astype(int)
    task_totals = np.array([delta[tasks == t].sum() for t in sorted(set(tasks))])
    # Exact percentile bootstrap distribution of the sum from n whole-task draws.
    # All tasks have equal size. Each draw carries both regimes and all its resets.
    assert len(set(Counter(tasks).values())) == 1
    offset = task_totals.min()
    single = np.bincount(task_totals - offset) / len(task_totals)
    pmf = np.array([1.0])
    for _ in task_totals:
        pmf = np.convolve(pmf, single)
    ends = np.searchsorted(pmf.cumsum(), [0.025, 0.975]) + offset * len(task_totals)
    return dict(wins=int((delta > 0).sum()), losses=int((delta < 0).sum()),
                delta_pp=float(100 * delta.mean()),
                task_bootstrap_95_pp=(100 * ends / len(tasks)).tolist(),
                task_net=task_totals.tolist())


def counts(scores, mask):
    return {k: int(v[mask].sum()) for k, v in scores.items()}


def report(chosen, hits, modes, tasks, regimes, pairs):
    indices = np.arange(len(hits))
    scores = {mode: hits[:, i] for i, mode in enumerate(modes)}
    scores.update({name: hits[indices, choices] for name, choices in chosen.items()})
    scores["oracle"] = hits.max(axis=1)
    if "gripper_wait" in modes:
        scores["oracle_no_phase"] = hits[:, :modes.index("gripper_wait")].max(axis=1)
    return dict(
        n=len(hits), tasks=sorted(set(tasks)), total=counts(scores, indices),
        by_regime={k: counts(scores, regimes == k) for k in sorted(set(regimes))},
        selections={k: {m: int((v == i).sum()) for i, m in enumerate(modes)}
                    for k, v in chosen.items()},
        contrasts={f"{a} minus {b}": contrast(scores[a], scores[b], tasks) for a, b in pairs},
    )


def timing_screen(fit, dev, current, history):
    # Only public features reach the prediction step.
    choices = {}
    for name, scores in [("current", current.scores(dev.motion)),
                         ("history", history.scores(dev.motion, dev.lag))]:
        choices[name] = scores.argmax(axis=1)
        choices[name + "_no_phase"] = scores[:, :3].argmax(axis=1)
    fixed = fit.hits.sum(axis=0).argmax()
    by_lag = np.array([fit.hits[fit.lag == value].sum(axis=0).argmax() for value in (0, 1)])
    assert set(fit.lag) == {0, 1}
    choices["fit_fixed"] = np.full(len(dev.lag), fixed)
    choices["lag_only"] = by_lag[dev.lag]
    pairs = [("gripper_wait", k) for k in ("short_wait", "paced", "default", "lag_only")]
    pairs += [("current", "current_no_phase"), ("history", "history_no_phase"),
              ("history", "current"), ("history", "fit_fixed"), ("history", "lag_only"),
              ("oracle", "oracle_no_phase"), ("oracle", "lag_only")]
    result = report(choices, dev.hits, MODES, dev.tasks, dev.regimes, pairs)
    result["fit_fixed"] = MODES[fixed]
    result["fit_lag_table"] = [MODES[i] for i in by_lag]
    return result


def material_screen(current, expected_tasks):
    rows = [json.loads(line) for line in (FRICTION / "branches.jsonl").read_text().splitlines()]
    lookup = {(r["task"], r["material"], r["mode"]): r for r in rows}
    tasks = sorted(t.replace("/", "-") for t in expected_tasks)
    modes = MODES[:2]
    expected = {(t, r, m) for t in tasks for r in ("ordinary", "reduced") for m in modes}
    assert set(lookup) == expected and len(rows) == len(expected)
    motion, hits, names, regimes, transitions = [], [], [], [], []
    for task, material in sorted({key[:2] for key in expected}):
        outcomes, prefixes = [], []
        for mode in modes:
            record = lookup[task, material, mode]
            with np.load(FRICTION / f"{task}-{mode}-{material}.npz", allow_pickle=False) as tape:
                outcomes.append(int(np.any(tape["rewards"] > 0)))
                assert record["success"] == outcomes[-1]
                prefixes.append((tape["issued"][:6], tape["robot0_eef_pos"][:6]))
            for key in ("physics", "frame_state", "public", "reset", "seed", "cameras"):
                assert record[key] == lookup[task, material, "default"][key]
        (cmd, wrist), (other_cmd, other_wrist) = prefixes
        assert np.array_equal(cmd[:5], other_cmd[:5]) and np.array_equal(wrist, other_wrist)
        assert not np.array_equal(cmd[5], other_cmd[5])
        # Match pilot.py: cumulative Euclidean wrist travel / commanded xyz norm.
        denominator = np.linalg.norm(cmd[:5, :3].astype(float), axis=1).sum()
        ratio = np.linalg.norm(np.diff(wrist.astype(float), axis=0), axis=1).sum() / denominator
        assert np.isfinite(ratio)
        signs = np.sign(cmd[:5, -1])
        transitions.append(bool(np.any(np.diff(signs[signs != 0]))))
        motion.append(ratio)
        hits.append(outcomes)
        names.append(task)
        regimes.append(material)
    choices = {"current_two_tapes": current.scores(np.array(motion))[:, :2].argmax(axis=1)}
    result = report(choices, np.array(hits), modes, np.array(names), np.array(regimes),
                    [("current_two_tapes", "default"), ("oracle", "default")])
    result["prefix_with_gripper_sign_change"] = sum(transitions)
    return result


def main():
    menu, prefix, splits = map(read, (MENU, PREFIX, SPLIT))
    assert tuple(menu["modes"]) == MODES
    fit = panel(menu["fitting"], prefix["fit"], splits["development_scaling"]["task_subsets"]["8"])
    dev = panel(menu["dev_without_model"], prefix["development"], splits["development"])
    assert set(fit.tasks).isdisjoint(dev.tasks)
    current = Readout.fit(fit.motion, None, fit.hits)
    history = Readout.fit(fit.motion, fit.lag, fit.hits)
    result = dict(base_revision=PIN, collector="menu02 at b90ee5a0c049d301529d0bf661d0a682b858ed78",
                  endpoint="offline lookup of recorded terminal branch success",
                  fit_cases=len(fit.hits), fit_tasks=sorted(set(fit.tasks)), gpu_hours=0,
                  timing=timing_screen(fit, dev, current, history),
                  friction=material_screen(current, set(dev.tasks)))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
