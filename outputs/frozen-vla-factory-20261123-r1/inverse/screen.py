"""Score a noise-damped inverse using the excitation experiment's support receipts."""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "excitation"))
sys.path.insert(0, str(HERE.parent / "native"))
from inputs import SEED, load, rot, supports
from score import calibrate, estimates, features
from orbit import command_tapes
from fluffy_bassoon.xemb.embodiment import get_embodiment_spec


def corrections(x, y, coef, sigma):
    maps = estimates(x, y, coef, sigma)
    model = maps["full"]
    exposure = features(x) @ coef
    cov = sigma**2 * np.linalg.inv(exposure.T @ exposure + sigma**2 * np.eye(2))
    gram = model.T @ model
    return {"pinv": np.linalg.pinv(model),
            "anisotropic": np.linalg.solve(gram + 2 * cov, model.T),
            "isotropic": np.linalg.solve(gram + np.trace(cov) * np.eye(2), model.T),
            "similarity": np.linalg.pinv(maps["similarity"])}


def score_case(x, y, truth, coef, sigma, tapes):
    values = {}
    for method, correction in corrections(x, y, coef, sigma).items():
        residual = np.linalg.norm(truth @ correction - np.eye(2)) / np.sqrt(2)
        distances = []
        for cmd in tapes:
            issued = np.clip(cmd @ correction.T, -1, 1)
            reached = np.clip(issued @ truth.T, -1, 1)
            distances.append(np.linalg.norm(reached - cmd, axis=1).mean())
        values[method] = [float(residual), float(np.mean(distances))]
    return values


def summarize(rows):
    tasks = sorted({task for task, _ in rows})
    rng = np.random.default_rng(SEED)
    draws = rng.integers(len(tasks), size=(20000, len(tasks)))
    methods = {}
    for method in rows[0][1]:
        delta = np.array([np.mean([np.subtract(v[method], v["pinv"]) for t, v in rows if t == task],
                                  axis=0) for task in tasks])
        values = np.array([v[method] for _, v in rows])
        methods[method] = dict(mean=values.mean(axis=0).tolist(), p95=np.quantile(values,.95,axis=0).tolist(),
                               delta_mean=delta.mean(axis=0).tolist(),
                               delta_ci95=np.quantile(delta[draws].mean(axis=1), [.025,.975],axis=0).tolist())
    return dict(n=len(rows), task_clusters=len(tasks), methods=methods)


def main():
    coef, sigma, _ = calibrate()
    tapes = [v[:, :2] for v in command_tapes().values()]
    dataset = load(HERE.parent / "excitation/supports.json.gz")
    triplets = defaultdict(dict)
    for row in dataset["rows"]:
        key = tuple(row[k] for k in ("task", "motor", "template", "sign", "amplitude"))
        assert row["design"] not in triplets[key]
        triplets[key][row["design"]] = row
    assert len(triplets) == 128
    rng = np.random.default_rng(SEED + 2)  # same draws in the same group order as the excitation screen
    groups = defaultdict(list)
    for key, triplet in sorted(triplets.items()):
        assert set(triplet) == {"one","two","passive"}
        jitter = np.diff(rng.normal(0, sigma / np.sqrt(2), size=(4, 2)), axis=0)
        for design, row in triplet.items():
            values = score_case(np.array(row["x"]), np.array(row["y"]) + jitter, np.array(row["truth"]),
                                coef, sigma, tapes)
            groups[f"{design}/{row['amplitude']}"].append((row["task"], values))
    for row in supports()[0]:
        spec = get_embodiment_spec(row["label"])
        truth = spec.xy_scale * rot(np.deg2rad(spec.yaw_deg))
        values = score_case(row["x"], row["y"], truth, coef, sigma, tapes)
        groups["recorded"].append((row["task"], values))
    result = {name: summarize(group) for name, group in groups.items()}
    report = {"endpoints": ["inverse operator residual", "clipped command XY error"], "groups": result}
    (HERE / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for group, values in result.items():
        print(group, {method: scores["mean"] for method, scores in values["methods"].items()})


if __name__ == "__main__":
    main()
