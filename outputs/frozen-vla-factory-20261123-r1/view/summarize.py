"""Recompute scores from the actual seven-channel chunks and bootstrap whole tasks."""
import argparse
import json
from pathlib import Path

import numpy as np


def rms(delta):
    return float(np.sqrt(np.mean(np.square(delta, dtype=np.float64))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    out = p.parse_args().out
    result = json.loads((out / "scores.json").read_text())
    tasks = result["tasks"]
    n = len(tasks)
    # Each bootstrap draw uses the same task indices for every paired contrast.
    draws = np.random.default_rng(231123).integers(0, n, (20000, n))
    summary = {}
    per_task = {}
    slices = dict(full=np.s_[:, :], first_action=np.s_[:1, :], first_five=np.s_[:5, :],
                  translation=np.s_[:, :3], rotation=np.s_[:, 3:6], gripper=np.s_[:, 6:])
    with np.load(out / "proposals.npz", allow_pickle=False) as chunks:
        for task in tasks:
            key = task["key"]
            nominal = chunks[f'{key}_{task["baseline"]}']
            assert nominal.shape == (50, 7) and np.isfinite(nominal).all()
            per_task[key] = {}
            for camera in ("agent", "wrist", "both_pooled", "upright"):
                rows = [r for r in task["rows"] if r["camera"] == camera
                        or (camera == "both_pooled" and r["camera"] is not None)
                        or (camera == "upright" and r["camera"] is None)]
                entry = {}
                for measure, select in slices.items():
                    metrics = {}
                    for label in ("raw", "estimated", "oracle"):
                        errors = []
                        for row in rows:
                            pred = chunks[f'{key}_{row["chunks"][label]}']
                            error = rms((pred - nominal)[select])
                            if measure == "full":
                                np.testing.assert_allclose(error, row["rmse"][label], atol=1e-15, rtol=0)
                            errors.append(error)
                        metrics[label] = float(np.mean(errors))
                    metrics["raw_minus_estimated"] = metrics["raw"] - metrics["estimated"]
                    entry[measure] = metrics
                per_task[key][camera] = entry
    for camera in ("agent", "wrist", "both_pooled", "upright"):
        summary[camera] = {}
        for measure in slices:
            summary[camera][measure] = {}
            for label in ("raw", "estimated", "oracle", "raw_minus_estimated"):
                values = np.array([v[camera][measure][label] for v in per_task.values()])
                ci = np.percentile(values[draws].mean(axis=1), [2.5, 97.5])
                summary[camera][measure][label] = dict(mean=float(values.mean()), ci95_tasks=ci.tolist())
    data = dict(n_tasks=n, bootstrap_seed=231123, bootstrap_replicates=len(draws),
                summary=summary, per_task_full={k:{c:m["full"] for c,m in v.items()} for k,v in per_task.items()})
    (out / "summary.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k:v["full"] for k,v in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
