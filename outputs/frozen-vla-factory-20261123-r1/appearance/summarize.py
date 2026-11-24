"""Task-paired intervals. Each entire task contributes a single value per contrast."""

import csv
import json
import sys
from pathlib import Path

import numpy as np


def summarize(path):
    with (path / "scores.csv").open() as f:
        rows = list(csv.DictReader(f))
    tasks = sorted({r["task"] for r in rows})
    rng = np.random.default_rng(20261123)
    draws = rng.integers(len(tasks), size=(20000, len(tasks)))
    result = {}
    for metric in ("first_xyz_rms", "xyz_rms", "rotation_rms", "gripper_rms", "action_rms"):
        values = {}
        for contrast in sorted({r["contrast"] for r in rows}):
            groups = sorted([r for r in rows if r["contrast"] == contrast], key=lambda r: r["task"])
            assert [r["task"] for r in groups] == tasks
            values[contrast] = np.array([float(r[metric]) for r in groups])
        for a, b in [("masked_pair", "raw"), ("masked_pair", "sham_pair"),
                     ("target_mask_to_nominal", "raw"), ("target_mask_to_nominal", "target_sham_to_nominal"),
                     ("source_overlay_to_nominal", "raw"), ("source_overlay_to_nominal", "sham_overlay_to_nominal")]:
            values[f"{a}_minus_{b}"] = values[a] - values[b]
        table = {}
        for key, x in values.items():
            low, high = np.quantile(x[draws].mean(1), [0.025, 0.975])
            table[key] = dict(mean=x.mean(), lo=low, hi=high, positive=int((x > 0).sum()))
        result[metric] = table
    (path / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    summarize(Path(sys.argv[1]))
