"""Summarize the three methods after exact assignment and opening-trajectory checks."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from reground.eval.stats import summarise
from reground.utils import noise_seed

HERE = Path(__file__).resolve().parent
METHODS = ("base", "reground", "symmeta")
KEY = ["task", "init_id", "rep", "adapt_seed", "noise_seed"]


def compare(path):
    cases = pd.DataFrame(json.loads(path.read_text()))
    with gzip.open(HERE / "evidence/rotation-jobs.json.gz", "rt") as f:
        jobs = json.load(f)
    planned = {(task, start, rep, j["seed"], noise_seed(start, rep))
               for j in jobs for task, start, rep in j["episodes"]}
    for col in KEY[1:]:
        cases[col] = pd.to_numeric(cases[col], errors="raise").astype(int)
    if len(cases) != len(planned) or set(cases[KEY].itertuples(index=False, name=None)) != planned:
        raise ValueError("The three-method comparison requires the exact five-seed released test grid")
    if not cases["matched_prefix"].between(1, 10).all():
        raise ValueError("Every matched episode needs an audited common opening")
    methods, contrasts = {}, {}
    for method in METHODS:
        rows = cases.rename(columns={method + "_" + metric: metric for metric in ("success", "unsafe")})
        if not rows[["success", "unsafe"]].isin([0, 1]).all().all():
            raise ValueError(f"Invalid outcome in {method}")
        result = summarise(rows)
        result["per_task"] = rows.groupby("task")[["success", "unsafe"]].mean().to_dict("index")
        disagreement = rows.groupby(["task", "init_id"])[["success", "unsafe"]].nunique().gt(1)
        result["starts_with_variation_across_seeds"] = disagreement.sum().to_dict()
        methods[method] = result

    # Each start and its policy noise is shared by five fits. Keep the entire vector
    # of five paired outcomes together when resampling starts within a fixed task.
    for a, b in (("symmeta", "base"), ("reground", "base"), ("symmeta", "reground")):
        pair = {}
        for metric in ("success", "unsafe"):
            delta = cases[a + "_" + metric] - cases[b + "_" + metric]
            blocks = cases.assign(delta=delta).groupby(["task", "init_id"]).delta.mean()
            rng = np.random.default_rng(20261123)
            per_task = []
            for _, group in blocks.groupby(level="task", sort=True):
                d = group.to_numpy()
                per_task.append(rng.choice(d, (10000, len(d)), replace=True).mean(axis=1))
            boot = np.mean(per_task, axis=0)
            pair[metric] = dict(mean_difference=float(delta.mean()),
                                start_cluster_ci95=np.percentile(boot, [2.5, 97.5]).tolist(),
                                a_only=int((delta == 1).sum()), b_only=int((delta == -1).sum()),
                                ties=int((delta == 0).sum()))
        contrasts[a + "_minus_" + b] = pair
    return dict(assignments=len(cases), unique_task_starts=len(cases[["task", "init_id"]].drop_duplicates()),
                official_iqm=methods, paired_assignment_differences=contrasts,
                sensitivity="10,000 bootstrap draws of starts within each fixed task, retaining all five "
                            "adaptation seeds and both methods within start; conditional on the five seeds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paired_cases", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = compare(args.paired_cases)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
