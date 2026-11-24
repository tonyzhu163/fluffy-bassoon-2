"""Paired task bootstrap of degree-valued estimands. Weights are equal within each level."""
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np


def wrap(a):
    return (a + 180) % 360 - 180


def main(folder):
    rows = json.loads((folder / "predictions.json").read_bytes())
    groups = defaultdict(dict)
    for r in rows:
        key = (r["index"], r["seed"])
        condition = groups[r["condition"]]
        if key in condition:
            raise ValueError(key)
        condition[key] = r
    keys = {(i,s) for i in range(96) for s in (41,42,43)}
    if len(groups) != 13 or any(set(group) != keys for group in groups.values()):
        raise ValueError("Incomplete paired matrix")
    draws = np.random.default_rng(20261124).integers(8, size=(20000, 8))
    metrics = ["mae", "paired_mae_change", "signed_shift", "signed_shift_near_expected", "absolute_shift",
               "tracking_residual", "tracking_minus_zero_residual", "fraction_closer_to_expected_than_zero"]
    summary = dict(task_bootstrap_seed=20261124, draws=len(draws), interval="paired task percentile 95%",
                   populations={})
    for label, filter_ood in [("all", False), ("ood", True)]:
        indices = [i for i in range(96) if not filter_ood or groups["original"][i,41]["regime"]
                   in ("continuous_angle_ood", "compositional_ood")]
        trace_tasks = [f"{groups['original'][i,41]['suite']}/{groups['original'][i,41]['task_id']}" for i in indices]
        tasks = sorted(set(trace_tasks))
        if len(tasks) != 8 or len(indices) != (64 if filter_ood else 96):
            raise ValueError("Unexpected panel")
        pop = dict(n_traces=len(indices), n_tasks=len(tasks), seeds=[41,42,43], conditions={})
        for name, group in groups.items():
            values = []
            for i in indices:
                triple = []
                for seed in (41,42,43):
                    row = group[i,seed]
                    a = row["shift_deg"]
                    expected = row["expected_shift_deg"]
                    residual = row["tracking_residual_deg"]
                    triple.append([row["error_deg"], row["error_change_deg"], a, expected+wrap(a-expected),
                                   abs(a), residual, residual-abs(a), float(residual < abs(a))])
                values.append(triple)
            arr = np.asarray(values)
            traces = arr.mean(axis=1)
            by_task = np.stack([traces[np.asarray(trace_tasks)==t].mean(axis=0) for t in tasks])
            stats = np.stack([by_task.mean(axis=0), *np.quantile(by_task[draws].mean(axis=1), [.025,.975], axis=0)])
            pop["conditions"][name] = dict(
                metrics={k:dict(zip(("mean","lower","upper"), v)) for k,v in zip(metrics,stats.T,strict=True)},
                task_means={t:dict(zip(metrics,v)) for t,v in zip(tasks,by_task,strict=True)},
                seed_means={s:dict(zip(metrics,v)) for s,v in zip((41,42,43),arr.mean(axis=0),strict=True)})
        summary["populations"][label] = pop
    with (folder/"summary.json").open("x") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
        f.write("\n")
    print("Scored 13 complete paired conditions across eight tasks.")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
