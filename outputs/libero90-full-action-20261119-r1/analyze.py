"""Score prospective live retries with paired uncertainty across held-out task IDs."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t

HERE = Path(__file__).resolve().parent
MANIFEST = HERE.parent / "libero90-meta-retry-20261119-r1/cases.json"
ARMS = ("base", "matched", "meta_intact", "meta_cleared", "imitation_intact", "imitation_cleared")


def interval(task_means):
    values = np.array(task_means, dtype=float)
    mean = float(values.mean())
    radius = t.ppf(.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return dict(estimate=mean, lower=float(mean-radius), upper=float(mean+radius))


def analyze(root, split):
    assignments = [r for r in json.loads(MANIFEST.read_text()) if r["split"] == split]
    data = defaultdict(list)
    totals = defaultdict(lambda: defaultdict(int))
    for case in assignments:
        name = f"task{case['task_id']:02d}-r{case['repeat']:02d}"
        path = root / "live" / split / name
        if json.loads((path / "trial.json").read_text())["manifest_row"] != case:
            raise ValueError(f"Case mismatch for {name}")
        row = {}
        for arm in ("support", *ARMS):
            record = json.loads((path / arm / "result.json").read_text())
            if type(record["success"]) is not bool:
                raise ValueError(f"Invalid success in {name}/{arm}")
            row[arm] = int(record["success"])
            for field in ("controls", "policy_calls", "feature_calls"):
                totals[arm][field] += record.get(field, 0)
        data[case["task_id"]].append(row)
    per_task = {}
    for task, rows in data.items():
        per_task[task] = {arm:float(np.mean([r[arm] for r in rows])) for arm in ("support", *ARMS)}
    flat = [r for rows in data.values() for r in rows]
    pairs = {}
    for lhs, rhs in (("meta_intact","base"),("meta_intact","meta_cleared"),
                     ("meta_intact","imitation_intact"),("imitation_intact","imitation_cleared"),
                     ("matched","base")):
        delta = np.array([v[lhs]-v[rhs] for v in per_task.values()])
        sign = np.array([r[lhs]-r[rhs] for r in flat])
        pairs[f"{lhs}_minus_{rhs}"] = dict(**interval(delta), rescues=int(sum(sign>0)),
            regressions=int(sum(sign<0)), tasks_up=int(sum(delta>0)), tasks_down=int(sum(delta<0)),
            tasks_tied=int(sum(delta==0)))
    interaction = [v["meta_intact"]-v["meta_cleared"]-v["imitation_intact"]+v["imitation_cleared"]
                   for v in per_task.values()]
    return dict(split=split,tasks=len(data),cases=len(flat),per_task=per_task,paired=pairs,
                history_gain_difference=interval(interaction),
                rates={arm:interval([v[arm] for v in per_task.values()]) for arm in ARMS},
                at_least_one_success={arm:np.mean([r[arm] or r["support"] for r in flat]) for arm in ARMS},
                budget=totals)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("split", choices=("development","test"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = analyze(args.root, args.split)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
