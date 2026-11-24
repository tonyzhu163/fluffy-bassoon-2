"""Does lower support yaw error identify the better retry on a matched pair?"""

import csv
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parents[1] / "symmeta-matched-flow-20261123-r1/evidence"


def summarize(rows):
    tasks = sorted({(r["suite"], r["task_id"]) for r in rows})
    columns = ["learned", "linear", "lower_error", "either", "discordant", "correct_when_discordant"]
    counts = np.zeros((len(tasks), len(columns)), dtype=int)
    for row in rows:
        index = tasks.index((row["suite"], row["task_id"]))
        counts[index] += [row[name] for name in columns]
    task_sizes = Counter((r["suite"], r["task_id"]) for r in rows)
    if len(set(task_sizes.values())) != 1:
        raise ValueError("Expected balanced tasks")
    # Pair controls and resample all recorded cases of a task together.
    rng = np.random.default_rng(20261123)
    resampled = counts[rng.integers(len(tasks), size=(20000, len(tasks)))].sum(axis=1)
    diff = (resampled[:, 2] - resampled[:, 0]) / len(rows)
    return dict(n=len(rows), tasks=len(tasks), counts=dict(zip(columns, counts.sum(axis=0).tolist())),
                lower_error_minus_learned_pp=100 * (counts[:, 2] - counts[:, 0]).sum() / len(rows),
                paired_ci95_pp=np.percentile(100 * diff, [2.5, 97.5]).tolist())


def main():
    with gzip.open(ARCHIVE / "target-details.json.gz", "rt") as f:
        traces = {r["index"]: r for r in json.load(f)}
    rows = []
    with (ARCHIVE / "gpu01-cases.csv").open() as f:
        for pair in csv.DictReader(f):
            trace = traces.pop(int(pair["index"]))
            for key in ["suite", "task_id", "embodiment", "regime"]:
                if str(trace[key]) != pair[key]:
                    raise ValueError(f"Pair mismatch on {key}")
            learned, linear = int(pair["learned_success"]), int(pair["linear_success"])
            chosen = learned if trace["learned_abs_error_deg"] <= trace["linear_abs_error_deg"] else linear
            rows.append(dict(suite=pair["suite"], task_id=pair["task_id"], regime=pair["regime"],
                             learned=learned, linear=linear, lower_error=chosen, either=max(learned, linear),
                             discordant=int(learned != linear),
                             correct_when_discordant=int(learned != linear and chosen == 1)))
    if traces or len(rows) != 96:
        raise ValueError("Incomplete original experiment")
    groups = {"all": rows, "ood": [r for r in rows if r["regime"] != "familiar"]}
    summary = {label: summarize(group) for label, group in groups.items()}
    output = json.dumps(summary, indent=2) + "\n"
    (HERE / "result.json").write_text(output)
    print(output)


if __name__ == "__main__":
    main()
