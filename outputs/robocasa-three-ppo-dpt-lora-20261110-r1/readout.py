"""Read final teacher-action errors, grouping the 12 queries from each reset."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def summarize(run):
    data = json.loads((run / "data-01/dataset.json").read_text())
    fit = run / "fit-01"
    scores = json.loads((fit / "held-out.json").read_text())
    assert "Completed both fits and held-out comparison" in (fit / "fit.log").read_text()
    metrics = ["frozen", "memoryless", "own", "same_task_donor"]
    grouped, means = {}, {}
    used = set()
    for row in scores:
        case = row["task"], row["rank"], row["lane"]
        key = (*case, row["chunk"])
        assert 1 <= key[-1] <= 12 and key not in used
        used.add(key)
        grouped.setdefault(case, []).append([row[k] for k in metrics])
    expected = {(b["task"], b["rank"], i) for b in data["banks"]
                for i, split in enumerate(b["split"]) if split == "validation"}
    assert set(grouped) == expected and all(len(rows) == 12 for rows in grouped.values())
    for task in data["tasks"]:
        means[task] = np.stack([np.mean(grouped[k], axis=0) for k in sorted(expected) if k[0] == task])
        assert means[task].shape == (16, 4) and np.isfinite(means[task]).all()
    base = {"OpenDrawer": 11010000, "CloseToasterOvenDoor": 11020000, "TurnOnSinkFaucet": 73280000}
    for b in data["banks"]:
        assert set(b["seeds"]).isdisjoint(range(base[b["task"]], base[b["task"]]+512))
    arr = np.stack(list(means.values()))
    rng = np.random.default_rng(10)
    boot = np.stack([rows[rng.integers(0, 16, size=(10000, 16))].mean(1) for rows in arr]).mean(0)
    gains = {}
    for name, a, b in (("frozen_minus_memoryless", 0, 1), ("memoryless_minus_own", 1, 2),
                       ("donor_minus_own", 3, 2)):
        gains[name] = {"difference":float((arr[:,:,a]-arr[:,:,b]).mean()),
                       "scene_bootstrap_95":np.quantile(boot[:,a]-boot[:,b], [.025,.975]).tolist()}
    return {"per_task":{k:dict(zip(metrics, x.mean(0))) for k,x in means.items()},
            "pooled":dict(zip(metrics, arr.mean((0,1)))), "paired_gains":gains,
            "cases":len(expected), "queries":len(scores), "historical_512_overlap":0,
            "checkpoint_sha256":hashlib.sha256((fit / "checkpoint.pt").read_bytes()).hexdigest(),
            "raw_scores_sha256":hashlib.sha256((fit / "held-out.json").read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    print(json.dumps(summarize(parser.parse_args().run), indent=2))
