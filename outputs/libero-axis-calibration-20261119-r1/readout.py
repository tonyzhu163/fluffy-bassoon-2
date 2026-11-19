"""Recount saved predictions for each physical source trial. Pass the Baige attempt directory."""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def recount(folder, trials):
    records = [json.loads(line) for line in (folder / "predictions.jsonl").read_text().splitlines()]
    metrics = json.loads((folder / "metrics.json").read_text())
    counts = defaultdict(Counter)
    clears, targets = {}, {}
    identities = set()
    by_source = defaultdict(Counter)
    for row in records:
        key, method, convention, support, direction = (
            row[k] for k in ("key", "method", "convention", "support", "direction"))
        identity = (key, method, convention, support, direction)
        assert identity not in identities
        identities.add(identity)
        target, pred = row["target"], row["prediction"]
        assert all(x in (-1, 1) for x in target + pred)
        assert target == targets.setdefault((key, convention), target)
        if support == "cleared" and method.startswith("seed"):
            assert pred == clears.setdefault((key, method), pred)
        scores = dict(n=1, joint=int(target == pred), x=int(target[0] == pred[0]), y=int(target[1] == pred[1]))
        counts[direction, method, support, convention].update(scores)
        if support == "matched" and convention == "xy":
            by_source[method, key].update(scores)
    assert len(identities) == trials * 4 * 3 * 4 * 6
    summary = {}
    for (direction, method, support, convention), count in counts.items():
        assert count["n"] == trials
        for axis in ("joint", "x", "y"):
            reported = metrics[direction][method][support]["by_convention"][convention][axis]
            assert abs(reported - count[axis] / trials) < 1e-6
    for method in metrics["nom"]:
        per_condition = {}
        for support in ("matched", "cleared", "opposite"):
            panels = {}
            for directions, name in ((("nom",), "archived_direction"), (tuple(metrics), "four_directions")):
                per_convention = {}
                for conv in ("nom", "x", "y", "xy"):
                    total = Counter()
                    for direction in directions:
                        total.update(counts[direction, method, support, conv])
                    per_convention[conv] = dict(total)
                panels[name] = per_convention
            per_condition[support] = panels
        summary[method] = per_condition
    return dict(counts=summary, xy_by_source={m: {k: dict(by_source[m, k]) for k, _ in targets}
                                              for m in metrics["nom"]}, prediction_rows=len(records),
                prediction_sha256=hashlib.sha256((folder / "predictions.jsonl").read_bytes()).hexdigest(),
                checkpoint_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.glob("*.pt"))})


if __name__ == "__main__":
    root = Path(sys.argv[1])
    code, end, start = map(int, (root / "exit.txt").read_text().split())
    assert code == 0
    fit = root / "fit"
    summary = json.loads((fit / "summary.json").read_text())
    arms = {arm: recount(fit / arm, summary["development"]["retained_trials"]) for arm in ("archived", "balanced")}
    assert all(len(arm["checkpoint_sha256"]) == len(summary["seeds"]) for arm in arms.values())
    print(json.dumps(dict(run=root.name, source=(root / "launch.txt").read_text().strip(),
                          runtime=(root / "runtime.txt").read_text().strip(), exit=code,
                          start_unix=start, end_unix=end, progress=(root / "run.log").read_text().splitlines(),
                          data=summary, results=arms), indent=1))
