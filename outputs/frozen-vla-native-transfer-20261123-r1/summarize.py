"""Review pilot uncertainty at the registered six matched starts, including failures."""
import argparse
import json
from collections import Counter
from pathlib import Path
from scipy.stats import binomtest
from panel import cases, key, save


def read(path):
    return json.loads(path.read_text())


def main(out):
    rows = []
    for case in cases("Kinova3"):
        stem = key(case)
        recs = {name: read(out / f"{stem}-{name}.json")["record"]
                for name in ("Panda", "Kinova3", "matrix", "planar_matrix")}
        tracker = read(out / f"{stem}-tracking.json")
        if not tracker.get("skipped"):
            recs["tracking"] = tracker
        rows.append(dict(case=case, success={k: r["success"] for k, r in recs.items()},
                         controls={k: r["controls"] for k, r in recs.items()}))
    pairs = {}
    for first, second in (("matrix", "Kinova3"), ("planar_matrix", "Kinova3"), ("matrix", "planar_matrix"),
                          ("Panda", "Kinova3"), ("tracking", "Kinova3")):
        counts = Counter((r["success"][first], r["success"][second]) for r in rows if first in r["success"])
        gains, losses = counts[True, False], counts[False, True]
        n = sum(counts.values())
        discordant = gains + losses
        pairs[f"{first}_minus_{second}"] = dict(n=n, gains=gains, losses=losses,
            difference=(gains-losses)/n if n else None,
            exact_discordant_sign_p=binomtest(gains, discordant).pvalue if discordant else None)
    save(out / "summary.json", dict(rows=rows, contrasts=pairs,
         inference="Six selected within-task starts; exact sign probabilities assume independent symmetric discordance. "
                   "Task/start effects and individual pairs matter more than pooled significance in this pilot."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    main(parser.parse_args().out)
