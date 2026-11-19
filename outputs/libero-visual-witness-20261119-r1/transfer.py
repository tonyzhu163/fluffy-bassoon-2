"""Apply the previous interruption study's trained motion and jaw readers to the matched camera panel."""
import argparse
from importlib.util import module_from_spec, spec_from_file_location
from itertools import permutations, product
import json
from pathlib import Path

import numpy as np


PRIOR = "libero-interrupted-practice-20261119-r1"


def prior_reader():
    path = Path(__file__).resolve().parents[1] / PRIOR / "score.py"
    spec = spec_from_file_location("interrupted_score", path)
    previous = module_from_spec(spec)
    spec.loader.exec_module(previous)
    archive = previous.ROOT / PRIOR / "pair01/branches"
    source = previous.load(archive)
    source = [row for (family, pose, *_), row in source.items() if family == "bowl" and 32 <= pose <= 35]
    assert len(source) == 64
    return previous, previous.fit(source)


def evaluate(path):
    previous, model = prior_reader()
    preds, outcomes = {}, {}
    for p, g, v in product(range(12, 20), range(2), ("BASE", "D18")):
        with np.load(path / f"p{p}-g{g}-{v}.npz") as z:
            features = previous.features(z["issued"], z["states"])
            preds[p, g, v] = previous.predict(model, dict(goal=g, **features))
            outcomes[p, g, v] = tuple(z["hits"].any(axis=0))
    rows = []
    for bank in (range(12, 16), range(16, 20)):
        for a, b, q in permutations(bank, 3):
            if not all(outcomes[p, g, "BASE"] == (g == 0, g == 1) for p, g in ((a, 0), (b, 1), (q, 0), (q, 1))):
                continue
            if any(any(outcomes[p,g,"D18"]) for p,g in ((a,0),(b,1))):
                continue
            cyclic = bank[(bank.index(a)+1)%4] == b and bank[(bank.index(a)+2)%4] == q
            record = dict(a=a, b=b, query=q, cyclic=cyclic, source=a<16)
            for mode in ("history", "jaw", "commands", "prior"):
                successes = []
                for assigned in range(2):
                    va, vb = ("D18", "BASE") if assigned == 0 else ("BASE", "D18")
                    pa, pb = preds[a,0,va][mode], preds[b,1,vb][mode]
                    # Two observed zero rewards. Follow the earlier model, including its off-target prior.
                    choose_a = previous.decide([0,0], np.array([[pa, model["off"]], [model["off"], pb]]))
                    successes.append(choose_a if assigned == 0 else 1-choose_a)
                record[mode]=successes
            rows.append(record)
    summary = []
    for split, only_cyclic in product(("fit","development"), (True, False)):
        subset = [r for r in rows if r["source"] == (split == "fit") and (r["cyclic"] or not only_cyclic)]
        count = len(subset)
        summary.append(dict(split=split, cyclic=only_cyclic, strict=count, assignments=2*count,
                            successes={m:sum(sum(r[m]) for r in subset) for m in ("history","jaw","commands","prior")}))
    return dict(source="Goal 8 S02 forced openings; fit on 64 physical branches from poses 32–35",
                jaw_source_rule=[float(x) for x in model["jaw"]], summary=summary, groups=rows)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    args.output.write_text(json.dumps(evaluate(args.input),indent=2)+"\n")
