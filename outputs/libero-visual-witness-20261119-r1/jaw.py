"""Exploratory test for loss of the bowl during a sustained close command."""
import json
from itertools import product
from pathlib import Path
import sys

import numpy as np


def narrowing(z):
    width = np.abs(z["states"][:, -1] - z["states"][:, -2])
    closed = np.lib.stride_tricks.sliding_window_view(z["issued"][:, -1] > 0, 11).all(axis=1)
    return (width[:-10] - width[10:])[closed].max(initial=0.)


def main(directory, scores, output):
    values, labels = {}, {}
    for p, g, v in product(range(12,20), (0,1), ("BASE","D18")):
        with np.load(directory / f"p{p}-g{g}-{v}.npz") as tape:
            values[p,g,v] = float(narrowing(tape))
            labels[p,g,v] = bool(tape["hits"][:,g].any())
    centers = []
    for g in (0,1):
        centers.append([float(np.mean([x for key,x in values.items()
                                       if key[0]<16 and key[1]==g and labels[key]==hit])) for hit in (False,True)])
    def hit_score(key):
        miss, hit = [abs(values[key]-c) for c in centers[key[1]]]
        return miss/(miss+hit) if miss+hit else .5
    decision = json.loads(scores.read_text())["splits"]["development"]
    panels = {}
    for name, panel in decision["panels"].items():
        correct = 0.
        for row in decision["groups"]:
            a,b,q = row["A"],row["B"],row["query"]
            if not row["strict"] or [a,b,q] not in panel["triples"]:
                continue
            for c in row["cases"]:
                va,vb=c["variants"]
                sa,sb=hit_score((a,0,va)),hit_score((b,1,vb))
                if abs(sa-sb)<1e-12:correct+=.5
                else:correct+=int((sa>sb)==c["hidden"])
        panels[name]=dict(expected_correct=correct,denominator=panel["assignments"])
    dev=[k for k in values if k[0]>=16]
    result=dict(feature="Largest ten-control jaw narrowing with eleven consecutive close commands",
                fit_per_goal_centers_m=centers,development_completions=sum((hit_score(k)>.5)==labels[k] for k in dev),
                development_tapes=len(dev),development=panels,
                per_tape=[dict(pose=p,goal=g,variant=v,narrowing_m=x,hit_score=hit_score((p,g,v)))
                          for (p,g,v),x in values.items()])
    output.write_text(json.dumps(result,indent=2)+"\n")


if __name__ == "__main__":
    main(*map(Path,sys.argv[1:]))
