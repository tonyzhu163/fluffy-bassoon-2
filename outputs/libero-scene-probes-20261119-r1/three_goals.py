"""Post hoc two-support replay on all triplets of the four fitting goals."""

import argparse
from itertools import combinations, product
import json
from pathlib import Path

import numpy as np

from report import KNOWN, load_panel, next_scene


def examples(panel, goals):
    """Reward and query-success rows index the same hidden goal on three fresh poses."""
    rows = []
    for (pose, layout), block in sorted(panel.items()):
        second_pose = next_scene(panel, pose)
        query_pose = next_scene(panel, second_pose)
        assert len({pose, second_pose, query_pose}) == 3
        blocks = (block, panel[second_pose, layout], panel[query_pose, layout])
        for hidden in goals:
            hits = [[int(b['records'][g][str(hidden)] is not None) for g in goals] for b in blocks]
            rows.append((pose, *hits))
    return rows


def history(row, tree, bits):
    if bits == 0:
        return ()
    _, first_hits, second_hits, _ = row
    first, on_miss, on_hit = tree
    reward = first_hits[first]
    if bits == 1:
        return (reward,)
    second = (on_miss, on_hit)[reward]
    return (reward, second_hits[second])


def maximizers(counts):
    return np.flatnonzero(counts == counts.max()).tolist()


def fit_decoder(rows, tree, bits):
    counts = {h: np.zeros(3, dtype=int) for h in product(range(2), repeat=bits)}
    seen = set()
    for row in rows:
        h = history(row, tree, bits)
        seen.add(h)
        counts[h] += row[-1]
    marginal = sum(counts.values())
    decoder = {h: maximizers(v if h in seen else marginal) for h, v in counts.items()}
    return sum(int(v.max()) for v in counts.values()), decoder


def fit(rows, bits, adaptive):
    # A single representative suffices for decisions invisible to the decoder.
    trees = list(product(range(3), repeat=3))
    if bits < 2:
        trees = [(a, a, a) for a in range(3 if bits else 1)]
    elif not adaptive:
        trees = [t for t in trees if t[1] == t[2]]
    candidates = [(t, *fit_decoder(rows, t, bits)) for t in trees]
    best = max(score for _, score, _ in candidates)
    winners = [(t, decoder) for t, score, decoder in candidates if score == best]
    # If conditioning the second probe buys nothing on source, use fixed schedules.
    fixed = [(t, decoder) for t, decoder in winners if t[1] == t[2]]
    return best, len(winners), fixed or winners


def evaluate(source, fresh, goals, bits, adaptive=False):
    best, ties, policies = fit(source, bits, adaptive)
    by_pose = {}
    for row in fresh:
        scores = []
        for tree, decoder in policies:
            choices = decoder[history(row, tree, bits)]
            scores.append(sum(row[-1][a] for a in choices) / len(choices))
        pose = row[0]
        by_pose[pose] = by_pose.get(pose, 0) + sum(scores) / len(scores)
    serialized = []
    for tree, decoder in policies:
        serialized.append(dict(probes=[goals[i] for i in tree],
                               query={' '.join(map(str, h)): [goals[i] for i in choices]
                                      for h, choices in decoder.items()}))
    return dict(fit=best, fresh=sum(by_pose.values()), by_pose=by_pose,
                tied_before_simplicity=ties, policies=serialized)


def summarize(root):
    source = load_panel(root / 'fitting')
    fresh = load_panel(root / 'fresh')
    panels = []
    modes = {'blind': (0, False), 'one_reward': (1, False),
             'fixed_two': (2, False), 'adaptive_two': (2, True)}
    for goals in combinations(KNOWN, 3):
        train, test = examples(source, goals), examples(fresh, goals)
        methods = {name: evaluate(train, test, goals, *spec) for name, spec in modes.items()}
        ceiling = sum(max(row[-1]) for row in test)
        panels.append(dict(goals=goals, assignments=len(test), query_ceiling=ceiling, methods=methods))
    totals = {name: {split: sum(p['methods'][name][split] for p in panels)
                     for split in ('fit', 'fresh')} for name in modes}
    return dict(assignments=sum(p['assignments'] for p in panels), totals=totals,
                query_ceiling=sum(p['query_ceiling'] for p in panels), panels=panels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episodes', type=Path)
    print(json.dumps(summarize(parser.parse_args().episodes), indent=2))
