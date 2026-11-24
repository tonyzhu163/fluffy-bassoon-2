"""Exact scalar calibration of the absolute circular loss on dyadic decoded yaws."""
from collections import Counter, defaultdict
from fractions import Fraction as Q

HALF = Q(1, 2)


def wrap(x):
    return (x + 180) % 360 - 180


def knots(rows, architecture):
    """Sweep integer-slope changes at true-yaw and antipodal crossings for each task.

    Between crossings each row is an affine loss. At a zero crossing its slope
    increases by 2|d|, and at a 180-degree crossing it decreases by 2|d|.
    The sum is continuous, so the intercept change is -knot * slope change.
    All arithmetic here, including objective ties, uses exact rational numbers.
    """
    slopes, intercepts, sizes = defaultdict(Q), defaultdict(Q), Counter()
    events = defaultdict(lambda: defaultdict(Q))
    for r in rows:
        task = r["task"]
        sizes[task] += 1
        a = wrap(Q(r["t"]) - Q(r["y"]))
        d = wrap(Q(r[architecture]) - Q(r["t"])) if r["tv"] and r[architecture+"v"] else 0
        intercepts[task] += abs(a)
        if not d or abs(d) == 180:
            continue
        slopes[task] += abs(d) if a == 0 else -abs(d) if a == -180 else d if a > 0 else -d
        for level in (-180, 0, 180):
            w = (level - a) / d
            if 0 < w < 1:
                events[w][task] += (2 if level == 0 else -2) * abs(d)
    tasks = sorted(sizes)
    candidates = sorted(set(events) | {Q(0), HALF, Q(1)})
    costs = []
    for w in candidates:
        costs.append(tuple((intercepts[t] + slopes[t] * w) / sizes[t] for t in tasks))
        for t, step in events.get(w, {}).items():
            slopes[t] += step
            intercepts[t] -= step * w
    return tasks, list(zip(candidates, costs, strict=True))


def optimum(table, counts):
    def rank(item):
        w, loss = item
        return sum(n*v for n,v in zip(counts, loss, strict=True)), abs(w-HALF), w
    weight, losses = min(table, key=rank)
    return weight, sum(n*v for n,v in zip(counts, losses, strict=True)) / sum(counts)
