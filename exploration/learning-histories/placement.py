"""Geometry-only check of a composable rule family for LIBERO-Gen.

A scene holds two bowls, three movable receptacles (plate, ramekin, cookie
box) and two fixed ones (stove, cabinet). A hidden rule is one choice on each
of six independent axes, all computed from positions:

    pick      which bowl: an image relation, or nearest/farthest from a landmark
    pick-if   optionally, a scene predicate that switches to the other bowl
    anchor    which receptacle: a named one, or a relation
    anchor-if optionally, a scene predicate that switches to the partner anchor
    offset    on the receptacle, or beside it on a fixed or scene-dependent side
    then      optionally, a second step: the other bowl goes on a receptacle

Each axis is a list; extending the family means appending to one. No
simulator, motor or learner: this enumerates the rules exactly and reports how
fast one demonstration plus K attempts identifies the query behaviour.

    python placement.py --K 16
"""

from __future__ import annotations

import argparse

import numpy as np

REC = ("plate", "ramekin", "cookies", "stove", "cabinet")  # the last two never move
FIXED = np.array([[0.85, 0.80], [0.15, 0.90]])
MIN_SEP = 0.12
N_NAMED = len(REC)
PARTNER = np.array([1, 2, 0, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11])  # anchor used when anchor-if holds
OPPOSITE = np.array([0, 2, 1, 4, 3])  # placements: on, left, right, above, below
N_PLACE, N_THEN = 5, 1 + N_NAMED
N_BEH = 2 * N_NAMED * N_PLACE * N_THEN

# name: (axes that vary beyond pick, pick-if and anchor; balanced anchor prior; sessions)
FAMILIES = {
    "pick, pick-if, anchor; uniform prior": ((), False, 2000),
    "same; named and relational anchors equally likely": ((), True, 2000),
    "+ anchor-if": (("anchor-if",), True, 2000),
    "+ anchor-if + offset": (("anchor-if", "offset"), True, 1000),
    "+ anchor-if + then": (("anchor-if", "then"), True, 1000),
    "all six axes": (("anchor-if", "offset", "then"), True, 300),
}


def scenes(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Bowl positions [n, 2, 2] and receptacle positions [n, 5, 2], rejection-sampled apart."""
    out = np.empty((0, 7, 2))
    while len(out) < n:
        p = np.concatenate([rng.random((2 * n, 5, 2)), np.broadcast_to(FIXED, (2 * n, 2, 2))], 1)
        d = np.linalg.norm(p[:, :, None] - p[:, None], axis=-1) + np.eye(7)
        out = np.concatenate([out, p[d.min((1, 2)) > MIN_SEP]])
    out = out[:n]
    return out[:, :2], out[:, 2:]


def behaviours(bowl: np.ndarray, rec: np.ndarray, vary: tuple[str, ...], balanced: bool):
    """The behaviour index [n, R] every rule names in every scene, and the rule prior [R]."""
    n = len(bowl)
    rows = np.arange(n)
    lt = lambda a, b, axis: rec[:, a, axis] < rec[:, b, axis]
    never = np.zeros(n, bool)

    # pick [n, P]
    pick = []
    for axis in (0, 1):
        first = bowl[:, :, axis].argmin(1)
        pick += [first, 1 - first]
    for j in range(N_NAMED):
        near = np.linalg.norm(bowl - rec[:, j, None], axis=-1).argmin(1)
        pick += [near, 1 - near]
    pick = np.stack(pick, 1)

    # pick-if [n, Cp]
    pick_if = np.stack([never, lt(0, 1, 0), lt(2, 0, 1), lt(1, 2, 0)], 1)
    picked = pick[:, :, None] ^ pick_if[:, None, :]
    own = bowl[rows[:, None, None], picked]  # [n, P, Cp, 2]
    other = bowl[rows[:, None, None], 1 - picked]

    # anchor [n, P, Cp, Q]: five named, then relations over the movable three
    movable = rec[:, None, None, :3]
    d_own = np.linalg.norm(own[..., None, :] - movable, axis=-1)
    d_other = np.linalg.norm(other[..., None, :] - movable, axis=-1)
    x = np.broadcast_to(movable[..., 0], d_own.shape)
    y = np.broadcast_to(movable[..., 1], d_own.shape)
    named = np.broadcast_to(np.arange(N_NAMED), (*picked.shape, N_NAMED))
    relational = [d_own.argmin(-1), d_own.argmax(-1), x.argmin(-1), x.argmax(-1),
                  y.argmin(-1), y.argmax(-1), d_other.argmin(-1), d_other.argmax(-1)]
    anchor = np.concatenate([named, np.stack(relational, -1)], -1)

    # anchor-if [n, Ca]
    anchor_if = np.stack([never, lt(0, 1, 1), lt(2, 0, 0), lt(1, 2, 1)], 1)
    if "anchor-if" not in vary:
        anchor_if = anchor_if[:, :1]
    anchor = np.where(anchor_if[:, None, None, None, :], anchor[..., PARTNER, None], anchor[..., None])

    # offset [n, P, Cp, Q, Ca, O]: on; four fixed sides; toward or away from each bowl's start
    place = np.zeros((*anchor.shape, 1), int)
    if "offset" in vary:
        at = rec[rows[:, None, None, None, None], anchor]

        def side(ref):
            v = ref[:, :, :, None, None, :] - at
            horizontal = np.abs(v[..., 0]) >= np.abs(v[..., 1])
            return np.where(horizontal, np.where(v[..., 0] > 0, 2, 1), np.where(v[..., 1] > 0, 3, 4))

        to_own, to_other = side(own), side(other)
        fixed = [np.full(anchor.shape, s) for s in range(N_PLACE)]
        place = np.stack(fixed + [to_own, OPPOSITE[to_own], to_other, OPPOSITE[to_other]], -1)

    # then [n, P, Cp, S]: nothing, or the other bowl on a named or relational receptacle
    then = np.zeros((*picked.shape, 1), int)
    if "then" in vary:
        second = np.stack([d_other.argmin(-1), d_other.argmax(-1), x.argmin(-1), x.argmax(-1)], -1)
        then = np.concatenate([then, 1 + named, 1 + second], -1)

    first = (picked[..., None, None] * N_NAMED + anchor).astype(np.int16)
    beh = (first[..., None] * N_PLACE + place.astype(np.int16))[..., None] * N_THEN
    beh = beh + then[:, :, :, None, None, None, :].astype(np.int16)

    prior = np.ones(beh.shape[1:])
    if balanced:  # half the mass on named anchors, half on relational ones
        q = np.arange(anchor.shape[3]) < N_NAMED
        prior = prior * np.where(q, 1 / q.sum(), 1 / (~q).sum())[None, None, :, None, None, None]
    return beh.reshape(n, -1), (prior / prior.sum()).ravel()


def run(rng: np.random.Generator, n: int, K: int, vary: tuple[str, ...], balanced: bool):
    """Per feedback type: query accuracy and surviving rules after k = 0..K greedy attempts."""
    T = K + 2
    bowl, rec = scenes(rng, n * T)
    bowl, rec = bowl.reshape(n, T, 2, 2), rec.reshape(n, T, 5, 2)
    at = lambda t: behaviours(bowl[:, t], rec[:, t], vary, balanced)
    rows = np.arange(n)

    query, prior = at(K + 1)
    R = len(prior)
    rule = rng.choice(R, n, p=prior)
    demo = at(0)[0]
    start = demo == demo[rows, rule, None]

    def best(b, w):
        flat = (rows[:, None] * N_BEH + b)[w]
        mass = np.bincount(flat, np.broadcast_to(prior, w.shape)[w], n * N_BEH)
        return mass.reshape(n, N_BEH).argmax(1)

    alive = {"success bit": start, "correction": start.copy()}
    acc = {c: [] for c in alive}
    left = {c: [] for c in alive}
    for k in range(K + 1):
        if k > 0:
            b = at(k)[0]
            tr = b[rows, rule]
            c = best(b, alive["success bit"])
            hit = b == c[:, None]
            alive["success bit"] = alive["success bit"] & np.where((c == tr)[:, None], hit, ~hit)
            alive["correction"] = alive["correction"] & (b == tr[:, None])
        for cond, w in alive.items():
            acc[cond].append(float((best(query, w) == query[rows, rule]).mean()))
            left[cond].append(float(w.sum(1).mean()))
    return R, acc, left


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every family's session count")
    args = ap.parse_args()
    ks = [k for k in (0, 2, 4, 6, 8, 12, 16, 24) if k <= args.K]
    print(f"{N_BEH} behaviours at most, chance {1 / N_BEH:.4f}")
    for family, (vary, balanced, n) in FAMILIES.items():
        n = max(50, int(n * args.scale))
        R, acc, left = run(np.random.default_rng(1), n, args.K, vary, balanced)
        print(f"\n== {family}: {R} rules, {n} sessions")
        print(f"{'':<24}" + "".join(f"  k={k:<5}" for k in ks))
        for cond in acc:
            print(f"{cond:<24}" + "".join(f"  {acc[cond][k]:<7.3f}" for k in ks))
            print(f"{'  rules still possible':<24}" + "".join(f"  {left[cond][k]:<7.1f}" for k in ks), flush=True)


if __name__ == "__main__":
    main()
