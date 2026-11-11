"""Geometry-only check of a composable rule family for LIBERO-Gen.

A scene holds two bowls, three movable receptacles (plate, ramekin, cookie
box) and two fixed ones (stove, cabinet). A behaviour is (bowl, receptacle,
on or beside): 20 behaviours. A hidden rule is one choice on each of four
independent axes, all computed from positions:

    pick     which bowl: an image relation, or nearest/farthest from a landmark
    anchor   which receptacle: a named one, or a relation
    offset   on the receptacle, or beside it
    context  optionally, a scene predicate that flips the pick when true

Each axis is a list; extending the family means appending to one. No
simulator, motor or learner: this enumerates the rules exactly and reports how
fast one demonstration plus K attempts identifies the query behaviour.

    python placement.py --K 12
"""

from __future__ import annotations

import argparse

import numpy as np

REC = ("plate", "ramekin", "cookies", "stove", "cabinet")  # the last two never move
FIXED = np.array([[0.85, 0.80], [0.15, 0.90]])
N_BEH = 2 * len(REC) * 2
MIN_SEP = 0.12
CONDS = ("demo", "success bit", "bowl bit + target bit", "correction")
FAMILIES = {  # which axes vary; a fixed axis keeps its first entry
    "pick x anchor": (),
    "+ offset": ("offset",),
    "+ context": ("context",),
    "+ offset + context": ("offset", "context"),
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


def behaviours(bowl: np.ndarray, rec: np.ndarray, vary: tuple[str, ...]) -> tuple[np.ndarray, list[str]]:
    """The behaviour index [n, R] every rule names in every scene, and rule names."""
    n = len(bowl)
    # Axis 1, pick: [n, P] bowl index.
    pick, pick_names = [], []
    for axis, lo, hi in ((0, "leftmost", "rightmost"), (1, "lowermost", "uppermost")):
        first = bowl[:, :, axis].argmin(1)
        pick += [first, 1 - first]
        pick_names += [lo, hi]
    for j, name in enumerate(REC):
        near = np.linalg.norm(bowl - rec[:, j, None], axis=-1).argmin(1)
        pick += [near, 1 - near]
        pick_names += [f"nearest {name}", f"farthest {name}"]
    pick = np.stack(pick, 1)

    # Axis 4, context: [n, C] whether the pick flips.
    ctx = [np.zeros(n, bool)]
    ctx_names = [""]
    if "context" in vary:
        for a, b, axis, word in ((0, 1, 0, "left of"), (2, 0, 1, "below"), (1, 2, 0, "left of")):
            ctx.append(rec[:, a, axis] < rec[:, b, axis])
            ctx_names.append(f", other bowl if {REC[a]} {word} {REC[b]}")
    ctx = np.stack(ctx, 1)
    picked = pick[:, :, None] ^ ctx[:, None, :]  # [n, P, C]

    # Axis 2, anchor: [n, P, C, Q] receptacle index. Relations use the movable three.
    chosen = bowl[np.arange(n)[:, None, None], picked]  # [n, P, C, 2]
    d = np.linalg.norm(chosen[..., None, :] - rec[:, None, None, :3], axis=-1)
    x = np.broadcast_to(rec[:, None, None, :3, 0], d.shape)
    named = np.broadcast_to(np.arange(len(REC)), (*picked.shape, len(REC)))
    anchor = np.concatenate([named, np.stack([d.argmin(-1), d.argmax(-1), x.argmin(-1), x.argmax(-1)], -1)], -1)
    anchor_names = list(REC) + ["receptacle nearest it", "receptacle farthest from it", "leftmost receptacle", "rightmost receptacle"]

    # Axis 3, offset.
    offs = ("on", "beside") if "offset" in vary else ("on",)
    beh = (picked[..., None, None] * len(REC) + anchor[..., None]) * 2 + np.arange(len(offs))
    names = [f"{p} -> {o} {a}{c}" for p in pick_names for c in ctx_names for a in anchor_names for o in offs]
    return beh.reshape(n, -1), names


def run(rng: np.random.Generator, n: int, K: int, cond: str, vary: tuple[str, ...]):
    """Query accuracy and surviving rule count after k = 0..K greedy attempts, by exact enumeration."""
    T = K + 2
    bowl, rec = scenes(rng, n * T)
    beh, _ = behaviours(bowl, rec, vary)
    R = beh.shape[1]
    beh = beh.reshape(n, T, R)
    rule = rng.integers(0, R, n)
    rows = np.arange(n)
    true = beh[rows[:, None], np.arange(T)[None], rule[:, None]]  # [n, T]
    w = beh[:, 0] == true[:, 0, None]  # rules consistent with the demonstration
    acc, alive = [], []

    def best(t):
        mass = np.zeros((n, N_BEH))
        np.add.at(mass, (rows[:, None], beh[:, t]), w)
        return mass.argmax(1)

    for k in range(K + 1):
        if k > 0 and cond != "demo":
            c, tr, b = best(k), true[:, k], beh[:, k]
            if cond == "correction":
                ok = b == tr[:, None]
            elif cond == "success bit":
                hit = b == c[:, None]
                ok = np.where((c == tr)[:, None], hit, ~hit)
            else:  # which half of the attempt was right: the bowl, the place
                half = len(REC) * 2
                pb, pt = b // half == (c // half)[:, None], b % half == (c % half)[:, None]
                ok = (pb == (c // half == tr // half)[:, None]) & (pt == (c % half == tr % half)[:, None])
            w = w & ok
        acc.append(float((best(K + 1) == true[:, K + 1]).mean()))
        alive.append(float(w.sum(1).mean()))
    return acc, alive


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--n", type=int, default=4000)
    args = ap.parse_args()
    ks = [k for k in (0, 1, 2, 3, 4, 6, 8, 12, 16, 24) if k <= args.K]

    for family, vary in FAMILIES.items():
        bowl, rec = scenes(np.random.default_rng(0), 2000)
        beh, names = behaviours(bowl, rec, vary)
        agree = np.zeros((len(names), len(names)))
        for lo in range(0, len(beh), 100):
            agree += (beh[lo : lo + 100, :, None] == beh[lo : lo + 100, None, :]).sum(0)
        agree = np.triu(agree / len(beh), 1)
        i, j = np.unravel_index(agree.argmax(), agree.shape)
        used = np.mean([len(np.unique(b)) for b in beh])
        print(f"\n== {family}: {len(names)} rules, {used:.1f} of {N_BEH} behaviours named per scene")
        print(f"closest pair agrees in {agree.max():.2f} of scenes: [{names[i]}] vs [{names[j]}]")
        print(f"{'':<24}" + "".join(f"  k={k:<3}" for k in ks))
        for cond in CONDS:
            acc, alive = run(np.random.default_rng(1), args.n, args.K, cond, vary)
            print(f"{cond:<24}" + "".join(f"  {acc[k]:.3f}" for k in ks))
            if cond != "demo":
                print(f"{'  rules still possible':<24}" + "".join(f"  {alive[k]:5.1f}" for k in ks))


if __name__ == "__main__":
    main()
