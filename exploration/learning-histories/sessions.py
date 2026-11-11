"""Hidden-rule selection sessions and their Bayes-optimal learner.

A session hides one rule: "pick the object whose attribute d is closest to v",
or the same over two attributes with summed distance. Every scene holds N
objects with A continuous attributes, so every rule names exactly one object in
every scene and a scene by itself says nothing about the rule. The session is
one teacher demonstration, K learner attempts in fresh scenes, then a query in
another fresh scene.

Execution is perfect by construction: a choice always lands on the chosen
object. Every failure is therefore a wrong choice, never a motor failure.

Run `python sessions.py` for the oracle's accuracy under each context
condition. It needs only numpy.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

A, V, N = 3, 6, 5  # attribute dimensions, anchor values per attribute, objects per scene
F = A + 5  # per-object feature width, see `encode`

# What the learner is shown about each attempt scene.
CONDS = (
    "demo",  # nothing: the first demonstration only
    "blind",  # the learner's choice, without its outcome
    "attempts",  # the choice and whether it succeeded
    "corrections",  # the teacher's choice only
    "linked",  # the choice, its outcome and the teacher's choice
)


def _rules() -> tuple[np.ndarray, np.ndarray]:
    mask, val = [], []
    for a in range(A):
        for v in range(V):
            m, x = np.zeros(A, bool), np.zeros(A)
            m[a], x[a] = True, (v + 0.5) / V
            mask.append(m), val.append(x)
    for a in range(A):
        for b in range(a + 1, A):
            for v in range(V):
                for u in range(V):
                    m, x = np.zeros(A, bool), np.zeros(A)
                    m[[a, b]], x[a], x[b] = True, (v + 0.5) / V, (u + 0.5) / V
                    mask.append(m), val.append(x)
    return np.array(mask), np.array(val)


RULE_MASK, RULE_VAL = _rules()
R = len(RULE_MASK)
ARITY = RULE_MASK.sum(1)
# A quarter of the conjunctions never appear in training.
_perm = np.random.default_rng(0).permutation(np.flatnonzero(ARITY == 2))
HELDOUT = np.sort(_perm[: len(_perm) // 4])
TRAIN = np.setdiff1d(np.arange(R), HELDOUT)


@dataclass
class Sessions:
    rule: np.ndarray  # [B] hidden rule index
    scenes: np.ndarray  # [B, T, N, A] attribute values in [0, 1); T = K + 2
    tgt: np.ndarray  # [B, T, R] the object each rule names
    true: np.ndarray  # [B, T] the object the hidden rule names


def sample(rng: np.random.Generator, B: int, K: int, split: str) -> Sessions:
    rule = rng.choice(TRAIN if split == "train" else HELDOUT, B)
    T = K + 2
    scenes = rng.random((B, T, N, A))
    dist = np.abs(scenes[:, :, None] - RULE_VAL[:, None, :]) * RULE_MASK[:, None, :]
    tgt = dist.sum(-1).argmin(-1)
    true = tgt[np.arange(B)[:, None], np.arange(T)[None], rule[:, None]]
    return Sessions(rule, scenes, tgt, true)


# --- Bayes-optimal learner -------------------------------------------------


def event_ll(s: Sessions, t: int, cond: str, choice, outcome) -> np.ndarray | float:
    """Log-likelihood [B, R] of what `cond` reveals about scene t."""
    tg = s.tgt[:, t]
    if t == 0 or cond in ("corrections", "linked"):
        ok = tg == s.true[:, t, None]
    elif cond == "attempts":
        hit = tg == choice[:, None]
        ok = np.where(outcome[:, None], hit, ~hit)
    else:
        return 0.0
    return np.where(ok, 0.0, -np.inf)


def history_logw(s: Sessions, cond: str, choice, outcome, k: int) -> np.ndarray:
    """Unnormalised log-posterior over rules after the first k attempt events."""
    lw = event_ll(s, 0, cond, None, None) + np.zeros((len(s.rule), R))
    if cond != "demo":
        for t in range(1, k + 1):
            lw = lw + event_ll(s, t, cond, choice[:, t], outcome[:, t])
    return lw


def predict(s: Sessions, logw: np.ndarray, q: int) -> tuple[np.ndarray, np.ndarray]:
    """Posterior over the N objects of scene q, and the rule weights behind it."""
    w = np.exp(logw - logw.max(1, keepdims=True))
    probs = np.stack([(w * (s.tgt[:, q] == o)).sum(1) for o in range(N)], 1)
    return probs / probs.sum(1, keepdims=True), w


def oracle_act(s: Sessions, cond: str):
    def act(choice, outcome, k, q):
        return predict(s, history_logw(s, cond, choice, outcome, k), q)[0].argmax(1)

    return act


def behave(rng: np.random.Generator, s: Sessions, K: int, eps: float = 0.2):
    """Training attempts: sample a rule from the posterior and act on it.

    The actor executes perfectly but does not know the rule. With probability
    `eps` it picks a random object instead.
    """
    B = len(s.rule)
    choice = np.zeros((B, K + 2), int)
    outcome = np.zeros((B, K + 2), bool)
    for t in range(1, K + 1):
        _, w = predict(s, history_logw(s, "attempts", choice, outcome, t - 1), t)
        u = rng.random((B, 1)) * w.sum(1, keepdims=True)
        r = (w.cumsum(1) > u).argmax(1)
        c = s.tgt[np.arange(B), t, r]
        rand = rng.random(B) < eps
        c[rand] = rng.integers(0, N, int(rand.sum()))
        choice[:, t], outcome[:, t] = c, c == s.true[:, t]
    return choice, outcome


def rollout(s: Sessions, K: int, cond: str, act):
    """Let `act` make its own attempts, scoring the query after each one.

    `act(choice, outcome, k, q)` returns [B] choices on scene q given the first
    k events. Returns the actor's query accuracy per k, and the accuracy of the
    Bayes-optimal learner reading the same histories.
    """
    B, q = len(s.rule), K + 1
    choice = np.zeros((B, K + 2), int)
    outcome = np.zeros((B, K + 2), bool)
    acc, bayes = [], []
    for k in range(K + 1):
        if k > 0:
            c = act(choice, outcome, k - 1, k)
            choice[:, k], outcome[:, k] = c, c == s.true[:, k]
        acc.append(float((act(choice, outcome, k, q) == s.true[:, q]).mean()))
        best = predict(s, history_logw(s, cond, choice, outcome, k), q)[0].argmax(1)
        bayes.append(float((best == s.true[:, q]).mean()))
    return acc, bayes


# --- Learner input ---------------------------------------------------------


def encode(s: Sessions, cond: str, choice, outcome, k, q: int):
    """Features [B, T, N, F] and a scene mask [B, T] (True = hidden).

    `k` is the number of visible attempt events, an int or a [B, 1] array.
    Scene `q` is the query. Per object: its attributes, then learner chose,
    teacher chose, chosen-and-failed, chosen-and-succeeded, belongs-to-query.
    """
    B, T = s.true.shape
    x = np.zeros((B, T, N, F), np.float32)
    x[..., :A] = s.scenes
    ts = np.arange(T)[None, :]
    hist = (ts <= k) & (ts != q)
    if cond == "demo":
        hist = hist & (ts == 0)
    att = hist & (ts >= 1)
    learner = att & (cond in ("blind", "attempts", "linked"))
    feedback = att & (cond in ("attempts", "linked"))
    teacher = (hist & (ts == 0)) | (att & (cond in ("corrections", "linked")))
    lc = choice[..., None] == np.arange(N)
    tc = s.true[..., None] == np.arange(N)
    x[..., A] = lc & learner[..., None]
    x[..., A + 1] = tc & teacher[..., None]
    x[..., A + 2] = lc & (feedback & ~outcome)[..., None]
    x[..., A + 3] = lc & (feedback & outcome)[..., None]
    x[:, q, :, A + 4] = 1
    return x, np.broadcast_to(~(hist | (ts == q)), (B, T)).copy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--n", type=int, default=20000)
    args = ap.parse_args()
    print(f"{R} rules ({len(TRAIN)} train, {len(HELDOUT)} held out), chance {1 / N:.2f}")
    print("Oracle query accuracy after k attempts (own greedy attempts)")
    for split in ("train", "heldout"):
        s = sample(np.random.default_rng(1), args.n, args.K, split)
        print(f"\n{split:<12}" + "".join(f"  k={k}  " for k in range(args.K + 1)))
        for cond in CONDS:
            acc, _ = rollout(s, args.K, cond, oracle_act(s, cond))
            print(f"{cond:<12}" + "".join(f"  {a:.3f}" for a in acc))


if __name__ == "__main__":
    main()
