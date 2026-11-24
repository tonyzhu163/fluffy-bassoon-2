"""Synthetic finite-sample checks and power sensitivity; no target outcomes."""

import json
import math
import os
import sys
from fractions import Fraction
from itertools import product
from pathlib import Path

import numpy as np
import scipy
from scipy.signal import fftconvolve
from scipy.stats import binom, binomtest

from analyze import (case_seed, check_manifest, cp, decide, digest, exact_p, interval, read_endpoint, require, spend_at,
                     summary, validate_rows)


def rejected(function, *args):
    try:
        function(*args)
    except (ValueError, KeyError):
        return
    raise AssertionError("invalid input accepted")


def synthetic_rows(modes):
    return [dict(case_id=i, layout_seed=case_seed("layout", i), noise_seed=case_seed("noise", i),
                 scene_hash=f"{i:064x}", movable_xy_hash=f"{i+1:064x}", stock=s, panda=p, mixed=m)
            for i, (s, p, m) in enumerate(modes)]


def endpoints():
    def tape(n, win):
        bits = [0] * (n - 1) + [win]
        return dict(reward=bits.copy(), success=bits.copy(), terminated=bits.copy(),
                    truncated=[0]*(n-1) + [1-win], worker_step_index=list(range(1, n+1)))
    good = [(1, 1), (35, 1), (300, 1), (300, 0)]
    for n, win in good:
        assert read_endpoint(tape(n, win)) == win
    rejected(read_endpoint, tape(299, 0))
    for key, idx in (("reward", 1), ("terminated", 0), ("truncated", -1), ("success", -1), ("worker_step_index", 1)):
        corrupted = tape(300, 1)
        corrupted[key][idx] = 1 - corrupted[key][idx]
        rejected(read_endpoint, corrupted)
    return good


def gate_checks():
    tie = (0, 0, 0)
    rows = synthetic_rows([tie] * 64)
    assert decide([rows])["decision"] == "continue_primary"
    assert "test" not in decide([rows])["looks"][0]["secondary"]
    assert decide([synthetic_rows([(1, 0, 0)] * 9 + [tie] * 55)])["decision"] == "primary_harm"
    # Primary benefit at N=64; Panda and MIXED have tied on every case to this point.
    rows = synthetic_rows([(0, 1, 1)] * 9 + [tie] * 55 + [(0, 0, 1)] * 9 + [tie] * 55)
    assert decide([rows[:64]])["decision"] == "continue_secondary"
    result = decide([rows[:64], rows])
    assert result["decision"] == "both_benefit" and result["crossings"]["secondary"]["n"] == 128
    # Early secondary benefit disappears at 128. It must be latched when primary opens at 128.
    rows = synthetic_rows([(1, 0, 1)] * 9 + [tie] * 55 + [(0, 0, 1)] * 15 + [(0, 1, 0)] * 24 + [tie] * 25)
    result = decide([rows[:64], rows])
    assert result["crossings"]["secondary"] == {"n": 64, "direction": "benefit", "authorized_at": 128}
    assert result["decision"] == "both_benefit"
    rows = synthetic_rows([(1, 1, 1)] * 52 + [(0, 1, 1)] * 12 + [(0, 1, 0)] * 20 + [tie] * 44)
    assert decide([rows[:64], rows])["decision"] == "primary_benefit_secondary_harm"
    rejected(decide, [rows[:63]])
    rejected(decide, [rows])
    rejected(decide, [rows[:64], rows, rows * 2])
    mixed_cells = synthetic_rows(list(product(range(2), repeat=3)) * 8)
    stat = summary(mixed_cells, "stock", spend_at(64))
    assert [stat[k] for k in ("both_succeed", "both_fail", "rescues", "losses")] == [16] * 4
    validate_rows(mixed_cells, 64)
    for name in ("layout_seed", "noise_seed", "scene_hash", "movable_xy_hash", "case_id"):
        bad = [*mixed_cells[:-1], {**mixed_cells[-1], name: mixed_cells[0][name]}]
        rejected(validate_rows, bad, 64)
    for bad_val in (None, True, 3, "0"):
        rejected(validate_rows, [*mixed_cells[:-1], {**mixed_cells[-1], "panda": bad_val}], 64)
    rejected(validate_rows, mixed_cells[:-1], 64)
    return result["crossings"]


def scenarios(delta, discordance):
    """Trinomial increments, propagated only on paths still undecided at the previous look."""
    live, old_n = np.ones((1, 1)), 0
    benefit = harm = 0.0
    rows = []
    for n in (64, 128, 256):
        step = n - old_n
        b, c = np.indices((step + 1, step + 1))
        d = b + c
        sign = (1 + delta/discordance)/2 if discordance else 0.5
        increments = binom.pmf(d, step, discordance) * binom.pmf(b, d, sign)
        propagated = fftconvolve(live, increments, mode="full")
        b, c = np.indices((n+1, n+1))
        propagated[b + c > n] = 0
        live = np.maximum(0, propagated)
        p = 2 * binom.cdf(np.minimum(b, c), b+c, 0.5)
        crossing = p <= float(spend_at(n))
        benefit += float(live[crossing & (b > c)].sum())
        harm += float(live[crossing & (b < c)].sum())
        live[crossing] = 0
        remaining = float(live.sum())
        assert math.isclose(benefit + harm + remaining, 1, abs_tol=2e-11)
        rows.append(dict(n=n, positive_by_look=benefit, negative_by_look=harm, unresolved=remaining))
        old_n = n
    return rows


def reference_checks():
    compared = 0
    for d in range(1, 65):
        for b in range(d+1):
            result = binomtest(b, d)
            assert abs(float(exact_p(b, d-b)) - result.pvalue) < 1e-13
            compared += 1
    assert exact_p(0, 0) == 1 and exact_p(8, 0) > spend_at(64) >= exact_p(9, 0)
    assert exact_p(7, 0) > spend_at(256)
    assert spend_at(1024) == Fraction(3, 400) and spend_at(4096) == Fraction(3, 1600)
    assert sum(spend_at(64 * 2**i) for i in range(5)) + 2*spend_at(2048) == Fraction(1, 20)
    assert cp(0, 0, .01) == (0, 1)
    max_cp_error = 0.0
    for k, n, alpha in ((0, 64, .005/4), (64, 64, .005/4), (3, 17, .01/4), (37, 128, .01/4)):
        ref = binomtest(k, n).proportion_ci(1-alpha, method="exact")
        max_cp_error = max(max_cp_error, *(abs(x-y) for x,y in zip(cp(k,n,alpha), ref, strict=True)))
    assert max_cp_error < 5e-11
    for b, c in ((0, 0), (11, 0), (0, 11), (32, 32), (61, 3)):
        bounds = interval(b,c,64,spend_at(64))["risk_difference"]
        reverse = interval(c,b,64,spend_at(64))["risk_difference"]
        assert bounds[0] <= (b-c)/64 <= bounds[1] and np.allclose(bounds, [-reverse[1],-reverse[0]])
    return {"exact_binomial_reference_tables": compared, "maximum_CP_absolute_error": max_cp_error}


def main():
    require(sys.platform.startswith("linux") and len(os.sched_getaffinity(0)) <= 2, "use bounded Linux CPU")
    path = Path(sys.argv[1])
    manifest = check_manifest(path)
    physical = dict(placements=len(manifest["cases"]), rejected=len(manifest["rejected"]),
                    sha256=digest(path), prior_sha256=manifest["prior_sha256"])
    null = {str(q): scenarios(0, q) for q in (0, 1/12, 1/4, 1/2, 1)}
    for sequence in null.values():
        for j, r in enumerate(sequence):
            spent = sum(float(spend_at(64*2**i)) for i in range(j+1))
            assert r["positive_by_look"] + r["negative_by_look"] <= spent + 1e-12
            assert math.isclose(r["positive_by_look"], r["negative_by_look"], abs_tol=1e-12)
    planning = {str(q): scenarios(1/12, q) for q in (1/12, 1/4, 1/2)}
    for r in planning[str(1/12)]:
        assert r["negative_by_look"] < 1e-12
    print(json.dumps(dict(physical=physical, reference=reference_checks(), endpoint_examples=endpoints(), gate_example=gate_checks(),
                          null_equal_margins=null, planning_delta_1_over_12=planning,
                          scipy=scipy.__version__, python=sys.version, cores=sorted(os.sched_getaffinity(0)),
                          host=os.uname().nodename, pin=os.environ["STATS_PIN"]), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
