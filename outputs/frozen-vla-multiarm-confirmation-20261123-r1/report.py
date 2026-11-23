"""Native terminal task outcomes with explicit eligible denominators and missing cells."""
import argparse
import json
from pathlib import Path

from scipy.stats import binomtest

from screen import ARMS, METHODS, MATRIX_SHA, calibrations, cases, check_case, check_pair, historical, key, require, save


def rate(k, n):
    if not n:
        return dict(events=k, eligible=n, rate=None, exact_95=None)
    interval = binomtest(k, n).proportion_ci(method="exact")
    return dict(events=k, eligible=n, rate=k/n, exact_95=[interval.low, interval.high])


def contrast(comparison):
    """Comparison contains only complete reference/candidate pairs, one per identity."""
    n = len(comparison)
    counts = {pair: comparison.count(pair) for pair in ((False, False), (False, True), (True, False), (True, True))}
    ff, fs, sf, ss = counts.values()
    # Simultaneous binomial bounds for the two marginal discordance probabilities.
    # This remains non-degenerate when every observed difference is zero.
    bounds = None
    if n:
        gain = binomtest(fs, n).proportion_ci(confidence_level=.975, method="exact")
        loss = binomtest(sf, n).proportion_ci(confidence_level=.975, method="exact")
        bounds = [max(-1., gain.low - loss.high), min(1., gain.high - loss.low)]
    return dict(paired_identities=n, both_failed=ff, rescued=fs, harmed=sf, both_succeeded=ss,
                rescue_given_reference_failure=rate(fs, ff+fs), harm_given_reference_success=rate(sf, sf+ss),
                native_success_difference=(fs-sf)/n if n else None, conservative_paired_95=bounds)


def read_report(arm, out):
    rows = []
    mapping = calibrations(arm)
    history = historical(arm)
    for case in cases(arm):
        stem = key(case)
        row = mapping[stem]
        past = history[(case["suite"], case["task_id"])]
        loaded = {}
        for method in METHODS:
            path = out / f"{stem}-{method}.json"
            if not path.exists():
                continue
            obj = json.loads(path.read_bytes())
            require(obj["method"] == method and obj["historical_layout13_raw"] == past, "method or historical identity")
            check_case(obj, case)
            if method != "raw":
                require(obj["matrices_sha256"] == MATRIX_SHA and obj["calibration"] == row, "matrix or support readback")
                check_pair(loaded["raw"], obj, row, method)
            loaded[method] = obj
        rows.append(dict(case=case, historical_layout13_raw=past, support_sha256=row["support"]["sha256"],
                         native={m: dict(success=v["record"]["success"], controls=v["record"]["controls"],
                                         native_return=v["record"]["native_return"],
                                         terminated=v["record"]["terminated"], truncated=v["record"]["truncated"]) for m, v in loaded.items()},
                         missing=[m for m in METHODS if m not in loaded]))
    occupancy = {}
    for method in METHODS:
        observed = [r["native"][method] for r in rows if method in r["native"]]
        successes = sum(r["success"] for r in observed)
        controls = sum(r["controls"] for r in observed)
        rewarded = int(sum(r["native_return"] for r in observed))
        occupancy[method] = dict(successes=successes, failures=len(observed)-successes,
                                 observed=len(observed), planned=len(rows), native_controls=controls,
                                 zero_reward_controls=controls-rewarded, one_reward_controls=rewarded)
    pairs = {}
    for base, candidate in (("raw", "planar"), ("raw", "full"), ("planar", "full")):
        observed = [(r["native"][base]["success"], r["native"][candidate]["success"])
                    for r in rows if base in r["native"] and candidate in r["native"]]
        pairs[f"{candidate}_minus_{base}"] = contrast(observed)
    past_pairs = [(r["historical_layout13_raw"], r["native"]["raw"]["success"]) for r in rows if "raw" in r["native"]]
    return dict(arm=arm, complete=all(not r["missing"] for r in rows), task_rows=rows, native_occupancy=occupancy,
                raw_layout13_to_layout14=dict(historical_failure_still_failed=past_pairs.count((False, False)),
                    historical_failure_now_succeeded=past_pairs.count((False, True)),
                    historical_success_now_failed=past_pairs.count((True, False)),
                    historical_success_still_succeeded=past_pairs.count((True, True))),
                contrasts=pairs, uncertainty="Exact conditional binomial intervals for eligible rates. "
                "Conservative 95% difference bounds subtract simultaneous 97.5% Clopper-Pearson intervals "
                "for the two discordance probabilities. These sampling references assume independent, "
                "exchangeable task identities. The finite screen has eight fixed tasks and one layout per task.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=ARMS)
    parser.add_argument("out", type=Path)
    parser.add_argument("--partial", action="store_true", help="write a partial readback under its own filename")
    args = parser.parse_args()
    result = read_report(args.arm, args.out)
    require(args.partial or result["complete"], "missing native episodes; use --partial for an explicit partial readback")
    save(args.out / ("partial-summary.json" if args.partial else "summary.json"), result)
