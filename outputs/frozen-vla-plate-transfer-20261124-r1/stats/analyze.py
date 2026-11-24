"""Read independently sealed cumulative native triplets and apply the registered sequence."""

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from scipy.stats import beta


MODES = ("stock", "panda", "mixed")
SHA = re.compile("[0-9a-f]{64}")


def require(check, message):
    if not check:
        raise ValueError(message)


def spend_at(n):
    require(n >= 64 and n & (n - 1) == 0, "not a scheduled look")
    if n == 64:
        return Fraction(1, 200)
    if n <= 512:
        return Fraction(1, 100)
    return Fraction(3 * 1024, 400 * n)


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def valid_hash(value):
    return isinstance(value, str) and SHA.fullmatch(value) is not None


def exact_p(rescues, losses):
    d = rescues + losses
    numerator = 2 * sum(math.comb(d, j) for j in range(min(rescues, losses) + 1))
    return min(Fraction(1), Fraction(numerator, 2**d))


def cp(k, n, alpha):
    if n == 0:
        return (0.0, 1.0)
    lower = beta.ppf(alpha / 2, k, n - k + 1) if k else 0.0
    upper = beta.isf(alpha / 2, n - k, k + 1) if k < n else 1.0
    return float(lower), float(upper)


def interval(b, c, n, spend):
    alpha = float(spend) / 4
    q = cp(b + c, n, alpha)
    s = cp(b, b + c, alpha)
    corners = [rate * (2 * sign - 1) for rate in q for sign in s]
    return {"risk_difference": [min(corners), max(corners)], "discordance": q, "win_given_discordance": s}


def summary(rows, comparator, spend):
    n = len(rows)
    cells = Counter((row["mixed"], row[comparator]) for row in rows)
    b, c = cells[1, 0], cells[0, 1]
    totals = {"mixed": cells[1, 1] + b, comparator: cells[1, 1] + c}
    return dict(n=n, both_succeed=cells[1, 1], both_fail=cells[0, 0], rescues=b, losses=c,
                successes=totals, rates={mode: count/n for mode, count in totals.items()},
                paired_risk_difference=(b-c)/n, simultaneous_95=interval(b, c, n, spend))


def run_test(stat, spend, running_p, authorized_at):
    p = exact_p(stat["rescues"], stat["losses"])
    adjusted = min(running_p, Fraction(1, 20) * p / spend)
    test = {"p_two_sided": float(p), "alpha_increment": float(spend),
            "across_look_adjusted_p": float(adjusted), "authorized_at": authorized_at}
    crossing = None
    if p <= spend:
        direction = "benefit" if stat["rescues"] > stat["losses"] else "harm"
        crossing = dict(n=stat["n"], direction=direction, authorized_at=authorized_at)
    stat["test"] = test
    return adjusted, crossing


def decide(prefixes):
    """Replay the chronological gate and latch the first crossing for each comparison."""
    history, crossings = [], {"primary": None, "secondary": None}
    primary_p = secondary_p = Fraction(1)
    next_secondary = 0
    decision = "continue_primary"
    for rows in prefixes:
        require(decision.startswith("continue_"), "received an extra look after a conclusive stop")
        n = len(rows)
        require(n == 64 * 2**len(history), "requires every registered look in order")
        alpha = spend_at(n)
        look = {name: summary(rows, comparator, alpha)
                for name, comparator in (("primary", "stock"), ("secondary", "panda"))}
        history.append(look)
        if crossings["primary"] is None:
            primary_p, crossings["primary"] = run_test(look["primary"], alpha, primary_p, n)
        if crossings["primary"] is None:
            decision = "continue_primary"
        elif crossings["primary"]["direction"] == "harm":
            decision = "primary_harm"
        else:
            while next_secondary < len(history) and crossings["secondary"] is None:
                stat = history[next_secondary]["secondary"]
                secondary_p, crossings["secondary"] = run_test(stat, spend_at(stat["n"]), secondary_p, n)
                next_secondary += 1
            if crossings["secondary"] is None:
                decision = "continue_secondary"
            elif crossings["secondary"]["direction"] == "benefit":
                decision = "both_benefit"
            else:
                decision = "primary_benefit_secondary_harm"
    return dict(decision=decision, crossings=crossings, looks=history)


DOMAIN = "fluffy-bassoon/frozen-vla-plate-transfer-20261124-r1/v1"
GEOMETRY = ("initial_state_sha256", "scene_hash", "movable_xy_hash", "mobile_joint_xy_hash")


def case_seed(kind, index, draw=0):
    raw = hashlib.sha256(f"{DOMAIN}/{kind}/{index}/{draw}".encode()).digest()
    integer = int.from_bytes(raw[:8], "big")
    return 2**30 + integer % 2**30 if kind == "layout" else integer


def check_manifest(path):
    doc = load(path)
    require(doc["domain"] == DOMAIN, "wrong generator domain")
    prior_path = path.with_name("prior.json")
    require(digest(prior_path) == doc["prior_sha256"], "denied geometry inventory changed")
    prior = load(prior_path)["fingerprints"]
    seen = {key: set(prior[key]) for key in (*GEOMETRY, "layout_seed")}
    require(all(seen.values()), "empty pre-outcome denylist")
    noise_used = set()
    rejected = defaultdict(list)
    for item in doc["rejected"]:
        rejected[item["index"]].append(item)
    for i, row in enumerate(doc["cases"]):
        require(row["index"] == i and row["shard"] == i % 4 and row["split"] == "target", "bad enrollment ID")
        require((row["suite"], row["task_id"], row["case"]) == ("libero_goal", 5, "BASE"), "wrong task")
        require(row["collision_free"] is True and row["released_rows_checked"] == 50, "admission missing")
        rejected_here = rejected.pop(i, [])
        require(row["draw"] == len(rejected_here), "candidate skipped")
        candidates = [item["seed"] for item in rejected_here] + [row["layout_seed"]]
        for j, item in enumerate(rejected_here):
            require(item["draw"] == j and item["reason"], "missing native rejection reason")
        for j, candidate in enumerate(candidates):
            require(candidate == case_seed("layout", i, j) and candidate not in seen["layout_seed"],
                    "layout seed differs or repeats within enrolled/prior layouts")
            seen["layout_seed"].add(candidate)
        require(row["noise_seed"] == case_seed("noise", i) and row["noise_seed"] not in noise_used,
                "noise stream differs or repeats")
        noise_used.add(row["noise_seed"])
        for key in GEOMETRY:
            require(valid_hash(row[key]) and row[key] not in seen[key], f"physical overlap or invalid {key}")
            seen[key].add(row[key])
    require(not rejected, "unexpected extra candidate")
    return doc


def validate_rows(rows, n):
    require(len(rows) == n, "incomplete or extra rows")
    for i, row in enumerate(rows):
        require(type(row["case_id"]) is int and row["case_id"] == i, "nonconsecutive ID or wrong order")
        for field in ("scene_hash", "movable_xy_hash"):
            require(valid_hash(row[field]), f"invalid {field}")
        for field, bits in (("layout_seed", 32), ("noise_seed", 64)):
            require(type(row[field]) is int and 0 <= row[field] < 2**bits, f"invalid {field}")
        for mode in MODES:
            require(type(row[mode]) is int and row[mode] in (0, 1), f"missing or nonbinary {mode} endpoint")
    for field in ("scene_hash", "movable_xy_hash", "layout_seed", "noise_seed"):
        require(len({row[field] for row in rows}) == n, f"duplicate {field}; stop and audit")


def analyze(manifest_files, seals, budget_stop=False):
    manifests = {digest(path): check_manifest(path) for path in manifest_files}
    prefixes, receipts, previous = [], [], []
    registered = []
    for j, (file, expected_sha) in enumerate(seals):
        path = Path(file)
        require(valid_hash(expected_sha) and digest(path) == expected_sha, "independent seal SHA mismatch")
        seal = load(path)
        n = 64 * 2**j
        require(seal["n"] == n and seal["manifest_sha256"] in manifests, "schedule or manifest mismatch")
        panel = manifests[seal["manifest_sha256"]]["cases"]
        require(len(panel) >= n and panel[:len(registered)] == registered, "panel prefix changed or too short")
        registered = panel
        rows = seal["rows"]
        validate_rows(rows, n)
        require(rows[:len(previous)] == previous, "previously sealed prefix changed")
        readback = path.parent / seal["native_readback"]
        require(valid_hash(seal["native_readback_sha256"]) and digest(readback) == seal["native_readback_sha256"],
                "native readback receipt changed")
        for i, row in enumerate(rows):
            for key in ("noise_seed", "layout_seed", "movable_xy_hash", "scene_hash"):
                require(row[key] == panel[i][key], f"{key} differs from sealed physical enrollment")
        prefixes.append(rows)
        receipts.append({"n": n, "file": path.name, "seal_sha256": expected_sha,
                         "manifest_sha256": seal["manifest_sha256"], "native_readback_sha256": seal["native_readback_sha256"]})
        previous = rows
    report = decide(prefixes)
    if budget_stop and report["decision"].startswith("continue_"):
        report["decision"] = report["decision"].replace("continue_", "inconclusive_")
    return {"receipts": receipts, **report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--budget-stop", action="store_true", help="Parent closed allocation at or after last complete look.")
    parser.add_argument("--seal", nargs=2, action="append", metavar=("FILE", "SHA256"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.manifest, args.seal, args.budget_stop)
    with args.out.open("x") as out:
        json.dump(report, out, indent=2, allow_nan=False)
        out.write("\n")
    print(report["decision"], report["crossings"])


if __name__ == "__main__":
    main()
