"""Read native evidence and report paired UR5e outcomes on the registered 100 starts."""

import hashlib
import json
import re
from itertools import combinations
from math import comb
from pathlib import Path

ARMS = ("Panda", "Sawyer")
CONDITIONS = ("raw", "correct", "empty", "wrong", "calibration")
N = 100


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(data):
    return json.dumps(
        data, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def fingerprint(data):
    return hashlib.sha256(canonical(data)).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def sha_string(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def binomial_cdf(x, n, p):
    return sum(comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(x + 1))


def upper_bound(x, n, tail):
    """Exact one-sided Clopper-Pearson upper bound."""
    if x == n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(55):
        mid = (hi + lo) / 2
        if binomial_cdf(x, n, mid) > tail:
            lo = mid
        else:
            hi = mid
    return hi


def paired(a, b, *, alpha=0.05):
    pairs = list(zip(a, b, strict=True))
    n = len(pairs)
    g = sum(x and not y for x, y in pairs)
    l = sum(y and not x for x, y in pairs)
    d = g + l
    row = {
        "paired_n": n,
        "gains": g,
        "losses": l,
        "both_success": sum(x and y for x, y in pairs),
        "both_failure": sum(not (x or y) for x, y in pairs),
        "delta": None,
        "interval": None,
        "interval_confidence": 1 - alpha,
        "exact_mcnemar_p": None,
    }
    if n:
        # Simultaneous intervals on P(gain), P(loss); the difference is the paired risk difference.
        tail = alpha / 4
        lower = (1 - upper_bound(n - g, n, tail)) - upper_bound(l, n, tail)
        upper = upper_bound(g, n, tail) - (1 - upper_bound(n - l, n, tail))
        row.update(
            delta=(g - l) / n,
            interval=[max(-1.0, lower), min(1.0, upper)],
            exact_mcnemar_p=min(1.0, 2 * binomial_cdf(min(g, l), d, 0.5)),
        )
    return row


def native_outcome(path, rec, binding, slot, condition, admission, stride, context):
    """Recover success from hashed actions, proposals, sparse rewards and terminal flags."""
    import numpy as np

    require(sha(path) == rec["sha256"], f"Changed native evidence: {path}")
    with np.load(path, allow_pickle=False) as tape:
        row = json.loads(tape["receipt"].item())
        n = len(tape["rewards"])
        require(1 <= n <= 300, "Invalid episode length")
        require(
            row["binding"] == binding
            and row["slot"] == slot
            and row["condition"] == condition
            and row["body"] == "UR5e"
            and row["opening"]["admission"] == admission
            and row["stride"] == stride
            and row["context_sha256"] == context,
            f"Episode binding mismatch: {path}",
        )
        opening = row["opening"]
        actions, raw, motion = (tape[k] for k in ("actions", "proposals", "motion"))
        for v in (actions, raw, motion):
            require(
                v.shape == (n, 7) and np.isfinite(v).all(),
                "Invalid native motion/action array",
            )
        require(
            (abs(raw) <= 1).all() and (abs(actions) <= 1).all(), "Unbounded actions"
        )
        if condition == "raw":
            require(
                np.array_equal(raw, actions), "An assisted raw episode was supplied"
            )
        chunks = tape["chunks"]
        require(
            opening["first_noise_seed"] == 900000 + 1000 * admission["layout"]["seed"],
            "Different plan seed",
        )
        require(
            stride in (1, 2, 5)
            and chunks.shape == ((n + stride - 1) // stride, 5, 7)
            and np.isfinite(chunks).all()
            and (abs(chunks) <= 1).all()
            and opening["chunk"] == chunks[0].tolist()
            and np.array_equal(raw, chunks[:, :stride].reshape(-1, 7)[:n]),
            "Changed raw chunks",
        )
        rewards, done, success = (tape[k] for k in ("rewards", "done", "success"))
        require(
            all(v.shape == (n,) for v in (rewards, done, success))
            and done.dtype == success.dtype == np.bool_
            and np.isin(rewards, [0, 1]).all()
            and not rewards[:-1].any()
            and not done[:-1].any()
            and np.array_equal(rewards > 0, success)
            and done[-1]
            and (success[-1] or n == 300),
            "Incomplete or inconsistent native terminal evidence",
        )
        return bool(success[-1]), opening


def summarize(outcomes):
    require(len(outcomes) == N, "Need 100 registered layout slots")
    require(
        all(
            r is None
            or (
                set(r) == set(CONDITIONS)
                and all(v is None or type(v) is bool for v in r.values())
            )
            for r in outcomes
        ),
        "Need explicit native successes or missing values for all conditions",
    )
    observed = [r for r in outcomes if r is not None]
    n = len(observed)
    result = {
        "planned": N,
        "admitted": n,
        "physically_missing": N - n,
        "conditions": {},
        "comparisons": {},
    }
    for c in CONDITIONS:
        series = [r[c] for r in observed if r[c] is not None]
        score, count = sum(series), len(series)
        result["conditions"][c] = {
            "success": score,
            "observed": count,
            "planned": N,
            "missing": N - count,
            "observed_rate": score / count if count else None,
            "all_layout_rate_bounds": [score / N, (score + N - count) / N],
        }
    for a, b in combinations(CONDITIONS, 2):
        if b == "correct":
            a, b = b, a
        complete = [r for r in observed if r[a] is not None and r[b] is not None]
        contrast = paired([r[a] for r in complete], [r[b] for r in complete])
        # Also use known one-sided observations in partially observed cases.
        lower = upper = 0
        for row in outcomes:
            x, y = (row[a], row[b]) if row is not None else (None, None)
            lower += int(x or False) - int(True if y is None else y)
            upper += int(True if x is None else x) - int(y or False)
        contrast["all_layout_delta_bounds"] = [lower / N, upper / N]
        result["comparisons"][f"{a}_minus_{b}"] = contrast
    history = result["comparisons"]["correct_minus_empty"]
    ci = history["interval"]
    if ci is None:
        finding = "No complete correct-versus-empty pairs."
    elif ci[0] > 0:
        finding = "Correct prior history improved native success versus empty context on admitted layouts."
    elif ci[1] < 0:
        finding = "Correct prior history reduced native success versus empty context on admitted layouts."
    else:
        finding = (
            "The paired interval includes zero. A prior-history benefit is unresolved."
        )
    result["prior_history_answer"] = finding
    wrong_ci = result["comparisons"]["correct_minus_wrong"]["interval"]
    result["matched_body_history_answer"] = (
        "Correct practice outperformed command-matched wrong-arm practice."
        if wrong_ci and wrong_ci[0] > 0
        else "Wrong-arm practice outperformed correct practice."
        if wrong_ci and wrong_ci[1] < 0
        else "A body-specific history advantage is unresolved."
    )
    result["complete_five_way_pairs"] = sum(
        all(r[c] is not None for c in CONDITIONS) for r in observed
    )
    result["interval_method"] = (
        "Conservative 95% paired risk difference using simultaneous exact binomial bounds on gains and losses; marginal per contrast."
    )
    return result


def audited_shard(directory):
    report = read(directory / "report.json")
    require(
        report["status"] in ("complete", "shard_complete"),
        "Require a finished target shard",
    )
    shard = report.get("shard", {"count": 1, "index": 0, "slots": list(range(N))})
    count, index = shard["count"], shard["index"]
    require(
        type(index) is int
        and type(count) is int
        and 0 <= index < count <= N
        and shard["slots"] == list(range(index, N, count))
        and (report["status"] == "complete") == (count == 1),
        "Invalid shard assignment",
    )
    assigned = set(shard["slots"])
    plan = read(directory / "target-binding.json")
    require(
        report["binding"] == fingerprint(plan)
        and len(plan["layouts"]) == N
        and len(report["slots"]) == N,
        "Wrong target panel or plan hash",
    )
    for key in ("seed", "scene", "row", "xy"):
        require(
            len({r[key] for r in plan["layouts"]}) == N,
            f"Duplicate physical layout: {key}",
        )
    context = fingerprint(
        {"practice": report["practice"], "calibration": report["calibration"]}
    )
    require(
        report["gate_verified_before_target"] is True
        and report["context_sha256"] == context
        and sha(directory / "review.json") == report["receipt_sha256"]
        and read(directory / "review.json") == plan,
        "Unbound gate or practice bank",
    )
    for key, item in report["practice"].items():
        if item["status"] == "admitted":
            require(
                item["file"] == f"practice-{key}.npz"
                and sha(directory / item["file"]) == item["sha256"],
                "Changed motion practice",
            )
        else:
            require(
                item["status"] == "physically_unadmitted" and bool(item["reason"]),
                "Unknown practice status",
            )
    admissions = read(directory / "admission.json")
    require(
        sha(directory / "admission.json") == report["admission_sha256"]
        and len(admissions) == N,
        "Invalid target admission record",
    )
    cases = {}
    for i, (layout, adm, slot) in enumerate(
        zip(plan["layouts"], admissions, report["slots"], strict=True)
    ):
        require(adm["layout"] == layout, "Different scene/ordering at target admission")
        if i not in assigned:
            require(not slot, "Query outside assigned shard")
            continue
        if "shard" in report:
            require(
                read(directory / f"case-{i:03d}.json") == slot, "Changed case receipt"
            )
        if adm["status"] == "physically_unadmitted":
            require(
                not slot and bool(adm["reason"]),
                "Unadmitted layout has a query or lacks a reason",
            )
            cases[i] = (slot, None)
            continue
        require(
            adm["status"] == "admitted" and set(slot) == set(CONDITIONS),
            "Incomplete paired case",
        )
        reset = adm["reset"]
        phys = reset["environment_reproducibility"]
        require(
            reset["model_identity"]["arm"] == "UR5e"
            and reset["max_steps"] == 300
            and phys["generated_row_fingerprint"] == layout["row"]
            and phys["scene_fingerprint"] == layout["scene"]
            and phys["seed"] == layout["seed"]
            and all(
                sha_string(phys[k])
                for k in ("post_settle_scene_fingerprint", "initial_state_fingerprint")
            )
            and set(adm["observation"]) == {"rgb", "wrist", "state", "prompt"}
            and adm["observation"]["prompt"] == plan["prompt"]
            and all(
                sha_string(adm["observation"][k]) for k in ("rgb", "wrist", "state")
            ),
            "Unverified target reset or opening public observation",
        )
        p, donor = plan["supports"][i], plan["donors"][i]
        required_contexts = {
            "raw": [],
            "empty": [],
            "correct": [f"UR5e-{p}"],
            "wrong": [f"{donor}-{p}"],
            "calibration": [
                f"{b}-{j}"
                for b in ("Panda", "UR5e")
                for j in (*plan["calibration_poses"], plan["calibration_check"])
            ],
        }
        scores, reference = {}, None
        for condition in CONDITIONS:
            rec = slot[condition]
            if any(
                report["practice"][k]["status"] != "admitted"
                for k in required_contexts[condition]
            ):
                require(
                    rec == {"missing": "predeclared practice physically unadmitted"},
                    "Unexpected practice imputation",
                )
                scores[condition] = None
                continue
            name = f"query-{i:03d}-{condition}.npz"
            require(rec["file"] == name, "Incorrect tape assignment")
            scores[condition], opening = native_outcome(
                directory / name,
                rec,
                report["binding"],
                i,
                condition,
                adm,
                plan["stride"],
                context,
            )
            if reference is None:
                reference = opening
            require(
                reference == opening,
                "Different initial state, public observation or first raw proposal",
            )
        cases[i] = (slot, scores)
    identity = tuple(
        report[k]
        for k in ("binding", "receipt_sha256", "admission_sha256", "context_sha256")
    )
    return identity, plan, admissions, cases


def aggregate(directory, reuse=()):
    """Reaudit each finished run and require every registered case exactly once or identical."""
    if isinstance(reuse, (str, Path)):
        reuse = [reuse]
    identity, plan, admissions, cases = audited_shard(Path(directory))
    for extra in reuse or ():
        other, _, _, rows = audited_shard(Path(extra))
        require(
            other == identity,
            "Cannot combine different receipts, admissions or contexts",
        )
        for i, case in rows.items():
            require(
                i not in cases or cases[i] == case,
                f"Conflicting native receipts for case {i}",
            )
            cases[i] = case
    require(
        set(cases) == set(range(N)), f"Incomplete target coverage: {len(cases)}/{N}"
    )
    outcomes = [cases[i][1] for i in range(N)]
    missing = [
        {"slot": i, "layout": adm["layout"], "reason": adm["reason"]}
        for i, adm in enumerate(admissions)
        if outcomes[i] is None
    ]
    return {
        **summarize(outcomes),
        "binding": identity[0],
        "receipt_sha256": identity[1],
        "physically_unadmitted": missing,
        "per_layout": [
            {"slot": i, "layout": p, "native_success": r}
            for i, (p, r) in enumerate(zip(plan["layouts"], outcomes, strict=True))
        ],
    }
