"""Independent CPU audit of one closed Meta-World noise-screen condition."""
import argparse
from collections import defaultdict
import datetime
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import statistics

import numpy as np


def average(values):
    return statistics.fmean(values)


def ranks(values):
    return [sum(other < value for other in values) + (values.count(value) - 1) / 2
            for value in values]


def correlation(a, b):
    x, y = ranks(a), ranks(b)
    x = [v - average(x) for v in x]
    y = [v - average(y) for v in y]
    denominator = math.sqrt(sum(v*v for v in x) * sum(v*v for v in y))
    return sum(v*w for v, w in zip(x, y)) / denominator if denominator else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = args.run / "s0" / args.condition
    contract = json.loads((args.run / "s0/contract.json").read_text())
    reported = json.loads((directory / "gates.json").read_text())
    rows = [json.loads(line) for line in (directory / "outcomes.jsonl").read_text().splitlines()]
    assert len(rows) == 544
    assert [r["row"] for r in rows] == list(range(544))
    assert {r["task"] for r in rows} == {args.condition.rsplit("-", 1)[0]}
    assert {r["goal_visible"] for r in rows} == {args.condition.endswith("-visible")}
    identity = lambda r: (r["variant"], r["scene_index"], r["arm"], r["repeat"])
    expected = set(itertools.product(range(4), range(4), range(-1, 16), range(2)))
    assert set(map(identity, rows)) == expected
    assert all(type(r["success_once"]) is bool and 1 <= r["chunks"] <= 32
               and re.fullmatch("[0-9a-f]{64}", r["issued_actions_sha256"]) for r in rows)
    indexed = {identity(r): r for r in rows}
    rate = {(g, s, a): average([float(indexed[g, s, a, r]["success_once"]) for r in range(2)])
            for g, s, a in itertools.product(range(4), range(4), range(-1, 16))}
    profile = lambda g, scenes, a: average([rate[g, s, a] for s in scenes])
    per_goal = []
    for g in range(4):
        others = [t for t in range(4) if t != g]
        static = max(range(16), key=lambda a: (average([profile(t, range(4), a) for t in others]), -a))
        order = np.random.default_rng([0, g]).permutation(range(4)).tolist()
        halves = order[:2], order[2:]
        g1, g2 = [], []
        for select, score in (halves, halves[::-1]):
            chosen = max(range(16), key=lambda a: (profile(g, select, a), -a))
            g1.append(profile(g, score, chosen) - profile(g, score, -1))
            g2.append(profile(g, score, chosen) - profile(g, score, static))
        g3 = correlation(*[[profile(g, half, a) for a in range(16)] for half in halves])
        references = [[min(.98, max(.02, profile(t, range(4), a))) for a in range(16)] for t in others]
        observations = [(a, rate[g, s, a] >= .5) for s in range(4) for a in range(16)]
        order = np.random.default_rng([0, 1, g]).permutation(range(64))
        weights, hits, counts, gain = [1.] * 3, [0] * 16, [0] * 16, 0.
        for i in order:
            arm, success = observations[i]
            mixture = sum(w * p[arm] for w, p in zip(weights, references)) / sum(weights)
            independent = (hits[arm] + 1) / (counts[arm] + 2)
            gain += math.log(mixture if success else 1 - mixture) - math.log(independent if success else 1 - independent)
            weights = [w * (p[arm] if success else 1 - p[arm]) for w, p in zip(weights, references)]
            hits[arm] += int(success)
            counts[arm] += 1
        per_goal.append(dict(goal=g, native=profile(g, range(4), -1),
                             best_fixed=max(profile(g, range(4), a) for a in range(16)),
                             g1=average(g1), g2=average(g2), g3=g3, g4=gain / 64))
    statistics_by_gate = {}
    for key, name in (("g1", "g1_headroom"), ("g2", "g2_heterogeneity"),
                      ("g3", "g3_consistency"), ("g4", "g4_structure")):
        values = [r[key] for r in per_goal if r[key] is not None]
        value = average(values) if values else None
        saved = reported["summary"][name]
        assert (value is None and math.isnan(saved["mean"])) or math.isclose(value, saved["mean"], abs_tol=1e-12)
        threshold = {"g1": .15, "g2": .10, "g3": .5, "g4": 0.}[key]
        assert saved["threshold"] == threshold
        passed = value is not None and (value > threshold if key == "g4" else value >= threshold)
        assert saved["passed"] == passed
        statistics_by_gate[name] = {"mean": value, "undefined_conditions": 4-len(values),
                                   "threshold": threshold, "passed": passed}
        for own, original in zip(per_goal, reported["per_condition"]):
            expected_value = own[key]
            assert (expected_value is None and math.isnan(original[name])) or math.isclose(expected_value, original[name], abs_tol=1e-12)
    coverage = []
    preferences = set()
    for g in range(4):
        coverage.append(dict(goal=g, best_arm_success=per_goal[g]["best_fixed"],
                             arms_reaching=sum(profile(g, range(4), a) >= .5 for a in range(16))))
    for a in range(16):
        preferred = max(range(4), key=lambda g: (profile(g, range(4), a), -g))
        if profile(preferred, range(4), a) >= .5:
            preferences.add(preferred)
    for own, saved in zip(coverage, reported["goal_coverage"]["per_goal"]):
        assert all(own[k] == saved[k] for k in own)
    coverage_passed = all(r["arms_reaching"] for r in coverage) and len(preferences) > 1
    assert reported["goal_coverage"]["passed"] == coverage_passed
    assert reported["goal_coverage"]["goals_preferred_by_some_arm"] == len(preferences)
    repeated = [[indexed[g, s, a, r] for r in range(2)]
                for g, s, a in itertools.product(range(4), range(4), range(16))]
    determinism = dict(repeated_groups=len(repeated),
                       outcome_matched=sum(a["success_once"] == b["success_once"] for a, b in repeated),
                       trajectory_matched=sum(a["issued_actions_sha256"] == b["issued_actions_sha256"] for a, b in repeated))
    assert all(reported["determinism"][k] == v for k, v in determinism.items())
    trajectory_file = directory / "trajectories/first-chunks.npz"
    with np.load(trajectory_file, allow_pickle=False) as data:
        assert np.array_equal(data["row"], np.arange(544))
        chunks = data["issued"]
        assert chunks.shape == (544, 1, 5, 4) and np.isfinite(chunks).all()
        pair_distances = [float(np.linalg.norm(chunks[a["row"]] - chunks[b["row"]])) for a, b in repeated]
        between, native = [], []
        for g, s in itertools.product(range(4), range(4)):
            first = [chunks[indexed[g, s, a, 0]["row"]] for a in range(16)]
            between.extend(float(np.linalg.norm(a-b)) for a, b in itertools.combinations(first, 2))
            native.append(float(np.linalg.norm(chunks[indexed[g, s, -1, 0]["row"]] - chunks[indexed[g, s, -1, 1]["row"]])))
        assert math.isclose(average(between), reported["mode_effect"]["first_chunk_distance_between_arms"], abs_tol=1e-7)
        assert math.isclose(average(native), reported["mode_effect"]["first_chunk_distance_native_repeats"], abs_tol=1e-7)
    files = [directory / "outcomes.jsonl", directory / "gates.json", trajectory_file, args.run / "s0/contract.json"]
    report = dict(checked_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                  condition=args.condition, expected_rows=544, audited_rows=len(rows),
                  successes=sum(r["success_once"] for r in rows),
                  native_successes=sum(r["success_once"] for r in rows if r["arm"] == -1), native_episodes=32,
                  per_goal=per_goal, gates=statistics_by_gate, determinism=determinism,
                  goal_coverage=dict(per_goal=coverage, preferred_goals=len(preferences), passed=coverage_passed),
                  fixed_repeat_first_chunk_distance=dict(mean=average(pair_distances), maximum=max(pair_distances)),
                  model_fingerprint_before=contract["model_fingerprint"], final_motor_fingerprint_pending=True,
                  files={str(f.relative_to(args.run)): dict(bytes=f.stat().st_size, sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
