"""Score paired two-zero histories against recorded native BASE query branches."""
import argparse
import hashlib
from collections import Counter
from itertools import permutations, product
import json
from pathlib import Path

import numpy as np


SPLITS = {"fit": (12, 13, 14, 15), "development": (16, 17, 18, 19)}
VARIANTS = ("BASE", "D18")
METHODS = ("rgb", "proprio", "commands")
GOALS = (0, 1)


def bitwise_equal(left, right):
    return (left.shape == right.shape and left.dtype.str == right.dtype.str
            and left.tobytes() == right.tobytes())


def rgb_features(frames):
    # Average every pixel, including edge pixels for non-divisible image sizes.
    pooled_rows = []
    for row in np.array_split(frames, min(8, frames.shape[2]), axis=2):
        row_mean = row.mean(axis=2, dtype=np.float64)
        cells = np.array_split(row_mean, min(8, frames.shape[3]), axis=2)
        pooled_rows.append(np.stack([cell.mean(axis=2) for cell in cells], axis=2))
    return np.diff(np.stack(pooled_rows, axis=2), axis=0).ravel()


def load_tape(file):
    with np.load(file, allow_pickle=False) as tape:
        frames, states, issued, hits = (tape[k] for k in ("frames", "states", "issued", "hits"))
        if (frames.ndim != 5 or frames.shape[:2] != (13, 2) or frames.shape[-1] != 3
                or min(frames.shape[2:4]) < 1 or frames.dtype != np.uint8):
            raise ValueError(f"{file}: expected uint8 frames [13,2,H,W,3], got {frames.shape}")
        for arr, shape in ((states, (300, 9)), (issued, (300, 7)), (hits, (300, 2))):
            if arr.shape != shape or not np.isfinite(arr).all():
                raise ValueError(f"{file}: expected finite values with shape {shape}")
        if not np.isin(hits, [0, 1]).all():
            raise ValueError(f"{file}: native predicates must be binary")
        # Fit with the same cadence for both trajectory readers.
        features = {"rgb": rgb_features(frames),
                    "proprio": states.reshape(12, 25, 9).mean(axis=1).ravel(),
                    "commands": issued.reshape(12, 25, 7).mean(axis=1).ravel()}
        initial = tape["initial_state"] if "initial_state" in tape else None
        if initial is not None and (initial.shape != (9,) or not np.isfinite(initial).all()):
            raise ValueError(f"{file}: invalid initial robot state")
        return dict(features=features, issued=issued, frame0=frames[0].copy(),
                    initial_state=initial, physics=tape["physics"] if "physics" in tape else None,
                    ever=hits.any(axis=0).tolist(), final=hits[-1].astype(bool).tolist(),
                    first_hit=[int(np.flatnonzero(col)[0]) + 1 if col.any() else None
                               for col in hits.T])


def fit_and_predict(tapes):
    fits = {}
    scores = {key: {} for key in tapes}
    for goal, method in product(GOALS, METHODS):
        keys = [(p, goal, v) for p, v in product(SPLITS["fit"], VARIANTS)]
        x = np.stack([tapes[k]["features"][method] for k in keys])
        y = np.array([tapes[k]["ever"][goal] for k in keys])
        # Per physical state/command channel across fitting attempts and time.
        # This keeps constant trajectory coordinates from receiving huge weights.
        scale = 1.0
        if method != "rgb":
            width = 9 if method == "proprio" else 7
            sd = x.reshape(-1, width).std(axis=0)
            scale = np.tile(np.where(sd > 1e-12, sd, 1.0), 12)
        available = bool(y.any() and (~y).any())
        fits.setdefault(goal, {})[method] = dict(misses=int((~y).sum()), hits=int(y.sum()),
                                                 available=available)
        if available:
            miss, hit = [(x[y == label] / scale).mean(axis=0) for label in (False, True)]
        for k, tape in tapes.items():
            if k[1] != goal:
                continue
            if not available:
                scores[k][method] = None
                continue
            z = tape["features"][method] / scale
            d0, d1 = (float(np.linalg.norm(z - center)) for center in (miss, hit))
            scores[k][method] = d0 / (d0 + d1) if d0 + d1 else 0.5
    return fits, scores


def preference(left, right):
    """Expected credit for selecting the smaller value, with fair unresolved choices."""
    if left is None or right is None or np.isclose(left, right, rtol=0, atol=1e-12):
        return 0.5
    return float(left < right)


def physical_summary(rows):
    counts = Counter(tuple(r["ever"]) for r in rows)
    result = dict(n=len(rows), own_hits=sum(r["own_hit"] for r in rows),
                  incidental_hits=sum(r["ever"][1-r["goal"]] for r in rows),
                  masks={"".join(str(int(v)) for v in mask): counts[mask]
                         for mask in product((False, True), repeat=2)})
    d18 = [r["event_controls"] for r in rows if r["variant"] == "D18"]
    result["D18_events"] = dict(recorded=sum(bool(v) for v in d18),
                                no_event=d18.count([]), unknown=d18.count(None))
    result["readers"] = {}
    for method in METHODS:
        credit = [r["classification_credit"][method] for r in rows]
        by_class = {}
        for label, name in ((False, "miss"), (True, "hit")):
            selected = [c for c, r in zip(credit, rows) if r["own_hit"] == label]
            by_class[name] = dict(correct=sum(selected), n=len(selected))
        result["readers"][method] = dict(correct=sum(credit), ties=credit.count(0.5),
                                         chance=len(rows)/2, by_class=by_class)
    return result


def physical_rows(poses, tapes, scores):
    rows = []
    for key in product(poses, GOALS, VARIANTS):
        p, g, variant = key
        tape = tapes[key]
        own = tape["ever"][g]
        pred = scores[key]
        credit = {}
        for m in METHODS:
            hit_choice = preference(0.5, pred[m])
            credit[m] = hit_choice if own else 1 - hit_choice
        rows.append(dict(pose=p, goal=g, variant=variant, ever=tape["ever"], final=tape["final"],
                         first_hit_control=tape["first_hit"], own_hit=own, hit_score=pred,
                         classification_credit=credit))
    return rows


def group_result(triple, tapes, scores, checks):
    a, b, q = triple
    cases = []
    masks = {}
    for p, g, role in ((a, 0, "A"), (b, 1, "B")):
        for v in VARIANTS:
            masks[role + "_" + v] = tapes[p, g, v]["ever"]
    for g in GOALS:
        masks[f"query_{g}"] = tapes[q, g, "BASE"]["ever"]
    required = {"A_BASE": [True, False], "A_D18": [False, False],
                "B_BASE": [False, True], "B_D18": [False, False],
                "query_0": [True, False], "query_1": [False, True]}
    criteria = {name: masks[name] == target for name, target in required.items()}
    criteria.update(practice_A_pair=checks[a]["pair"][0]["matched"],
                    practice_B_pair=checks[b]["pair"][1]["matched"],
                    shared_query_start=checks[q]["shared_query_start"])
    for hidden, (av, bv) in enumerate((("D18", "BASE"), ("BASE", "D18"))):
        a_key, b_key = (a, 0, av), (b, 1, bv)
        rewards = [int(tapes[k]["ever"][hidden]) for k in (a_key, b_key)]
        projected = {}
        for method in METHODS:
            observations = {}
            for name, (ka, kb) in (("own", (a_key, b_key)),
                                   ("swapped", ((a, 0, bv), (b, 1, av)))):
                sa, sb = scores[ka][method], scores[kb][method]
                pa = preference(sa, sb)
                branch_hits = [tapes[q, g, "BASE"]["ever"][hidden] for g in GOALS]
                branch_final = [tapes[q, g, "BASE"]["final"][hidden] for g in GOALS]
                observations[name] = dict(sA=sa, sB=sb, prob_A=pa,
                                          destination=pa if hidden == 0 else 1-pa,
                                          native=pa*branch_hits[0] + (1-pa)*branch_hits[1],
                                          final=pa*branch_final[0] + (1-pa)*branch_final[1])
            projected[method] = observations
        cases.append(dict(hidden=hidden, variants=[av, bv], native_practice_rewards=rewards,
                          two_zeros=not any(rewards), projections=projected))
    strict = all(criteria.values())
    cmd = [case["projections"]["commands"]["own"] for case in cases]
    cmd_chance = cmd[0]["prob_A"] == cmd[1]["prob_A"] and sum(x["destination"] for x in cmd) == 1
    if strict and not cmd_chance:
        raise ValueError(f"Commands-only control failed on strict group {triple}")
    return dict(A=a, B=b, query=q, native_masks=masks, criteria=criteria, strict=strict,
                commands_balanced_at_chance=cmd_chance, cases=cases)


def panel_summary(groups):
    strict = [g for g in groups if g["strict"]]
    outcomes = [c for g in strict for c in g["cases"]]
    readers = {}
    for m in METHODS:
        reader = {}
        for side in ("own", "swapped"):
            predictions = [c["projections"][m][side] for c in outcomes]
            correct = sum(x["destination"] for x in predictions)
            reader[side] = dict(expected_correct=correct, projected_native=sum(x["native"] for x in predictions),
                                projected_final=sum(x["final"] for x in predictions),
                                tied=sum(x["prob_A"] == 0.5 for x in predictions),
                                accuracy=correct / len(outcomes) if outcomes else None,
                                excess_over_chance=correct - len(outcomes)/2)
        pair_credit = [sum(c["projections"][m]["own"]["destination"] for c in g["cases"])
                       for g in strict]
        reader["groups_above_at_below_chance"] = [sum(x > 1 for x in pair_credit),
                                                   pair_credit.count(1), sum(x < 1 for x in pair_credit)]
        reader["own_minus_swapped_correct"] = reader["own"]["expected_correct"] - reader["swapped"]["expected_correct"]
        readers[m] = reader
    zero_cases = [c for g in groups for c in g["cases"] if c["two_zeros"]]
    before_filter = dict(assignments=2*len(groups), factual_two_zero_assignments=len(zero_cases),
                         projected_native_on_two_zeros={m: {s: sum(c["projections"][m][s]["native"]
                                                                  for c in zero_cases)
                                                            for s in ("own", "swapped")} for m in METHODS})
    reuse = Counter()
    for g in strict:
        for pose, goal, variants in ((g["A"], 0, VARIANTS), (g["B"], 1, VARIANTS),
                                     (g["query"], 0, ("BASE",)), (g["query"], 1, ("BASE",))):
            for v in variants:
                reuse[f"p{pose}-g{goal}-{v}"] += 1
    return dict(candidate_groups=len(groups), before_filter=before_filter, strict_groups=len(strict),
                assignments=len(outcomes),
                chance_expected_correct=len(outcomes)/2, readers=readers,
                unique_strict_tapes=len(reuse), strict_groups_using_each_tape=dict(reuse),
                unique_strict_practice_pairs=len({(g["A"], g["B"]) for g in strict}),
                unique_strict_query_poses=len({g["query"] for g in strict}))


def read_records(directory):
    path = directory / "branches.jsonl"
    if not path.exists():
        return {}
    records = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            key = row["pose"], row["goal"], row["variant"]
            if key in records:
                raise ValueError(f"Duplicate {key} in {path}")
            records[key] = row
    return records


def same_start(left, right):
    evidence = {}
    for field in ("initial_state", "physics"):
        if left[field] is not None and right[field] is not None:
            evidence[field] = bitwise_equal(left[field], right[field])
    for field in ("start_state", "physics"):
        records = [t["record"].get(field) for t in (left, right)]
        if all(x is not None for x in records):
            evidence["record_" + field] = records[0] == records[1]
    rgb = bitwise_equal(left["frame0"], right["frame0"])
    return dict(frame0=rgb, state_evidence=evidence, matched=rgb and bool(evidence) and all(evidence.values()))


def audit_starts(poses, tapes):
    checks = {}
    for p in poses:
        pairs = {}
        for g in GOALS:
            base, drop = [tapes[p, g, v] for v in VARIANTS]
            pairs[g] = same_start(base, drop)
            pairs[g]["issued"] = bitwise_equal(base["issued"], drop["issued"])
            pairs[g]["matched"] = pairs[g]["matched"] and pairs[g]["issued"]
        query = same_start(tapes[p, 0, "BASE"], tapes[p, 1, "BASE"])
        checks[p] = dict(pair=pairs, query=query, shared_query_start=query["matched"])
    return checks


def attach_record(tape, record, key):
    tape["record"] = record
    for field, data in (("actions", tape["issued"]), ("start_rgb", tape["frame0"]),
                        ("start_state", tape["initial_state"])):
        if data is not None and field in record:
            if hashlib.sha256(data.tobytes()).hexdigest() != record[field]:
                raise ValueError(f"{key}: branch record disagrees with {field}")
    if tape["physics"] is not None and "physics" in record:
        if tape["physics"].item() != record["physics"]:
            raise ValueError(f"{key}: branch record disagrees with physics")
    for field, data in (("any_goal", tape["ever"]), ("final", tape["final"])):
        if field in record and record[field] != data:
            raise ValueError(f"{key}: branch record disagrees with {field}")


def summarize_split(poses, tapes, scores):
    checks = audit_starts(poses, tapes)
    rows = physical_rows(poses, tapes, scores)
    for row in rows:
        key = row["pose"], row["goal"], row["variant"]
        record = tapes[key]["record"]
        row["event_controls"] = ([e["control"] for e in record["events"]]
                                 if "events" in record else None)
    all_groups = {t: group_result(t, tapes, scores, checks) for t in permutations(poses, 3)}
    cyclic = [(p, poses[(i+1) % 4], poses[(i+2) % 4]) for i, p in enumerate(poses)]
    panels = {}
    for name, triples in (("cyclic", cyclic), ("all_ordered", list(all_groups))):
        selected = [all_groups[t] for t in triples]
        panels[name] = panel_summary(selected)
        panels[name]["triples"] = [list(t) for t in triples]
        panels[name]["by_query_pose"] = {
            p: panel_summary([g for g in selected if g["query"] == p]) for p in poses
        }
    return dict(poses=poses, start_checks=checks, physical=physical_summary(rows),
                physical_by_pose={p: physical_summary([r for r in rows if r["pose"] == p]) for p in poses},
                physical_by_goal_variant={f"g{g}-{v}": physical_summary(
                    [r for r in rows if r["goal"] == g and r["variant"] == v])
                    for g, v in product(GOALS, VARIANTS)},
                physical_tapes=rows, panels=panels, groups=list(all_groups.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing the 32 NPZ tapes")
    parser.add_argument("destination_json", type=Path)
    args = parser.parse_args()
    directory = args.directory
    if not (directory / "p12-g0-BASE.npz").exists() and (directory / "out").is_dir():
        directory = directory / "out"
    records = read_records(directory)
    tapes = {}
    for key in product(range(12, 20), GOALS, VARIANTS):
        p, g, v = key
        tapes[key] = load_tape(directory / f"p{p}-g{g}-{v}.npz")
        attach_record(tapes[key], records.get(key, {}), key)
    fits, scores = fit_and_predict(tapes)
    report = dict(
        goals=["cabinet", "plate"], hit_window_controls=[1, 300],
        methods=dict(rgb="Successive differences of spatial mean RGB in up to 8x8 cells per camera.",
                     proprio="12 consecutive 25-control means of all nine measured state channels.",
                     commands="12 consecutive 25-control means of all seven issued command channels.",
                     fit="Per goal, pool both variants on poses 12-15. Scale state and command channels "
                         "by their fit standard deviation. Compute separate hit and miss centroids.",
                     score="Euclidean distance to miss divided by summed distances to miss and hit. "
                           "After two zeros choose the goal with the lower score. This score is uncalibrated.",
                     ties="Equal scores within 1e-12 or an unavailable class in either reader give "
                          "equal destination probabilities. A missing completion score is null.",
                     swap="Replace both practice trajectories with their opposite-variant executions "
                          "at the same starts. Preserve hidden goal, factual rewards and query branches."),
        counting="Each strict triple contributes two opposite hidden goals. Native projections weight "
                 "the two recorded BASE branches by choice probability. Before filtering, interpret "
                 "the two-zero rule only where two_zeros is true. All-order sensitivity includes the "
                 "cyclic triples and reuses the same 16 physical tapes per split. Counts are descriptive.",
        training_counts=fits,
        splits={name: summarize_split(poses, tapes, scores) for name, poses in SPLITS.items()})
    output = json.dumps(report, allow_nan=False, indent=2)
    args.destination_json.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
