"""Transfer bowl-only completion centroids to the previously studied bottle panel."""
import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import sys

ENCODERS = Path(__file__).resolve().parents[1] / "libero-witness-encoders-20261119-r1"
sys.path.insert(0, str(ENCODERS))
import probe  # Pins CPU environment before importing torch or numpy.

import numpy as np

SOURCE = tuple(range(12, 16))
TARGET = tuple(range(24, 30))
GOALS = (0, 1)
VARIANTS = ("BASE", "D18")


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def read_tape(directory, key, arm, model, batch):
    pose, goal, variant = key
    path = directory / f"p{pose}-g{goal}-{variant}.npz"
    with np.load(path, allow_pickle=False) as tape:
        hits = tape["hits"]
        if hits.shape != (300, 2) or not np.isin(hits, (0, 1)).all():
            raise ValueError(f"Invalid predicates in {path}")
        ever = hits.any(axis=0).tolist()
        if ever[1-goal]:
            raise ValueError(f"Off-target predicate in {path}")
        issued, initial, frames = tape["issued"], tape["initial_state"], tape["frames"]
        if (issued.shape != (300, 7) or initial.shape != (9,)
                or not np.isfinite(issued).all() or not np.isfinite(initial).all()
                or frames.shape != (13, 2, 96, 96, 3) or frames.dtype != np.uint8):
            raise ValueError(f"Invalid public tape in {path}")
        row = dict(pose=pose, goal=goal, variant=variant, ever=ever,
                   final=hits[-1].astype(bool).tolist(), own_hit=ever[goal],
                   physics=str(tape["physics"].item()), start_state=digest(initial),
                   start_rgb=digest(frames[0]), commands=digest(issued))
        vector = None
        if variant != "QUERY":
            if arm == "motion":
                states = tape["states"]
                if states.shape != (300, 9) or not np.isfinite(states).all():
                    raise ValueError(f"Invalid motion in {path}")
                vector = states.reshape(12, 25, 9).mean(axis=1, dtype=np.float64).ravel()
            else:
                vector = probe.features(frames, model, batch)
    return row, vector


def audit_starts(rows):
    starts = {}
    for p in TARGET:
        pose_rows = [r for (pose, _, _), r in rows.items() if pose == p]
        identities = {(r["physics"], r["start_state"], r["start_rgb"]) for r in pose_rows}
        if len(identities) != 1:
            raise ValueError(f"Bottle branches disagree on start at pose {p}")
        for g in GOALS:
            base, drop, query = (rows[p, g, v]["commands"] for v in (*VARIANTS, "QUERY"))
            if base != drop or base == query:
                raise ValueError(f"Expected paired practice and separate query commands at {(p, g)}")
        starts[p] = pose_rows[0]["physics"]
    if len(set(starts.values())) != len(TARGET):
        raise ValueError("Bottle reset fingerprints repeat across pose IDs")
    return starts


def pooled_predictions(source, source_rows, target, arm):
    vectors = np.stack(list(source.values()))
    labels = np.array([source_rows[key]["own_hit"] for key in source], dtype=bool)
    if not labels.any() or labels.all():
        raise ValueError("Pooled source fit needs hits and misses")
    scale = 1.0
    if arm == "motion":
        channel_sd = vectors.reshape(-1, 9).std(axis=0)
        scale = np.tile(np.where(channel_sd > 1e-12, channel_sd, 1.0), 12)
    miss, hit = [(vectors[labels == label] / scale).mean(axis=0) for label in (False, True)]
    result = {}
    for key, vector in target.items():
        distances = [np.linalg.norm(vector / scale - center) for center in (miss, hit)]
        total = sum(distances)
        result[key] = float(distances[0] / total) if total else 0.5
    return result


def tally(values):
    values = list(values)
    return dict(numerator=sum(values), denominator=len(values), ties=values.count(0.5))


def fit_counts(rows):
    hits = sum(r["own_hit"] for r in rows)
    return dict(hits=hits, misses=len(rows)-hits, denominator=len(rows))


def evaluate(rows, scores):
    practices = []
    for key, score in scores.items():
        row = rows[key]
        hit = probe.preference(0.5, score)
        practices.append(dict(**row, score=score, correct=hit if row["own_hit"] else 1-hit))

    groups, cases = [], []
    for i, a in enumerate(TARGET):
        b, q = TARGET[(i+1) % len(TARGET)], TARGET[(i+2) % len(TARGET)]
        order_eligible = []
        for first in GOALS:
            order = (first, 1-first)
            practices_in_order = ((a, first), (b, 1-first))
            eligible = all(rows[p, g, "BASE"]["own_hit"] and not rows[p, g, "D18"]["own_hit"]
                           for p, g in practices_in_order) and all(rows[q, g, "QUERY"]["own_hit"] for g in GOALS)
            order_eligible.append(eligible)
            for va, vb, hidden in product(VARIANTS, VARIANTS, GOALS):
                variants = (va, vb)
                keys = [(p, g, v) for (p, g), v in zip(practices_in_order, variants)]
                bits = [int(rows[k]["ever"][hidden]) for k in keys]
                pair = [scores[k] for k in keys]
                query_hits = [int(rows[q, g, "QUERY"]["ever"][hidden]) for g in order]
                prob = float(bits[0]) if any(bits) else probe.preference(*pair)
                native = prob * query_hits[0] + (1-prob) * query_hits[1]
                strict = eligible and va != vb and variants[order.index(hidden)] == "D18"
                case = dict(triple=[a, b, q], prompt_order=order, variants=variants, hidden=hidden,
                            scores=pair, rewards=bits, query_hits=query_hits, prob_first=prob,
                            native=native, strict=strict)
                if strict:
                    if any(bits):
                        raise ValueError("A strict practice unexpectedly received positive reward")
                    swapped = [scores[p, g, v] for (p, g), v in zip(practices_in_order, reversed(variants))]
                    sw_prob = probe.preference(*swapped)
                    case.update(swapped_scores=swapped, swapped_prob_first=sw_prob,
                                swapped_native=sw_prob*query_hits[0] + (1-sw_prob)*query_hits[1])
                cases.append(case)
        groups.append(dict(triple=[a, b, q], order_eligible=order_eligible,
                           strict_cases=2*sum(order_eligible)))
    strict = [r for r in cases if r["strict"]]
    if len(cases) != 96 or len(strict) != sum(g["strict_cases"] for g in groups):
        raise ValueError("Assignment count mismatch")
    completion = tally(r["correct"] for r in practices)
    completion["by_class"] = {name: tally(r["correct"] for r in practices if r["own_hit"] == label)
                              for name, label in (("hit", True), ("miss", False))}
    completion["by_goal_variant"] = {f"g{g}-{v}": tally(r["correct"] for r in practices
                                      if r["goal"] == g and r["variant"] == v)
                                      for g, v in product(GOALS, VARIANTS)}
    all_native = tally(r["native"] for r in cases)
    all_native.update(decision_ties=sum(r["prob_first"] == 0.5 for r in cases),
                      reachable=sum(any(r["query_hits"]) for r in cases),
                      two_zero=sum(not any(r["rewards"]) for r in cases))
    return dict(physical_completion=completion, physical_practices=practices,
                query_tapes=[rows[p, g, "QUERY"] for p, g in product(TARGET, GOALS)],
                physical_denominators=dict(bowl_fit=16, bottle_practices=len(practices),
                                           bottle_query_branches=12, bottle_resets=len(TARGET)),
                strict=dict(candidate_triples=len(groups), candidate_orders=2*len(groups),
                            candidate_cases=4*len(groups), eligible_cases=len(strict),
                            native=tally(r["native"] for r in strict),
                            swapped=tally(r["swapped_native"] for r in strict)),
                cyclic_all_assignments=all_native, groups=groups, cases=cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("vlm", "robot", "rgb", "motion"), required=True)
    parser.add_argument("--source", type=Path, default=probe.WITNESS / "cpu02")
    parser.add_argument("--target", type=Path, default=probe.ROOT / "outputs/libero-witness-transfer-20261119-r1/cpu01")
    parser.add_argument("--pt-weights", type=Path, default=probe.STAGED_PT / "vision.safetensors")
    parser.add_argument("--robot-weights", type=Path, default=probe.ROBOT / "model.safetensors")
    parser.add_argument("--batch-size", type=int, choices=range(1, 11), default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    probe.torch.set_num_threads(4)
    probe.torch.set_num_interop_threads(1)
    probe.torch.set_float32_matmul_precision("highest")
    model, checkpoint = None, None
    if args.arm in ("vlm", "robot"):
        model, checkpoint = probe.load_encoder(*probe.checkpoint(args))

    source, source_rows, target, rows = {}, {}, {}, {}
    for key in product(SOURCE, GOALS, VARIANTS):
        source_rows[key], source[key] = read_tape(args.source, key, args.arm, model, args.batch_size)
    print(args.arm, "finished 16 bowl tapes", flush=True)
    for key in product(TARGET, GOALS, (*VARIANTS, "QUERY")):
        rows[key], vector = read_tape(args.target, key, args.arm, model, args.batch_size)
        if vector is not None:
            target[key] = vector
        if len(rows) % 6 == 0:
            print(args.arm, len(rows), "of 36 bottle tapes", flush=True)
    starts = audit_starts(rows)
    scores = pooled_predictions(source, source_rows, target, args.arm)
    result = evaluate(rows, scores)
    result.update(arm=args.arm, checkpoint=checkpoint, device="cpu", batch_size=args.batch_size,
                  threads=4, interop_threads=1, controls=probe.CONTROLS,
                  motion="12 consecutive 25-control means of nine measured wrist and jaw channels, scaled by source channel SD",
                  feature_dim=next(iter(source.values())).size,
                  source=str(args.source), target=str(args.target), goals=["top cabinet", "rack"],
                  scope="Bowl-only pooled fit on poses 12–15. Bottle outcomes were previously studied.",
                  fit_counts=dict(pooled=fit_counts(list(source_rows.values())),
                                  by_goal={g: fit_counts([r for r in source_rows.values() if r["goal"] == g])
                                           for g in GOALS},
                                  by_goal_variant={f"g{g}-{v}": fit_counts([r for r in source_rows.values()
                                                   if r["goal"] == g and r["variant"] == v])
                                                   for g, v in product(GOALS, VARIANTS)}),
                  source_fit_tapes=list(source_rows.values()), independent_start_fingerprints=starts)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
