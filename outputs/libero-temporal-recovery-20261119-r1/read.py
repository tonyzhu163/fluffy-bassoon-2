"""Audit the temporal-recovery panel and score pose-disjoint native-outcome readers."""
import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "libero-interrupted-practice-20261119-r1"))
from score import features as prior_features


POSES = tuple(range(24, 30))
GOALS = (0, 1)
FAULTS = ("D17", "D18")
MODES = ("default", "regrasp", "transport")
SENSORS = ("terminal_images", "terminal_proprio", "history_images", "history_proprio", "six_motion")
NEXT = dict(zip(POSES, POSES[1:] + POSES[:1]))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_rows(directory):
    expected = set(product(POSES, GOALS, FAULTS, ("support", *MODES)))
    rows = {}
    for number, line in enumerate((directory / "branches.jsonl").read_text().splitlines(), 1):
        r = json.loads(line)
        key = tuple(r[k] for k in ("pose", "goal", "fault", "mode"))
        require(type(r["pose"]) is int and type(r["goal"]) is int and key in expected,
                f"Unexpected branch at line {number}: {key}")
        require(key not in rows, f"Duplicate branch {key}")
        for field in ("hit", "final", "fired"):
            require(type(r[field]) is bool, f"{key}: invalid {field}")
        require(not r["final"] or r["hit"], f"{key}: terminal hit missing from ever-hit")
        for field in ("physics", "start_state", "start_rgb", "commands"):
            require(isinstance(r[field], str) and r[field].strip(), f"{key}: missing {field}")
        rows[key] = r
    require(rows.keys() == expected, f"Missing branches: {sorted(expected - rows.keys())}")
    for p, g in product(POSES, GOALS):
        group = [rows[p, g, f, m] for f, m in product(FAULTS, ("support", *MODES))]
        for field in ("physics", "start_state", "start_rgb"):
            require(len({r[field] for r in group}) == 1, f"Unmatched {field} at {(p, g)}")
        require(rows[p, g, "D17", "support"]["commands"] == rows[p, g, "D18", "support"]["commands"],
                f"Unmatched support command hashes at {(p, g)}")
    return rows


def load_support(directory, p, g, fault):
    path = directory / f"p{p}-g{g}-{fault}.npz"
    with np.load(path, allow_pickle=False) as tape:
        frames, states, commands = (tape[k] for k in ("frames", "states", "issued"))
    require(frames.shape == (13, 2, 96, 96, 3) and frames.dtype == np.uint8,
            f"{path}: invalid RGB")
    for name, a, shape in (("states", states, (300, 9)), ("issued", commands, (300, 7))):
        require(a.shape == shape and a.dtype.kind in "fiu" and np.isfinite(a).all(),
                f"{path}: invalid {name}")
    # Each camera is reduced to an 8 by 8 grid of spatial RGB means.
    images = frames.reshape(13, 2, 8, 12, 8, 12, 3).mean(axis=(3, 5))
    states = states.astype(np.float64)
    features = dict(terminal_images=images[-1].ravel(), terminal_proprio=states[-1],
                    history_images=images.ravel(), history_proprio=states.reshape(12, 25, 9).mean(1).ravel())
    features["six_motion"] = prior_features(commands, states)["history"]
    return frames, states, commands, features


def endpoints(left, right):
    frame_delta = np.abs(left[0][-1].astype(np.int16) - right[0][-1])
    camera_equal = ~(frame_delta != 0).any(axis=(1, 2, 3))
    delta = np.abs(left[1][-1] - right[1][-1])
    equal = dict(agent_rgb=bool(camera_equal[0]), wrist_rgb=bool(camera_equal[1]),
                 both_rgb=bool(camera_equal.all()), xyz=bool((delta[:3] == 0).all()),
                 quat=bool((delta[3:7] == 0).all()), jaw=bool((delta[7:] == 0).all()),
                 state_9d=bool((delta == 0).all()), joint=bool(camera_equal.all() and (delta == 0).all()))
    return dict(exact=equal, rgb_mae=frame_delta.mean(axis=(1, 2, 3)).tolist(),
                abs_state_delta=delta.tolist(), max_abs_state_delta=float(delta.max()))


def build_blocks(directory, rows):
    blocks, sensors, outcomes = {}, {}, {}
    for p, g in product(POSES, GOALS):
        q = NEXT[p]
        tapes = [load_support(directory, p, g, f) for f in FAULTS]
        require(np.array_equal(tapes[0][2], tapes[1][2]), f"Support tapes differ in commands at {(p, g)}")
        require(np.array_equal(tapes[0][0][0], tapes[1][0][0]), f"Support initial RGB differs at {(p, g)}")
        support = [rows[p, g, f, "support"] for f in FAULTS]
        dual_zero = not any(r["hit"] for r in support)
        both_fired = all(r["fired"] for r in support)
        for f, tape in zip(FAULTS, tapes):
            sensors[p, g, f] = tape[3]
            outcomes[p, g, f] = np.array([rows[q, g, f, m]["hit"] for m in MODES], dtype=int)
        left, right = (outcomes[p, g, f] for f in FAULTS)
        reachable = bool(left.any() and right.any())
        no_common = not (left & right).any()
        blocks[p, g] = dict(support_pose=p, query_pose=q, goal=g, dual_zero=dual_zero,
                            both_support_fired=both_fired, both_reachable=reachable,
                            no_common_winner=no_common,
                            strict=dual_zero and both_fired and reachable and no_common,
                            endpoints=endpoints(*tapes))
    return blocks, sensors, outcomes


def summarize_blocks(blocks):
    all_blocks = list(blocks.values())
    result = {}
    for group in ("all", "dual_zero", "strict"):
        selected = [b for b in all_blocks if group == "all" or b[group]]
        ends = [b["endpoints"] for b in selected]
        exact = {k: sum(e["exact"][k] for e in ends) for k in all_blocks[0]["endpoints"]["exact"]}
        result[group] = dict(blocks=len(selected), zero_denominator=not selected,
                             both_support_fired=sum(b["both_support_fired"] for b in selected),
                             both_reachable=sum(b["both_reachable"] for b in selected),
                             no_common_winner=sum(b["no_common_winner"] for b in selected),
                             both_reachable_no_common=sum(b["no_common_winner"] and b["both_reachable"]
                                                         for b in selected),
                             endpoint_exact_counts=exact,
                             mean_rgb_mae=np.mean([e["rgb_mae"] for e in ends], axis=0).tolist() if ends else None,
                             max_abs_state_by_dim=np.max([e["abs_state_delta"] for e in ends], axis=0).tolist()
                             if ends else None,
                             max_abs_state_delta=max((e["max_abs_state_delta"] for e in ends), default=None))
    return result


def decision(feature, train_features, train_hits):
    distances = np.square(train_features - feature).mean(axis=1)
    closest = distances == distances.min()
    return int(train_hits[closest].sum(axis=0).argmax())


def evaluate(rows, blocks, sensors, outcomes):
    splits, cases = [], []
    arms = ["fixed", *SENSORS, *(s + "_swapped" for s in SENSORS)]
    scores = {a: {k: dict(native_sum=0, final_sum=0, cases=0)
                  for k in ("all", "dual_zero", "strict")} for a in arms}
    for p, q in NEXT.items():
        train = [t for t in POSES if {t, NEXT[t]}.isdisjoint({p, q})]
        splits.append(dict(held_out=[p, q], fit_pairs=[[t, NEXT[t]] for t in train]))
        for g in GOALS:
            train_keys = list(product(train, (g,), FAULTS))
            labels = np.stack([outcomes[k] for k in train_keys])
            fixed = int(labels.sum(axis=0).argmax())
            choices = {f: {"fixed": fixed} for f in FAULTS}
            for s in SENSORS:
                features = np.stack([sensors[k][s] for k in train_keys])
                scale = 1.0
                if s == "six_motion":
                    sd = features.std(axis=0)
                    scale = np.where(sd > 1e-12, sd, 1.0)
                for f in FAULTS:
                    choices[f][s] = decision(sensors[p, g, f][s] / scale, features / scale, labels)
                for f, other in zip(FAULTS, reversed(FAULTS)):
                    choices[f][s + "_swapped"] = choices[other][s]
            for f in FAULTS:
                branch = [rows[q, g, f, m] for m in MODES]
                support = rows[p, g, f, "support"]
                case = dict(support_pose=p, query_pose=q, goal=g, fault=f,
                            support={k: support[k] for k in ("hit", "final", "fired")},
                            query={k: [r[k] for r in branch] for k in ("hit", "final", "fired")},
                            choice={a: MODES[i] for a, i in choices[f].items()})
                cases.append(case)
                for a, i in choices[f].items():
                    for group, total in scores[a].items():
                        if group == "all" or blocks[p, g][group]:
                            total["native_sum"] += branch[i]["hit"]
                            total["final_sum"] += branch[i]["final"]
                            total["cases"] += 1
    for groups in scores.values():
        for total in groups.values():
            total["zero_denominator"] = total["cases"] == 0
    return splits, scores, cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = load_rows(args.directory)
    blocks, sensors, outcomes = build_blocks(args.directory, rows)
    splits, scores, cases = evaluate(rows, blocks, sensors, outcomes)
    native = {}
    for f, m in product(FAULTS, ("support", *MODES)):
        subset = [rows[p, g, f, m] for p, g in product(POSES, GOALS)]
        native.setdefault(f, {})[m] = {k: sum(r[k] for r in subset) for k in ("hit", "final", "fired")}
    report = dict(audit=dict(records=len(rows), complete_keys=True, support_tapes=len(sensors),
                             matched_physics_state_rgb_groups=len(blocks),
                             matched_support_rgb_and_commands_pairs=len(blocks)),
                  mode_order=MODES, camera_order=["agent", "wrist"], native_counts_by_fault_mode=native,
                  blocks=list(blocks.values()), counts=summarize_blocks(blocks), splits=splits,
                  decision_rule=("Within each public goal, fit on every disjoint fitting pair and both conditions. "
                                 "Fixed maximizes fitting native hits. Sensor 1-NN uses unscaled mean squared "
                                 "distance (six-motion channels scaled by fit SD), averages native ever-hit vectors over all exact nearest ties, "
                                 "and chooses the first maximal mode in mode_order. Each case spends one query. "
                                 "Swapped uses the opposite support at the same pose and goal. "
                                 "Images use 8x8 spatial means per camera for the last or all 13 frames. "
                                 "Proprio uses the last 9D state or means over 12 consecutive 25-control bins."),
                  reader=scores, cases=cases)
    args.output.write_text(json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
