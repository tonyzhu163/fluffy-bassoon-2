"""Score the complete Goal 2 command/grasp pilot from its closed-loop episodes."""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np


POSES = range(30, 36)
SIGNS = (1, -1)
CONDITIONS = ("BASE", "D17")
GRIPS = (0, 1)
CASES = tuple(product(POSES, SIGNS, CONDITIONS))
CHOICES = tuple(product(SIGNS, GRIPS))


def check_episode(root, row, key):
    pose, wire, condition, decode, grip = key
    name = f"p{pose}-{condition}-wire{wire:+d}-decode{decode:+d}-grip{grip}"
    folder = root / name
    if (row["episode"] != name or (row["suite"], row["task_id"]) != ("libero_goal", 2)
            or row["net_x_sign"] != wire * decode or row["horizon"] != 300):
        raise ValueError(f"{name}: episode identity, net wire or horizon mismatch")
    with np.load(folder / "public.npz") as tape, np.load(folder / "evaluator.npz") as evaluator:
        success, reward = tape["success"], tape["reward"]
        done = tape["terminated"] | tape["truncated"]
        n = row["controls"]
        if (not 1 <= n <= 300 or success.shape != reward.shape or success.shape != (n,)
                or done.shape != (n,) or not done[-1] or done[:-1].any()
                or bool(success.any()) != row["native_success"] or reward.sum() != row["native_return"]):
            raise ValueError(f"{name}: native outcome or termination differs from the control tape")
        command = np.clip(tape["selected"].astype(np.float32), -1, 1)
        command[:, 0] *= decode
        if not np.array_equal(command, tape["wire_action"]):
            raise ValueError(f"{name}: decoder does not match issued commands")
        command[:, 0] *= wire
        if not np.array_equal(command, evaluator["native_input"]):
            raise ValueError(f"{name}: wire does not match native simulator inputs")
        recovery = np.count_nonzero(tape["phase"])
        if recovery != row["regrasp_controls"] or (recovery and not grip):
            raise ValueError(f"{name}: unexpected recovery controls")
    diag = json.loads((folder / "diagnostics.json").read_text())
    if diag["case_id"] != condition:
        raise ValueError(f"{name}: wrong native diagnostic")
    if condition == "D17":
        events = diag["events"]
        opportunity = any(e["event"] == "grasp_failure_window_started" for e in events)
        fired = [int(e["step"]) + 1 for e in events if e["event"] == "recoverable_grasp_failure"]
        first = fired[0] if fired else None
        if (row["d17_opportunity"] is not opportunity or row["d17_triggered"] is not bool(fired)
                or row["d17_event_control"] != first):
            raise ValueError(f"{name}: D17 event summary differs from the diagnostic log")


def read_panel(path):
    marker = json.loads(path.with_name("complete.json").read_text())
    if (marker["mode"], marker["episodes"], marker["equivalent_pairs_checked"]) != ("pilot", 96, 48):
        raise ValueError("expected the full 96-episode completion record with 48 equivalent pairs")
    panel = {}
    for line, raw in enumerate(path.read_text().splitlines(), 1):
        row = json.loads(raw)
        key = tuple(row[name] for name in ("pose", "wire", "condition", "decode", "regrasp"))
        if any(type(row[k]) is not int for k in ("pose", "wire", "decode", "regrasp")):
            raise ValueError(f"line {line}: invalid pose, wire or correction type")
        if key in panel:
            raise ValueError(f"line {line}: duplicate branch {key}")
        if type(row["native_success"]) is not bool:
            raise ValueError(f"line {line}: expected boolean native_success")
        if row["seed"] != 19193000 + row["pose"]:
            raise ValueError(f"line {line}: seed differs from the registered pose seed")
        panel[key] = row
    expected = {case + choice for case, choice in product(CASES, CHOICES)}
    if panel.keys() != expected:
        raise ValueError(f"incomplete or unexpected panel: missing={expected - panel.keys()}, "
                         f"extra={panel.keys() - expected}")
    if len({r["initial_physics"] for r in panel.values()}) != len(POSES):
        raise ValueError("expected six distinct physical starts")
    for pose in POSES:
        reference = panel[pose, 1, "BASE", 1, 0]
        for wire, condition, decode, grip in product(SIGNS, CONDITIONS, SIGNS, GRIPS):
            row = panel[pose, wire, condition, decode, grip]
            for field in ("initial_physics", "initial_public", "first_proposal", "noise_bank"):
                if not row[field] or row[field] != reference[field]:
                    raise ValueError(f"pose {pose}: missing or unpaired {field}")
            check_episode(path.parent, row, (pose, wire, condition, decode, grip))
    return panel


def paired(true, other):
    wins = sum(a and not b for a, b in zip(true, other))
    losses = sum(b and not a for a, b in zip(true, other))
    return dict(cases=len(true), true_success=sum(true), comparator_success=sum(other),
                true_only=wins, comparator_only=losses, both=sum(true) - wins,
                neither=len(true) - sum(true) - losses, net=wins - losses)


def choice_name(decode, grip):
    return f"fixed_decode_{decode:+d}_grip_{grip}"


def score(panel):
    cells = []
    for wire, condition, decode, grip in product(SIGNS, CONDITIONS, SIGNS, GRIPS):
        rows = [panel[pose, wire, condition, decode, grip] for pose in POSES]
        cell = dict(wire=wire, condition=condition, decode=decode, regrasp=grip,
                    successes=sum(r["native_success"] for r in rows), episodes=len(rows),
                    regrasp_executed=sum(r["regrasp_controls"] > 0 for r in rows))
        if condition == "D17":
            for field in ("d17_opportunity", "d17_triggered"):
                cell[field] = dict(yes=sum(r[field] for r in rows), observed=len(rows))
            cell["d17_event_controls_by_pose"] = {
                r["pose"]: r["d17_event_control"] for r in rows if r.get("d17_event_control") is not None
            }
        cells.append(cell)

    cases = []
    for pose, wire, condition in CASES:
        by_choice = {(d, g): panel[pose, wire, condition, d, g]["native_success"]
                     for d, g in CHOICES}
        true_grip = int(condition == "D17")
        outcomes = {choice_name(d, g): by_choice[d, g] for d, g in CHOICES}
        outcomes.update(true=by_choice[wire, true_grip], true_wire_grip_off=by_choice[wire, 0],
                        true_wire_grip_on=by_choice[wire, 1])
        for d in SIGNS:
            outcomes[f"fixed_decode_{d:+d}_true_grip"] = by_choice[d, true_grip]
        cases.append(dict(pose=pose, wire=wire, condition=condition, outcomes=outcomes))

    true = [c["outcomes"]["true"] for c in cases]
    comparisons = {name: paired(true, [c["outcomes"][name] for c in cases])
                   for name in cases[0]["outcomes"] if name != "true"}
    by_factor = []
    for wire, condition in product(SIGNS, CONDITIONS):
        group = [c["outcomes"] for c in cases if (c["wire"], c["condition"]) == (wire, condition)]
        fixed = "true_wire_grip_on" if condition == "BASE" else "true_wire_grip_off"
        by_factor.append(dict(wire=wire, condition=condition, comparator=fixed,
                              **paired([o["true"] for o in group], [o[fixed] for o in group])))
    contact = [comparisons[name]["net"] for name in ("true_wire_grip_off", "true_wire_grip_on")]
    rescues = [c["pose"] for c in cases if c["wire"] == -1 and c["condition"] == "D17"
               and c["outcomes"]["true_wire_grip_on"] and not c["outcomes"]["true_wire_grip_off"]]
    gate = dict(decision="GO" if min(contact) > 0 and rescues else "NO_GO",
                net_vs_true_wire_grip_off=contact[0], net_vs_true_wire_grip_on=contact[1],
                reversed_d17_rescue_poses=rescues)
    differing_endpoints = [
        dict(pose=p, condition=c, net_x=d, regrasp=g)
        for p, c, d, g in product(POSES, CONDITIONS, SIGNS, GRIPS)
        if panel[p, 1, c, d, g]["native_success"] != panel[p, -1, c, -d, g]["native_success"]
    ]
    return dict(episodes=len(panel), poses=len(POSES), equivalent_endpoint_differences=differing_endpoints,
                gate=gate, paired_vs_true=comparisons,
                true_wire_pairs_by_factor=by_factor, cells=cells, cases=cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", type=Path, help="episodes.jsonl or its directory")
    path = parser.parse_args().episodes
    if path.is_dir():
        path = path / "episodes.jsonl"
    try:
        panel = read_panel(path)
        report = score(panel)
        differences = []
        for pose, condition, net_x, grip in product(POSES, CONDITIONS, SIGNS, GRIPS):
            traces = []
            for wire in SIGNS:
                row = panel[pose, wire, condition, wire * net_x, grip]
                record = json.loads((path.parent / row["episode"] / "result.json").read_text())
                traces.append(record["trajectory"])
            left, right = traces
            changed = [k for k in left if left[k] != right[k]]
            if changed:
                differences.append(dict(pose=pose, condition=condition, net_x=net_x,
                                        regrasp=grip, fields=changed))
        report["trajectory_differences"] = differences
    except (KeyError, TypeError, ValueError, OSError) as exc:
        parser.exit(2, f"Invalid pilot evidence: {exc}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
