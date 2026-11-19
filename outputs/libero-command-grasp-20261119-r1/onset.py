"""Compare live BASE/D17 onset on the six registered nominal-wire, identity-decode starts."""

import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np


POSES = range(30, 36)
CASES = ("BASE", "D17")
CONTROL = ("wire_action", "selected", "phase", "reward", "success", "terminated", "truncated")
QUERY = ("query_rgb", "query_state", "query_proposal", "query_noise", "query_jax_key")


def read_panel(path):
    marker = json.loads(path.with_name("complete.json").read_text())
    if (marker["mode"], marker["episodes"], marker["equivalent_pairs_checked"],
            marker["distinct_physics"]) != ("pilot", 96, 48, 6):
        raise ValueError("Expected the completed six-pose, 96-episode pilot")
    panel = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        key = tuple(row[k] for k in ("pose", "wire", "condition", "decode", "regrasp"))
        if key in panel:
            raise ValueError(f"Duplicate branch {key}")
        panel[key] = row
    if panel.keys() != set(product(POSES, (1, -1), CASES, (1, -1), (0, 1))):
        raise ValueError("Episode keys differ from the registered panel")
    return panel


class Tape:
    def __init__(self, root, row):
        self.row = row
        self.n = row["controls"]
        self.choice = row["regrasp_start_after"]
        name = f"p{row['pose']}-{row['condition']}-wire+1-decode+1-grip{row['regrasp']}"
        if row["episode"] != name or row["seed"] != 19193000 + row["pose"]:
            raise ValueError(f"{name}: unexpected episode identity")
        with np.load(root / name / "public.npz", allow_pickle=False) as public:
            self.data = {k: public[k] for k in (*CONTROL, *QUERY, "query_control", "robot_state",
                                                "final_rgb", "final_state")}
        with np.load(root / name / "evaluator.npz", allow_pickle=False) as evaluator:
            self.data.update({k: evaluator[k] for k in ("native_input", "applied")})
        a = self.data
        n = self.n
        times = a["query_control"]
        self.slots = {int(t): i for i, t in enumerate(times)}
        self.rgb = {t: a["query_rgb"][i] for t, i in self.slots.items()}
        self.rgb[n] = a["final_rgb"]  # Motor preprocessing, with no inference at termination.
        if (not 1 <= n <= 300 or a["robot_state"].shape != (n + 1, 9)
                or any(len(a[k]) != n for k in CONTROL)
                or any(a[k].shape != (n, 7) for k in ("native_input", "applied", "wire_action"))
                or not np.array_equal(a["wire_action"], a["native_input"])
                or any(len(a[k]) != len(times) for k in QUERY)
                or not np.array_equal(times, np.flatnonzero(a["phase"][::5] == 0) * 5)):
            raise ValueError(f"{name}: invalid control, state or query timeline")
        started = np.flatnonzero(a["phase"])
        if (self.choice != (int(started[0]) if started.size else None)
                or (not row["regrasp"] and self.choice is not None)):
            raise ValueError(f"{name}: scheduled intervention disagrees with executed phases")
        done = a["terminated"] | a["truncated"]
        if (done[:-1].any() or not done[-1] or type(row["native_success"]) is not bool
                or row["native_success"] != bool(a["success"].any())):
            raise ValueError(f"{name}: invalid native outcome")
        changed = a["applied"] != a["native_input"]
        if (changed[:, :6].any() or (row["condition"] == "BASE" and changed.any())
                or np.any(a["applied"][changed[:, -1], -1] != -1)
                or int(changed[:, -1].sum()) != row["gripper_overrides"]):
            raise ValueError(f"{name}: inconsistent native actuator readback")
        if any(not np.isfinite(a[k]).all() for k in
               ("robot_state", "native_input", "applied", "query_state", "query_proposal")):
            raise ValueError(f"{name}: nonfinite action, proposal or state")


def first_diff(a, b, offset=0):
    """First unequal time row in two already aligned arrays."""
    if a.shape != b.shape:
        raise ValueError(f"Incompatible paired array shapes {a.shape} and {b.shape}")
    different = np.any(a != b, axis=tuple(range(1, a.ndim)))
    hits = np.flatnonzero(different)
    return int(hits[0]) + offset if hits.size else None


def first_query_diff(left, right, times, key):
    return next((t for t in times if not np.array_equal(
        left.data[key][left.slots[t]], right.data[key][right.slots[t]]
    )), None)


def check_on_off(off, on):
    # Commands up to and including d have run. Query d is skipped when recovery starts.
    shared = min(off.n, on.n)
    end = min(shared, on.choice) if on.choice is not None else shared
    mismatches = {}
    for k in (*CONTROL, "native_input", "applied", "robot_state"):
        state = k == "robot_state"
        stop = end + int(state)
        t = first_diff(off.data[k][:stop], on.data[k][:stop], offset=int(not state))
        if t is not None:
            mismatches[k] = t
    # Both gates have logged every ordinary query strictly before this boundary.
    a = {t for t in off.slots if t < end}
    b = {t for t in on.slots if t < end}
    if a != b:
        mismatches["query_control"] = min(a ^ b)
    for k in QUERY:
        t = first_query_diff(off, on, sorted(a & b), k)
        if t is not None:
            mismatches[k] = t
    if on.choice is None and off.n == on.n:
        for k in ("final_rgb", "final_state"):
            if not np.array_equal(off.data[k], on.data[k]):
                mismatches[k] = end
    result = dict(checked_through=end, equal=not mismatches)
    if mismatches:
        result["first_mismatch_after"] = mismatches
    return result


def meters(jaw, wrist, index):
    return dict(jaw_max_abs_m=float(np.max(jaw[index])), wrist_xyz_l2_m=float(np.max(wrist[index])))


def diagnose(root, panel, pose):
    tapes = {(case, grip): Tape(root, panel[pose, 1, case, 1, grip])
             for case, grip in product(CASES, (0, 1))}
    base, d17 = (tapes[case, 0] for case in CASES)
    for tape in tapes.values():
        for k in ("initial_physics", "initial_public", "first_motor_input", "first_proposal", "noise_bank"):
            if not tape.row[k] or tape.row[k] != base.row[k]:
                raise ValueError(f"Pose {pose}: unpaired {k}")
    checks = {c: check_on_off(tapes[c, 0], tapes[c, 1]) for c in CASES}
    h = min(base.n, d17.n)
    a, b = base.data, d17.data
    prefix = {}
    for k in ("native_input", "applied"):
        first = first_diff(a[k][:h], b[k][:h], offset=1)
        prefix[k] = h if first is None else first - 1
    grip_split = first_diff(a["applied"][:h, -1], b["applied"][:h, -1], offset=1)
    control = first_diff(b["native_input"][:, -1], b["applied"][:, -1], offset=1)
    override = None
    if control is not None:
        i = control - 1
        override = dict(control=control, d17_requested=float(b["native_input"][i, -1]),
                        d17_applied=float(b["applied"][i, -1]))
        if control <= h:
            override.update(base_applied=float(a["applied"][i, -1]),
                            same_requested_through_control=prefix["native_input"] >= control,
                            same_applied_before_control=prefix["applied"] >= i)

    x, y = (t["robot_state"][:h + 1] for t in (a, b))
    delta = x.astype(np.float64) - y.astype(np.float64)
    jaw = np.abs(delta[:, 7:9]).max(axis=1)
    wrist = np.linalg.norm(delta[:, :3], axis=1)
    parts = {"jaw": slice(7, 9), "wrist_xyz": slice(0, 3), "wrist_quat": slice(3, 7)}
    state = {k: first_diff(x[:, cols], y[:, cols]) for k, cols in parts.items()}
    queries = sorted(base.slots.keys() & d17.slots.keys())
    for k in ("query_noise", "query_jax_key"):
        if first_query_diff(base, d17, queries, k) is not None:
            raise ValueError(f"Pose {pose}: unpaired {k} at a shared ordinary query")
    motor = {k: first_query_diff(base, d17, queries, f"query_{k}") for k in ("rgb", "state", "proposal")}
    rgb_times = sorted(base.rgb.keys() & d17.rgb.keys())
    rgb = next((t for t in rgb_times if not np.array_equal(base.rgb[t], d17.rgb[t])), None)
    observed = [t for t in (*state.values(), *motor.values(), rgb) if t is not None]
    if prefix["native_input"] < h:  # With both signs +1 these are also the public wire actions.
        observed.append(prefix["native_input"] + 1)

    choice = {}
    for case in CASES:
        d = tapes[case, 1].choice
        if d is None:
            choice[case] = None
            continue
        end = min(d, h)
        # State[d] and commands 1..d exist at the choice. Only earlier proposals were issued.
        state_seen = any(t is not None and t <= end for t in state.values())
        prior_query = any(t is not None and t < d for t in motor.values())
        paired = checks[case]["equal"] and checks[case]["checked_through"] >= end
        available = state_seen or prefix["native_input"] < end or prior_query
        verdict = available if paired and (available or end == d) else None
        boundary_rgb = not np.array_equal(base.rgb[d], d17.rgb[d]) if d in rgb_times else None
        # An ordinary call at d samples the pre-action cameras, even though recovery skips inference.
        # A terminal image can document divergence but cannot support a choice in that episode.
        prospective = verdict
        if paired and d in base.slots and d in d17.slots and boundary_rgb:
            prospective = True
        choice[case] = dict(
            after=d, shared_through=end,
            recorded_difference_by_choice=verdict, ordinary_rgb_diff_at_boundary=boundary_rgb,
            ordinary_histories_differ_if_boundary_camera_used=prospective,
            both_ordinary_branches_active_at_choice=d < h,
            max_through_shared=meters(jaw, wrist, slice(end + 1)),
            at_choice=meters(jaw, wrist, d) if end == d else None,
        )
    return dict(
        pose=pose, native_success={c: {label: tapes[c, g].row["native_success"]
                                       for g, label in enumerate(("off", "on"))} for c in CASES},
        ordinary_controls={c: tapes[c, 0].n for c in CASES}, shared_through=h,
        equal_command_prefix_controls=prefix, first_paired_applied_gripper_diff_control=grip_split,
        first_d17_override=override, d17_event_control=d17.row["d17_event_control"],
        first_state_diff_after=state,
        max_meters_during_equal_applied_prefix=meters(jaw, wrist, slice(prefix["applied"] + 1)),
        meters_at_first_diff={k: meters(jaw, wrist, t) for k, t in state.items() if t is not None},
        first_shared_query_diff_after=motor, first_rgb_diff_including_final_after=rgb,
        first_recorded_public_difference_after=min(observed, default=None),
        on_off_prefix=checks, regrasp_choice=choice,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot", type=Path, help="completed pilot directory or episodes.jsonl")
    path = parser.parse_args().pilot
    if path.is_dir():
        path = path / "episodes.jsonl"
    try:
        panel = read_panel(path)
        poses = [diagnose(path.parent, panel, pose) for pose in POSES]
    except (OSError, KeyError, ValueError, TypeError, IndexError) as exc:
        parser.exit(2, f"Invalid pilot evidence: {exc}\n")
    counts = {}
    for case in CASES:
        choices = [p["regrasp_choice"][case] for p in poses]
        verdicts = [c["recorded_difference_by_choice"] for c in choices if c is not None]
        counts[case] = dict(different=verdicts.count(True), equal=verdicts.count(False),
                            undetermined=verdicts.count(None), no_intervention=choices.count(None))
    print(json.dumps(dict(
        wire=1, decode=1, poses=poses, recorded_difference_by_choice_counts=counts,
        time="Control 1 acts from reset. After d includes state[d] before command d+1.",
        choice_basis="Each condition supplies its own enabled start time. Both comparison histories keep "
                     "regrasp off. The verdict uses public state and issued commands through the boundary "
                     "and shared motor queries strictly before it. Ordinary boundary RGB is pre-action "
                     "evidence for a prospective reader. The enabled branch skips that query.",
        missing="Null onset means no recorded difference on comparable samples through the shared horizon. "
                "Null override means D17 never changed its own gripper input. Null choice means no intervention. "
                "A null choice verdict indicates a failed prefix check or insufficient shared coverage.",
        magnitude="Jaw distance is the largest absolute per-finger coordinate difference. Wrist distance "
                  "is Euclidean XYZ translation. Quaternions are compared componentwise.",
        interpretation="Exact differences establish existence only. Neither tiny separations nor later "
                       "trajectory differences establish useful discrimination or causally necessary memory.",
    ), allow_nan=False))


if __name__ == "__main__":
    main()
