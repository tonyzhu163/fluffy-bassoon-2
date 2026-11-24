"""Offline: does a progress-aligned donor window predict a held-out source success's next five controls?

Archived tapes only. Nothing here is a rollout or task evidence.
"""
import json
from pathlib import Path

import numpy as np

import donors

HERE = Path(__file__).resolve().parent
RULES = ("progress", "time", "same_phase_mean", "any_window_mean")


def errors(query, donor, frame="reset"):
    """Per query window, the error of each rule's donor prediction (NaN = the donor never visits the phase)."""
    keys, coordinates, targets = query.windows(frame)
    other_keys, other_coordinates, other_targets = donor.windows(frame)
    out = {rule: [] for rule in RULES}
    for w, (key, coordinate, target) in enumerate(zip(keys, coordinates, targets)):
        gap = np.linalg.norm(other_targets - target, axis=1)
        index = donors.aligned(other_keys, other_coordinates, key, coordinate)
        same = [i for i, other in enumerate(other_keys) if other == key]
        out["progress"].append(np.nan if index is None else gap[index])
        out["time"].append(gap[donors.timed(other_keys, w)])
        out["same_phase_mean"].append(gap[same].mean() if same else np.nan)
        out["any_window_mean"].append(gap.mean())
    return {rule: np.array(values) for rule, values in out.items()}


def interval(values, seed=0, draws=10000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values), size=(draws, len(values)))].mean(axis=1)
    return dict(mean=float(values.mean()), low=float(np.quantile(means, .025)), high=float(np.quantile(means, .975)),
                n=len(values), positive=int((values > 0).sum()))


def heldout(tapes, frame="reset", pick=lambda query, donor: donor.layout != query.layout):
    """Per query tape: mean error over its windows and its donors, on windows where every rule has a prediction."""
    rows = {}
    for query in tapes:
        per = {rule: [] for rule in RULES}
        covered = []
        for donor in tapes:
            if donor is query or not pick(query, donor):
                continue
            found = errors(query, donor, frame)
            keep = ~np.isnan(found["progress"])
            covered.append(keep.mean())
            for rule in RULES:
                per[rule].append(found[rule][keep].mean())
        if covered:
            rows[query.name] = dict({rule: float(np.mean(values)) for rule, values in per.items()},
                                    progress_coverage=float(np.mean(covered)), donors=len(covered))
    return rows


def paired(rows):
    return {f"{rule}_minus_progress": interval([row[rule] - row["progress"] for row in rows.values()])
            for rule in RULES[1:]}


def jaw_audit(tape):
    """Does the sign of the issued jaw command track the archived aperture one window later?"""
    closed = np.array([donors.jaw_closed(tape.commands[5 * w:5 * w + 5]) for w in range(len(tape.aperture) - 1)])
    narrowing = np.diff(tape.aperture) < -1e-3
    widening = np.diff(tape.aperture) > 1e-3
    return dict(windows=len(closed), closed_commands=int(closed.sum()),
                narrowing_given_closed_and_moving=int((closed & narrowing).sum()),
                widening_given_closed=int((closed & widening).sum()),
                narrowing_given_open=int((~closed & narrowing).sum()),
                widening_given_open_and_moving=int((~closed & widening).sum()))


def target_diagnostic(source):
    """Archived Kinova3 raw episodes against progress-aligned source windows from other layouts."""
    rows = {}
    for layout in (34, 35, 36, 37):
        tape = donors.cue_tape("Kinova3", layout)
        keys, _, _ = tape.windows()
        found = [errors(tape, donor) for donor in source if donor.layout != layout]
        nearest = np.nanmin(np.array([item["progress"] for item in found]), axis=0)
        phases = []
        for key in keys:
            if not phases or phases[-1][0] != key:
                phases.append([key, 0])
            phases[-1][1] += 1
        by_phase = {}
        for key in sorted(set(keys)):
            mask = np.array([other == key for other in keys])
            label = ("drawer open" if key[0] else "drawer shut") + (", jaw closed" if key[1] else ", jaw open")
            by_phase[label] = dict(windows=int(mask.sum()), mean_nearest_donor_gap=float(np.nanmean(nearest[mask])))
        rows[tape.name] = dict(native_success=tape.success, controls=len(tape.commands),
                               phase_runs=[[list(key), count] for key, count in phases],
                               mean_nearest_donor_gap=float(np.nanmean(nearest)), by_phase=by_phase)
    return rows


def main():
    tapes = donors.library()
    panda = [tape for tape in tapes if tape.arm == "Panda"]
    cross = heldout(tapes)
    arm = heldout(tapes, pick=lambda query, donor: donor.arm != query.arm)
    report = dict(
        evidence="offline analysis of archived tapes; no rollout, no task evidence",
        unit="norm of the summed five-control translation command, in normalized action units",
        tapes={tape.name: dict(controls=len(tape.commands), drawer_source=tape.drawer_source) for tape in tapes},
        leave_layout_out=dict(per_query=cross, paired=paired(cross)),
        other_arm_donors_only=dict(per_query=arm, paired=paired(arm)),
        panda_pair={frame: heldout(panda, frame) for frame in ("reset", "tool")},
        panda_tool_rotation_degrees={tape.name: float(np.degrees(np.arccos(np.clip(
            (np.trace(tape.rotation, axis1=1, axis2=2) - 1) / 2, -1, 1))).max()) for tape in panda},
        jaw_command_versus_aperture={tape.name: jaw_audit(tape) for tape in panda},
        kinova3_diagnostic=target_diagnostic(tapes),
    )
    (HERE / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
