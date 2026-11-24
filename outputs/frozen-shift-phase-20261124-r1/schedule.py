"""When would each prompt be active along the archived untreated Kinova3 failures?

CPU readback of archived evidence. No policy call, no simulator, no reward.
Usage: python schedule.py [schedule.json]
"""
import gzip
import json
from pathlib import Path
import sys
import tarfile

import numpy as np

import adapter as lane

OUTPUTS = Path(__file__).resolve().parents[1]
FRAMES = OUTPUTS / "frozen-vla-source-view-20261123-r1/evidence/cpu01"
TRUTH = OUTPUTS / "frozen-vla-reachability-20261123-r1/evidence"
CUE_TAPES = OUTPUTS / "frozen-transfer-cue-20261124-r1/evidence/native01/native-tapes.tar.gz"
NATIVE_OPEN_BELOW_M = json.loads(
    (OUTPUTS / "frozen-vla-selector-20261124-r1/drawer-calibration.json").read_bytes())["native_open_below_m"]
FAILURES = (14, 15, 17, 36, 37, 38)


def spans(controls, placing, end):
    """Merge per-sample phases into [start, stop) control intervals per prompt."""
    out = []
    for i, (control, flag) in enumerate(zip(controls, placing)):
        stop = controls[i + 1] if i + 1 < len(controls) else end
        prompt = lane.PLACING if flag else lane.OPENING
        if out and out[-1]["prompt"] == prompt:
            out[-1]["stop"] = stop
        else:
            out.append(dict(prompt=prompt, start=control, stop=stop))
    return out


def frames(layout):
    """Current reader on the archived full-resolution frames of the native failure tape."""
    paths = sorted(FRAMES.glob(f"s{layout}-t*.npz"))
    controls = [int(path.stem.split("-t")[1]) for path in paths]
    images = [np.load(path)["raw_agentview"] for path in paths]
    reset_front, _ = lane.read(None, images[0])
    with gzip.open(TRUTH / f"raw-drawer-{layout}-Kinova3.json.gz") as file:
        truth = json.load(file)
    travel = {0: truth["drawer_initial"], **{row["step"]: row["drawer_q"] for row in truth["frames"]}}
    rows, held, state = [], [], False
    for control, image in zip(controls, images):
        front, score = lane.read(reset_front, image)
        state = lane.hysteretic(state, reset_front, front, score)
        held.append(state)
        rows.append(dict(control=control, front_px=front, displacement_px=None if front is None else front - reset_front,
                         score=score, instant=lane.PLACING if lane.instant(False, reset_front, front, score) else lane.OPENING,
                         hysteretic=lane.PLACING if state else lane.OPENING,
                         clock=lane.PLACING if lane.clock_open(control) else lane.OPENING,
                         audit_drawer_m=travel[control], audit_open=travel[control] < NATIVE_OPEN_BELOW_M))
    # Audit only: the native open predicate at every replan of the whole tape.
    last = max(travel)
    replans = list(range(0, last + 1, 5))
    is_open = [travel[t] < NATIVE_OPEN_BELOW_M for t in replans]
    return dict(
        evidence="archived full-resolution frames, current reader", tape=truth["source"], tape_sha256=truth["sha256"],
        reset_front_px=reset_front, frames=rows,
        reader_agrees_with_audit_on_frames=all((row["score"] > .5) == row["audit_open"] for row in rows),
        audit=dict(note="simulator drawer joint from the CPU replay of the same tape; not an adapter input",
                   controls=last, min_drawer_m=min(travel.values()), final_drawer_m=travel[last],
                   first_open_replan=next((t for t, flag in zip(replans, is_open) if flag), None),
                   reclosed_replans=[t for a, b, t in zip(is_open, is_open[1:], replans[1:]) if a and not b],
                   schedule_if_reader_matched_audit=spans(replans, is_open, last)),
        clock_schedule=spans(replans, [lane.clock_open(t) for t in replans], last),
    )


def previous_reader(layout):
    """Archived per-replan scores of the earlier four-tile reader in the native cue tapes.

    The current reader was not run on these episodes and no image was archived.
    """
    out = {}
    with tarfile.open(CUE_TAPES) as tapes:
        for mode in ("raw", "tuned"):
            episode = json.load(tapes.extractfile(f"gpu01/episodes/Kinova3-{layout}-{mode}.json"))
            calls = episode["calls"]
            controls = [call["t"] for call in calls]
            scores = [call["visual"] for call in calls]
            flags = [score >= 0.5 for score in scores]
            out[mode] = dict(
                native_success=bool(episode["native"]["success"]), controls=episode["native"]["controls"],
                first_open_replan=next((t for t, flag in zip(controls, flags) if flag), None),
                returns_to_closed=[t for a, b, t in zip(flags, flags[1:], controls[1:]) if a and not b],
                schedule=spans(controls, flags, episode["native"]["controls"]),
                scores=[[t, round(score, 4)] for t, score in zip(controls, scores)])
    return dict(evidence="archived scores of the previous four-tile reader at 0.5; no frames, current reader not run",
                branches=out, clock_schedule=spans(controls, [lane.clock_open(t) for t in controls], 300))


def main(path):
    report = dict(
        scope="Untreated archived trajectories. A held prompt changes the trajectory, so these are not its schedule.",
        prompts=dict(opening=lane.OPENING, placing=lane.PLACING),
        source_band_px=[float(lane.THRESHOLDS.min()), float(lane.THRESHOLDS.max())],
        clock_first_placing_control=next(t for t in range(0, 300, 5) if lane.clock_open(t)),
        layouts={})
    for layout in FAILURES:
        if list(FRAMES.glob(f"s{layout}-t*.npz")):
            report["layouts"][layout] = frames(layout)
        elif layout in (36, 37):
            report["layouts"][layout] = previous_reader(layout)
        else:
            report["layouts"][layout] = dict(evidence="none: no frame and no reader score is archived in the repository")
    Path(path).write_text(json.dumps(report, indent=1) + "\n")
    return report


if __name__ == "__main__":
    result = main(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("schedule.json"))
    for layout, entry in result["layouts"].items():
        print(layout, entry["evidence"])
        for row in entry.get("frames", []):
            print("  ", row["control"], row["displacement_px"], row["score"], row["hysteretic"], row["audit_drawer_m"])
        if "audit" in entry:
            print("   audit", {k: v for k, v in entry["audit"].items() if k != "note"})
        for mode, branch in entry.get("branches", {}).items():
            print("  ", mode, branch["first_open_replan"], branch["returns_to_closed"], branch["schedule"])
