"""Calibrate on human Panda demos and score on disjoint Panda demos and layouts."""
import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np

from signals import PHASE_DIR, drawer_front, drawer_probability

sys.path.insert(0, str(PHASE_DIR))
from common import save, sha, tiles
from reader import predict
from source import SHA, offsets

JOINT = "wooden_cabinet_1_top_level"
OPEN = -0.14
FIT = (20, 21, *range(24, 36), *range(40, 44))
HELD = range(10)
OLD = json.loads((PHASE_DIR / "evidence/model.json").read_bytes())


def aligned_drawer(group):
    states = group["states"][:]
    # Each image is post-action. The last image has no matching state.
    np.testing.assert_array_equal(states[1:, 1:8], group["obs/joint_states"][:-1])
    return states[1:, offsets(group)[JOINT]]


def calibrate(archive):
    boundaries = {}
    for i in FIT:
        group = archive[f"data/demo_{i}"]
        frames = group["obs/agentview_rgb"]
        q = aligned_drawer(group)
        reset = drawer_front(frames[0])
        crossings = []
        for t in np.flatnonzero(np.diff(q < OPEN)):
            left, right = drawer_front(frames[t]), drawer_front(frames[t + 1])
            fraction = (OPEN - q[t]) / (q[t + 1] - q[t])
            crossings.append(left + fraction * (right - left) - reset)
        if not crossings or not np.isfinite(crossings).all():
            raise ValueError(f"Missing physical crossing or visible edge in demo {i}")
        boundaries[i] = float(np.median(crossings))
    return dict(source_h5_sha256=SHA, native_open_below_m=OPEN, crossing_px_by_demo=boundaries)


def sweep(group, layout):
    """Independent native renders. Change arm pose between opening and closing legs."""
    from arm_env import ArmEnv

    with ArmEnv("Panda", "libero_goal", 3) as arm:
        reset = arm.reset(layout, 42710300 + layout)["observation"]["agentview_image"]
        sim = arm._env.sim
        addr = sim.model._model.joint(JOINT).qposadr[0]
        states = group["states"]
        joints = [*arm.model_identity["joint_names"], *arm.model_identity["gripper_joints"]]
        pos = [sim.model._model.joint(name).qposadr[0] for name in joints]
        src = offsets(group)
        donor = [src[name] for name in joints]
        mm = [0, 25, 65, 95, 115, 127, 133, 137, 139.5, 140.5, 143, 147, 151, 157, 160]
        for leg, t in enumerate(np.linspace(1, len(states) - 1, 8, dtype=int)):
            sim.data.qpos[pos] = states[t][donor]
            for travel in (mm if leg % 2 == 0 else mm[::-1]):
                sim.data.qpos[addr] = -travel / 1000
                sim.forward()
                yield reset, sim.render(width=256, height=256, camera_name="agentview"), sim.data.qpos[addr]


def observations(group):
    frames = group["obs/agentview_rgb"]
    reset = frames[0]
    for t, q in enumerate(aligned_drawer(group)):
        if t % 5 == 0:
            yield reset, frames[t], q


def score(samples, thresholds):
    probabilities, deltas, q = [], [], []
    for reset, frame, position in samples:
        probabilities.append(drawer_probability(drawer_front(reset), drawer_front(frame), thresholds))
        deltas.append(tiles(frame) - tiles(reset))
        q.append(position)
    truth = np.asarray(q) < OPEN
    reclosed = np.maximum.accumulate(truth) & ~truth
    first_closed = np.r_[False, truth[:-1] & ~truth[1:]]
    old = predict(5 * np.arange(len(q)), np.stack(deltas), OLD)["instant"]
    result = dict(frames=len(q), open=int(truth.sum()), currently_reclosed=int(reclosed.sum()),
                  reclosures=int(first_closed.sum()))
    for name, p in (("current", np.array(probabilities)), ("previous", old)):
        advance = p > 0.5 if name == "current" else p >= 0.5
        wrong = advance != truth
        result[name] = dict(brier=float(np.mean((p - truth)**2)),
                            premature=int((advance & ~truth).sum()), missed=int((~advance & truth).sum()),
                            premature_after_reclosure=int((advance & reclosed).sum()),
                            missed_reclosure_transitions=int((advance & first_closed).sum()),
                            max_wrong_distance_m=float(np.max(np.abs(np.asarray(q)[wrong] - OPEN), initial=0)))
    return result


def summarize(cases):
    diff = np.array([case["current"]["brier"] - case["previous"]["brier"] for case in cases.values()])
    means = np.random.default_rng(20261124).choice(diff, size=(20000, len(diff))).mean(axis=1)
    return dict(cases=cases, mean_brier={key: float(np.mean([v[key]["brier"] for v in cases.values()]))
                                         for key in ("current", "previous")},
                paired_trajectory_brier_delta_95=np.quantile(means, [0.025, 0.975]).tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fit", "score"))
    parser.add_argument("source_h5", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if sha(args.source_h5) != SHA:
        parser.error("Expected the pinned original Panda HDF5")
    if args.mode == "score" and args.report is None:
        parser.error("score requires --report")
    with h5py.File(args.source_h5, "r") as archive:
        if args.mode == "fit":
            save(args.model, calibrate(archive))
        else:
            model = json.loads(args.model.read_bytes())
            thresholds = np.array(list(model["crossing_px_by_demo"].values()))
            demos = {i: score(observations(archive[f"data/demo_{i}"]), thresholds) for i in HELD}
            sweeps = {24+i: score(sweep(archive[f"data/demo_{i}"], 24+i), thresholds) for i in range(4)}
            save(args.report, dict(model_sha256=sha(args.model), demonstrations=summarize(demos),
                                   controlled_sweeps=summarize(sweeps)))
