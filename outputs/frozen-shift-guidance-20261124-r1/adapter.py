"""Choose, among K samples of the frozen policy, the one nearest a source-arm success at the same public progress."""
import hashlib
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
for folder in (HERE, HERE.parent / "frozen-vla-proposal-shift-20261124-r1", HERE.parent / "frozen-vla-selector-20261124-r1"):
    if str(folder) not in sys.path:
        sys.path.append(str(folder))

import donors
from harness import Adapter

K = 4


def rotation_matrix(xyzw):
    x, y, z, w = np.asarray(xyzw, dtype=np.float64) / np.linalg.norm(xyzw)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def quaternion(raw):
    return raw["robot0_eef_quat_xyzw"] if "robot0_eef_quat_xyzw" in raw else raw["robot0_eef_quat"]


def drawer_open(reset_front, image):
    """The selector's camera reader: a strict majority of source crossings passed."""
    import signals  # needs the checked-in source calibration; imported when an episode starts
    front = signals.drawer_front(np.asarray(image))
    return signals.drawer_probability(reset_front, front, signals._drawer_thresholds()) > .5


class Sampler(Adapter):
    """The untreated noise first, then three draws from the start's shared candidate stream."""

    def noises(self, base, rng):
        return [base] + [rng.standard_normal(base.shape).astype(base.dtype) for _ in range(K - 1)]


class Random(Sampler):
    name = "guidance_random"

    def reset(self, support, first):
        self.log = []

    def choose(self, raw, proposals):
        seed = hashlib.sha256(b"".join(np.ascontiguousarray(p).tobytes() for p in proposals)).digest()[:8]
        index = int(np.random.default_rng(int.from_bytes(seed, "little")).integers(len(proposals)))
        self.log.append(dict(chosen=index))
        return index


class Guidance(Sampler):
    """`tapes` are donors.Tape records. align is "progress" or "time"; frame is "reset" or "tool"."""

    def __init__(self, name, tapes, align="progress", frame="reset", reader=drawer_open):
        if align not in ("progress", "time") or frame not in ("reset", "tool") or not tapes:
            raise ValueError("Unsupported guidance configuration")
        self.name, self.align, self.frame, self.reader = name, align, frame, reader
        self.donors = [(tape.name, *tape.windows(frame)) for tape in tapes]

    def reset(self, support, first):
        import signals
        raw = first["observation"]
        self.reset_front = signals.drawer_front(np.asarray(raw["agentview_image"]))
        self.reset_rotation = rotation_matrix(quaternion(raw))
        self.progress, self.replan, self.log = donors.Progress(), 0, []

    def rotation(self, raw):
        """Tool rotation since reset; a constant sensor-frame offset cancels."""
        if self.frame == "reset":
            return None
        return rotation_matrix(quaternion(raw)) @ self.reset_rotation.T

    def choose(self, raw, proposals):
        key, coordinate = self.progress.at_replan(self.reader(self.reset_front, raw["agentview_image"]))
        targets = {}
        for name, keys, coordinates, values in self.donors:
            index = (donors.aligned(keys, coordinates, key, coordinate) if self.align == "progress"
                     else donors.timed(keys, self.replan))
            if index is not None:
                targets[name] = (index, values[index])
        rotation = self.rotation(raw)
        moves = np.array([donors.displacement(proposal, rotation) for proposal in proposals])
        event = dict(replan=self.replan, drawer_open=key[0], jaw_closed=key[1], coordinate=coordinate.tolist(),
                     donor_windows={name: index for name, (index, _) in targets.items()})
        if targets:
            gaps = np.array([[np.linalg.norm(move - value) for _, value in targets.values()] for move in moves])
            nearest = gaps.min(axis=1)
            chosen = int(np.argmin(nearest))
            event.update(distance=nearest.tolist(), donor=list(targets)[int(np.argmin(gaps[chosen]))])
        else:
            chosen = 0  # no donor visited this phase: keep the untreated sample
        event.update(chosen=chosen, fallback=not targets)
        self.log.append(event)
        self.replan += 1
        return chosen

    def observe(self, before, issued, after):
        self.progress.issued(issued, self.rotation(before))


def for_layout(layout, tapes=None):
    """Layout-aware factories for the lane driver. The shared panel cannot pass the layout."""
    tapes = donors.library() if tapes is None else tapes
    other = [tape for tape in tapes if tape.layout != layout]
    same = [tape for tape in tapes if tape.layout == layout]
    made = {
        "guidance": lambda: Guidance("guidance", other),
        "guidance_time": lambda: Guidance("guidance_time", other, align="time"),
        "guidance_random": Random,
    }
    if same:
        made["guidance_same"] = lambda: Guidance("guidance_same", same)
    return made


def _iiwa3435():
    return Guidance("guidance_iiwa3435", [donors.cue_tape("IIWA", 34), donors.cue_tape("IIWA", 35)])


# Shared panel: donors on layouts 34 and 35 only, so every failure start and 40-47 see other-layout donors.
ADAPTERS = [_iiwa3435, Random]
