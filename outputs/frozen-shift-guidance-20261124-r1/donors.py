"""Archived source-arm successes as progress-keyed five-control windows."""
from dataclasses import dataclass
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np

OUTPUTS = Path(__file__).resolve().parents[1]
CUE = OUTPUTS / "frozen-transfer-donor-20261124-r1/inputs/raw-tapes.tar.gz"
POSE = OUTPUTS / "frozen-transfer-pose-20261124-r1/evidence/replay"
DRAWER = OUTPUTS / "frozen-vla-reachability-20261123-r1/evidence"
SHA256 = {
    "raw-tapes.tar.gz": "5b8b787f7715a5c5ba5eb3a1f16a25d4463222589ca99cac1fd17901d37ee750",
    "libero_goal-3-14-Panda.npz": "5435de12528da31f3d4c69fd340af1696fe29b5c2900f11ad72572b8356ce055",
    "libero_goal-3-15-Panda.npz": "05082affdc0bdcd0b8c47189e7919b433bbf2a1892e602d8eeb0145daaf21b0e",
    "raw-drawer-14-Panda.json.gz": "4f701bb9bfc8fb22863751b616cc83162287ffeaa3f8534db61cbcee9df4613b",
    "raw-drawer-15-Panda.json.gz": "bf21ba8f6458aac4e58218eb959234acd59f663f39546d38d9cba25759d3f516",
}
NATIVE_OPEN_M = -0.14  # native drawer threshold the camera reader is calibrated to
WINDOW = 5


def displacement(window, rotation=None):
    """Summed translation command of the first five controls, in the frame `rotation` maps from."""
    total = np.asarray(window, dtype=np.float64)[:WINDOW, :3].sum(axis=0)
    return total if rotation is None else rotation.T @ total


def jaw_closed(window):
    return bool(np.asarray(window)[:, 6].mean() > 0)


class Progress:
    """Phase key (drawer open, jaw commanded closed) and summed issued translation within the phase."""

    def __init__(self):
        self.key, self.coordinate, self.jaw = None, np.zeros(3), False

    def at_replan(self, drawer_open):
        key = (bool(drawer_open), self.jaw)
        if key != self.key:
            self.key, self.coordinate = key, np.zeros(3)
        return self.key, self.coordinate.copy()

    def issued(self, window, rotation=None):
        self.coordinate = self.coordinate + displacement(window, rotation)
        self.jaw = jaw_closed(window)


@dataclass
class Tape:
    arm: str
    layout: int
    commands: np.ndarray   # (N, 7) issued
    drawer: np.ndarray     # (W,) drawer-open flag at each replan
    drawer_source: str     # "camera gate" (public) or "drawer joint" (donor simulator state)
    rotation: np.ndarray = None   # (W, 3, 3) tool rotation since reset, if archived
    aperture: np.ndarray = None   # (W,) jaw opening, if archived
    success: bool = True

    @property
    def name(self):
        return f"{self.arm}-{self.layout}"

    def windows(self, frame="reset"):
        """keys, coordinates (W, 3) and targets (W, 3) for every complete window."""
        if frame == "tool" and self.rotation is None:
            raise ValueError(f"{self.name} archived no tool orientation")
        progress, keys, coordinates, targets = Progress(), [], [], []
        for w in range(len(self.commands) // WINDOW):
            rotation = self.rotation[w] if frame == "tool" else None
            window = self.commands[WINDOW * w:WINDOW * (w + 1)]
            key, coordinate = progress.at_replan(self.drawer[w])
            keys.append(key)
            coordinates.append(coordinate)
            targets.append(displacement(window, rotation))
            progress.issued(window, rotation)
        return keys, np.array(coordinates), np.array(targets)


def _verified(path):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != SHA256[Path(path).name]:
        raise ValueError(f"Archive digest changed: {path}")
    return data


def cue_tape(arm, layout, mode="raw"):
    """Frozen-policy episode from the cue study: issued commands and the archived camera gate."""
    with tarfile.open(fileobj=io.BytesIO(_verified(CUE))) as archive:
        record = json.load(archive.extractfile(f"gpu01/episodes/{arm}-{layout}-{mode}.json"))
    calls = record["calls"]
    commands = np.array([row for call in calls for row in call["sent"]], dtype=np.float64)
    if len(commands) != record["native"]["controls"] or record["query"]["init_state_id"] != layout:
        raise ValueError("Unexpected cue tape")
    drawer = np.array([call["visual"] >= .5 for call in calls])
    return Tape(arm, layout, commands, drawer, "camera gate", success=bool(record["native"]["success"]))


def panda_tape(layout):
    """Panda success replay: commands, tool rotation, jaw aperture and the drawer joint."""
    pose = np.load(io.BytesIO(_verified(POSE / f"libero_goal-3-{layout}-Panda.npz")))
    replay = json.loads(gzip.decompress(_verified(DRAWER / f"raw-drawer-{layout}-Panda.json.gz")))
    frames = replay["frames"]
    commands = pose["cmd"].astype(np.float64)
    if len(frames) != len(commands) or not replay["success"] or replay["case"]["init_state_id"] != layout:
        raise ValueError("Unexpected Panda replay")
    starts = np.arange(len(commands) // WINDOW + 1) * WINDOW
    starts = starts[starts < len(commands)]
    # frames[i] is the state after control i + 1; the state at a replan is the previous frame.
    drawer = np.array([i > 0 and frames[i - 1]["drawer_q"] < NATIVE_OPEN_M for i in starts])
    aperture = np.array([np.subtract(*frames[max(i - 1, 0)]["fingers"]) for i in starts])
    rotation = np.einsum("wij,kj->wik", pose["R"][starts], pose["R"][0])
    return Tape("Panda", layout, commands, drawer, "drawer joint", rotation, aperture)


def library():
    return [cue_tape("IIWA", layout) for layout in (34, 35, 36, 37)] + [panda_tape(14), panda_tape(15)]


def aligned(keys, coordinates, key, coordinate):
    """Donor window with the same phase key whose within-phase coordinate is nearest, or None."""
    same = [i for i, other in enumerate(keys) if other == key]
    if not same:
        return None
    return same[int(np.argmin(np.linalg.norm(coordinates[same] - coordinate, axis=1)))]


def timed(keys, replan):
    """Donor window at the same control index, held at the donor's last window."""
    return min(replan, len(keys) - 1)
