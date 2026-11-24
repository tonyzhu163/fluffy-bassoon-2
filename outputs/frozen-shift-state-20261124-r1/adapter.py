"""State-lane adapters: change only the proprioceptive state the frozen policy is packed from.

The packer (`ur5e_motor.to_openpi_observation`) builds state from `robot0_eef_pos`, a rotation
vector of `robot0_eef_quat_xyzw` and `robot0_gripper_qpos`. Every adapter here edits those public
fields in the `observation` hook and nothing else.
"""
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
_HARNESS = str(HERE.parent / "frozen-vla-proposal-shift-20261124-r1")
if _HARNESS not in sys.path:
    sys.path.append(_HARNESS)

from harness import Adapter
import ur5e_motor

QUATS = ("robot0_eef_quat_xyzw", "robot0_eef_quat")
STATE = ("robot0_eef_pos", *QUATS, "robot0_gripper_qpos")
ARCHIVED = HERE / "source_reference.json"
CHECKPOINT_STATS = ur5e_motor.OPENPI_CKPT / "assets/physical-intelligence/libero/norm_stats.json"
STATS_ENV = "STATE_LANE_NORM_STATS"
SHUFFLE_SEED = 20261124


def source_sign(quat):
    """The same rotation on the source branch.

    Every source training state has positive rotation-vector X, and the packer's rotation vector
    has the sign of the quaternion's vector part, so the source convention is x >= 0.
    """
    quat = np.asarray(quat)
    return -quat if quat[0] < 0 else quat


def to_source(raw):
    for key in QUATS:
        if key in raw:
            raw[key] = source_sign(raw[key])
    return raw


def quat_of_rotvec(rotation):
    """xyzw quaternion that the packer turns back into `rotation` (norm below 2 pi)."""
    rotation = np.asarray(rotation, dtype=np.float64)
    angle = np.linalg.norm(rotation)
    if not 1e-9 < angle < 2 * np.pi:
        raise ValueError("Rotation vector norm must lie in (0, 2 pi)")
    return np.r_[rotation / angle * np.sin(angle / 2), np.cos(angle / 2)]


def source_mean():
    """Mean source state (8,) and where it came from.

    The frozen checkpoint's own state mean when its statistics file is present (run.sh requires
    it), otherwise the archived Panda mean on this task written by measure.py, for CPU work.
    """
    path = Path(os.environ.get(STATS_ENV, CHECKPOINT_STATS))
    if STATS_ENV in os.environ or path.is_file():
        stats = json.loads(path.read_text())
        mean = np.asarray(stats.get("norm_stats", stats)["state"]["mean"], dtype=np.float64)[:8]
        origin = f"checkpoint:{path}"
    else:
        mean = np.asarray(json.loads(ARCHIVED.read_text())["state_mean"], dtype=np.float64)
        origin = f"archived:{ARCHIVED.name}"
    if mean.shape != (8,) or not np.isfinite(mean).all() or mean[3] <= 0:
        raise ValueError(f"Unusable source state mean from {origin}: {mean}")
    return mean, origin


def state_fields(raw):
    return {key: np.array(raw[key]) for key in STATE if key in raw}


class StateMapped(Adapter):
    """Report the hand rotation on the source quaternion branch. Position and fingers are unchanged.

    Kinova3 carries the Panda gripper and its grip site is aligned to the released reset, so the
    archived records support no fixed position or finger offset (see RESULT.md). The 15-control
    calibration is not needed for the map; `reset` only counts how many of its four frames were
    reported on the other branch.
    """

    name = "state-mapped"

    def reset(self, support, first):
        frames = [frame[key] for frame in support.get("frames", []) for key in QUATS[:1] if key in frame]
        self.calibration_frames_off_source_branch = sum(np.asarray(quat)[0] < 0 for quat in frames)

    def observation(self, raw):
        return to_source(raw)


class StateDropped(Adapter):
    """Give the policy the mean source state at every replan."""

    name = "state-dropped"

    def __init__(self, mean=None):
        self.mean, self.origin = (np.asarray(mean, dtype=np.float64), "argument") if mean is not None else (None, None)

    def reset(self, support, first):
        if self.mean is None:
            self.mean, self.origin = source_mean()
            print(f"state-dropped: {self.mean.tolist()} from {self.origin}", file=sys.stderr, flush=True)
        self.fields = {"robot0_eef_pos": self.mean[:3], "robot0_gripper_qpos": self.mean[6:8],
                       **{key: quat_of_rotvec(self.mean[3:6]) for key in QUATS}}

    def observation(self, raw):
        for key, value in self.fields.items():
            if key in raw:
                raw[key] = value.astype(np.asarray(raw[key]).dtype)
        return raw


class StateShuffled(Adapter):
    """Give the policy a real state of this body from another time.

    The pool is the four calibration frames plus every earlier replan of this episode, never the
    current one. Position, rotation and fingers move together from one frame. `source` applies the
    source quaternion branch as well.
    """

    name = "state-shuffled"
    source = False

    def reset(self, support, first):
        self.rng = np.random.default_rng(SHUFFLE_SEED)
        self.pool = [{key: np.asarray(value) for key, value in frame.items() if key in STATE}
                     for frame in support.get("frames", [])]
        self.current = state_fields(first["observation"])
        self.drawn = []

    def observation(self, raw):
        if not self.pool:
            raise ValueError("No earlier state to draw from: the calibration record has no frames")
        index = int(self.rng.integers(len(self.pool)))
        self.drawn.append(index)
        for key, value in self.pool[index].items():
            if key in raw:
                raw[key] = value.astype(np.asarray(raw[key]).dtype)
        return to_source(raw) if self.source else raw

    def observe(self, before, issued, after):
        self.pool.append(self.current)
        self.current = state_fields(after)


class StateMappedShuffled(StateShuffled):
    name = "state-mapped-shuffled"
    source = True


ADAPTERS = [StateMapped, StateDropped, StateShuffled, StateMappedShuffled]


if __name__ == "__main__":
    mean, origin = source_mean()
    print(json.dumps(dict(adapters=[item.name for item in ADAPTERS], dropped_state=mean.tolist(), dropped_state_from=origin)))
