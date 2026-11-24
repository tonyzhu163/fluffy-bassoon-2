"""Read source examples and encode causal public LIBERO execution history."""

import json
import operator
from pathlib import Path

import numpy as np

HISTORY_STEPS = 4
HISTORY_DIM = 52
HORIZON = 10
IMAGE_KEYS = ("agentview_image", "robot0_eye_in_hand_image")
POS = "robot0_eef_pos"
QUAT = "robot0_eef_quat_xyzw"
FINGERS = "robot0_gripper_qpos"


def _vector(value, size):
    value = np.asarray(value, dtype=np.float32)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError(f"Expected finite vector of shape ({size},), got {value.shape}")
    return value


def _width(raw):
    fingers = _vector(raw[FINGERS], 2)
    return abs(fingers[0] - fingers[1])


class PublicHistory:
    """Call append once per issued command, with the observations bracketing it."""

    def __init__(self):
        self._slots = np.zeros((HISTORY_STEPS, 13), dtype=np.float32)

    def reset(self):
        self._slots.fill(0)

    def append(self, issued_action, previous_raw, next_raw):
        action = _vector(issued_action, 7)
        delta = _vector(next_raw[POS], 3) - _vector(previous_raw[POS], 3)
        width = _width(next_raw)
        delta_width = width - _width(previous_raw)
        self._slots[:-1] = self._slots[1:]
        self._slots[-1] = np.concatenate((action, delta, [delta_width, width, 1.0]))

    def vector(self):
        return self._slots.flatten()


class SourceDataset:
    """Read one physically separate split, without retaining an open HDF5 handle."""

    def __init__(self, root, split="train"):
        if split not in ("train", "heldout"):
            raise ValueError("Choose train or heldout")
        self.root = Path(root)
        manifest = json.loads((self.root / "manifest.json").read_text())
        if manifest["schema"] != "libero-panda-openpi-v1":
            raise ValueError("Unknown source dataset schema")
        part = manifest["splits"][split]
        self._path = self.root / part["file"]
        self._index = [
            (traj["id"], traj["prompt"], t)
            for traj in part["trajectories"]
            for t in traj["t"]
        ]

    def __len__(self):
        return len(self._index)

    def __getitem__(self, index):
        import h5py

        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        trajectory, prompt, t = self._index[index]
        with h5py.File(self._path, "r") as f:
            raw = {k: f[k][index] for k in (*IMAGE_KEYS, POS, QUAT, FINGERS)}
            return dict(raw=raw, prompt=prompt, actions=f["actions"][index],
                        history=f["history"][index], trajectory_id=trajectory, t=t)
