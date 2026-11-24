"""Causal motion and drawer-progress features at native five-control replans."""

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

import numpy as np


OUTPUTS = Path(__file__).resolve().parents[1]
MOTION_DIR = OUTPUTS / "frozen-vla-motion-models-20261123-r1"
PHASE_DIR = OUTPUTS / "frozen-vla-phase-transfer-20261123-r1"


def _load_module(name, path):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# These two checked-in modules have no sibling imports. Do not modify sys.path or
# register their generic filenames in sys.modules in the simulator worker.
_motion = _load_module("_selector_motion", MOTION_DIR / "motion.py")
_tiles = _load_module("_selector_phase_tiles", PHASE_DIR / "common.py").tiles
_source = json.loads((OUTPUTS / "frozen-vla-tracking-feedback-20261123-r1/source-model.json").read_bytes())
_prior = np.array(_source["prior_matrix"])
_noise = float(_source["source_residual_power"])
_ridge = json.loads((MOTION_DIR / "evidence/source-fit.json").read_bytes())["choice"]["jacobian"]
_phase = json.loads((PHASE_DIR / "evidence/model.json").read_bytes())
_patches = np.array(_phase["signed_luminance_tile_ids"])


def _finite(value, shape):
    data = np.asarray(value, dtype=np.float64)
    if data.shape != shape or not np.isfinite(data).all():
        raise ValueError(f"Expected finite values of shape {shape}, got {data.shape}")
    return data


def _chunk(value):
    data = np.asarray(value)
    if data.ndim != 2 or data.shape[1] != 7 or not len(data):
        raise ValueError("Expected a nonempty Nx7 action chunk")
    if not np.isfinite(data).all() or (np.abs(data) > 1).any():
        raise ValueError("Expected finite, bounded native OSC commands")
    return data


def _patch_values(raw):
    image = np.asarray(raw["agentview_image"])
    if image.dtype != np.uint8:
        raise ValueError("Expected the unflipped native RGB uint8 image")
    return _tiles(image)[_patches]


class Signals:
    """Episode-local history. Read operations are pure, including candidate scoring."""

    def __init__(self, support, initial_raw):
        q = np.asarray(initial_raw["robot0_joint_pos"])
        if q.ndim != 1 or not q.size:
            raise ValueError("Expected a vector of native joint angles in radians")
        _finite(q, q.shape)
        self._joints = _finite(support["joints"], (3, q.size)).copy()
        self._inputs = _finite(support["inputs"], (3, 6)).copy()
        self._response = _finite(support["response"], (3, 3)).copy()
        self._reset_patches = _patch_values(initial_raw)
        self._previous_object = self._object(initial_raw)
        self._previous_command = np.zeros(3)
        self._achieved = np.zeros(3)
        self._tracking_error = np.zeros(3)
        self._observed = False

    def _body(self, raw):
        """Weight each support row by a periodic kernel on joint angles.

        Each revolute joint has unit similarity at its support angle and zero
        at the opposite angle. The product localizes across joints without a
        target-tuned bandwidth. Leave weights unnormalized so distant or weak
        support shrinks to the source prior via the source-CV ridge penalty.
        """
        q = _finite(raw["robot0_joint_pos"], (self._joints.shape[1],))
        sqrt_weights = np.prod(np.abs(np.cos((self._joints - q) / 2)), axis=1)[:, None]
        trace = {"inputs": sqrt_weights * self._inputs, "response": sqrt_weights * self._response}
        return _motion.fit_target("jacobian", _prior, trace, _ridge)

    def _object(self, raw):
        patches = _patch_values(raw) - self._reset_patches
        z = (patches - _phase["mean"]) / _phase["scale"]
        logit = _phase["beta"][0] + z @ _phase["beta"][1:]
        open_probability = np.exp(-np.logaddexp(0.0, -logit))
        fingers = _finite(raw["robot0_gripper_qpos"], (2,))
        return np.r_[open_probability, patches, fingers[0] - fingers[1]]

    def features(self, raw, canonical):
        """Motion (19) and progress (13), in their native units.

        Motion: pending and last issued mean XYZ, predicted source and local TCP
        deltas, last measured delta, last local prediction minus measurement (m),
        and a valid-history bit. Progress: current and previous [instantaneous
        open probability, four reset-relative patch luminances, jaw width (m)]
        and pending mean gripper command. Only the first replan window is scored.
        """
        window = _chunk(canonical)[:5]
        current = window[:, :3].mean(axis=0, dtype=np.float64)
        commands = np.r_[current, self._previous_command]
        motion = np.r_[commands, commands @ _prior, commands @ self._body(raw),
                       self._achieved, self._tracking_error, float(self._observed)]
        progress = np.r_[self._object(raw), self._previous_object, window[:, 6].mean(dtype=np.float64)]
        return motion, progress

    def correct(self, raw, canonical):
        """Match source steady-command response with the source-noise-damped inverse."""
        proposal = _chunk(canonical)
        matrix = _motion.correction_matrix(_prior, self._body(raw), _noise)
        translated = _motion.map_action(proposal, matrix)
        corrected = proposal.astype(np.result_type(proposal.dtype, np.float32), copy=True)
        corrected[:, :3] = translated[:, :3]
        return corrected

    def observe(self, before_raw, issued_array, after_raw):
        """Commit one executed burst. A truncated terminal burst has no five-control response."""
        issued = _chunk(issued_array)
        if len(issued) > 5:
            raise ValueError("Expected the executed replan window, at most five controls")
        previous_object = self._object(before_raw)
        if len(issued) == 5:
            current = issued[:, :3].mean(axis=0, dtype=np.float64)
            before = _finite(before_raw["robot0_eef_pos"], (3,))
            after = _finite(after_raw["robot0_eef_pos"], (3,))
            expected = np.r_[current, self._previous_command] @ self._body(before_raw)
            self._achieved = after - before
            self._tracking_error = expected - self._achieved
            self._previous_command = current
            self._observed = True
        self._previous_object = previous_object

    def cue(self, raw):
        """Drawer-and-bowl subinstruction. A later reclosure can select opening again."""
        if self._object(raw)[0] >= 0.5:
            return "put the bowl in the top drawer"
        return "open the top drawer"
