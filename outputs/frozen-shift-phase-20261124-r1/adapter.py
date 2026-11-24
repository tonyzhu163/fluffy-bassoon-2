"""Hold a drawer subtask prompt for the whole episode on the frozen policy.

Only the prompt string changes. The phase comes from the public agentview image
through the Panda-frozen drawer reader, or from the source-fitted clock.
"""
from importlib.util import module_from_spec, spec_from_file_location
import hashlib
import json
import os
from pathlib import Path

import numpy as np

OUTPUTS = Path(__file__).resolve().parents[1]
_HARNESS = OUTPUTS / "frozen-vla-proposal-shift-20261124-r1/harness.py"
_SIGNALS = OUTPUTS / "frozen-vla-selector-20261124-r1/signals.py"
_CLOCK = OUTPUTS / "frozen-vla-phase-transfer-20261123-r1/evidence/model.json"

OPENING = "open the top drawer"
PLACING = "put the bowl in the top drawer"
TRACE_ENV = "PHASE_TRACE_DIR"  # optional directory for per-episode public reader traces


def _load(name, path):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    from harness import Adapter
except ImportError:  # imported outside the panel
    import sys
    sys.path.append(str(_HARNESS.parent))
    from harness import Adapter

_signals = _load("_phase_lane_signals", _SIGNALS)
drawer_front, drawer_probability = _signals.drawer_front, _signals.drawer_probability
THRESHOLDS = _signals._drawer_thresholds()
CLOCK = np.array(json.loads(_CLOCK.read_bytes())["clock"])


def read(reset_front, image):
    """Drawer-front x (128-pixel units, None if hidden) and the source-ensemble open score."""
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3:
        raise ValueError("Expected the unflipped native RGB uint8 agentview image")
    front = drawer_front(image)
    return front, drawer_probability(reset_front, front, THRESHOLDS)


def instant(placing, reset_front, front, score):
    """`Signals.cue`: strict source majority, a hidden edge reads closed."""
    return score > 0.5


def hysteretic(placing, reset_front, front, score):
    """Advance on a strict source majority. Return only when a visible edge is below
    every source crossing. Hold inside the source band and while the edge is hidden."""
    if reset_front is None or front is None:
        return placing
    if score > 0.5:
        return True
    if score == 0:
        return False
    return placing


def clock_open(control):
    """Source-fitted clock from the phase-transfer model, with its carry-last extrapolation."""
    return bool(CLOCK[min(control // 5, len(CLOCK) - 1)] >= 0.5)


class Phase(Adapter):
    """Subclasses set `name` and `phase(control, front, score) -> placing`."""

    name = None
    episodes = 0  # per adapter class, for trace file names

    def reset(self, support, first):
        image = first["observation"]["agentview_image"]
        self.reset_front, _ = read(None, image)
        self.control = 0
        self.placing = False
        self.seen = None
        self.trace = []
        self.reset_image_sha256 = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
        self.index = type(self).episodes
        type(self).episodes += 1

    def phase(self, control, front, score):
        raise NotImplementedError

    def prompt(self, raw, original):
        if self.seen != self.control:  # one transition per replan even if queried twice
            front, score = read(self.reset_front, raw["agentview_image"])
            self.placing = bool(self.phase(self.control, front, score))
            self.seen = self.control
            self.trace.append(dict(control=self.control, front_px=front, reset_front_px=self.reset_front,
                                   score=score, prompt=PLACING if self.placing else OPENING))
            self._save()
        return PLACING if self.placing else OPENING

    def observe(self, before, issued, after):
        self.control += len(issued)

    def _save(self):
        folder = os.environ.get(TRACE_ENV)
        if folder:
            Path(folder).mkdir(parents=True, exist_ok=True)
            body = dict(adapter=self.name, episode=self.index, reset_image_sha256=self.reset_image_sha256,
                        replans=self.trace)
            Path(folder, f"{self.name}-{self.index:03d}.json").write_text(json.dumps(body) + "\n")


class Reader(Phase):
    name = "phase-reader"

    def phase(self, control, front, score):
        return hysteretic(self.placing, self.reset_front, front, score)


class Instant(Phase):
    name = "phase-instant"

    def phase(self, control, front, score):
        return instant(self.placing, self.reset_front, front, score)


class Clock(Phase):
    name = "phase-clock"

    def phase(self, control, front, score):
        return clock_open(control)


class Open(Phase):
    name = "phase-open"

    def phase(self, control, front, score):
        return False


class Place(Phase):
    name = "phase-place"

    def phase(self, control, front, score):
        return True


ADAPTERS = [Reader, Instant, Clock, Open, Place]
