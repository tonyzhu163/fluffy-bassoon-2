"""Body-view adapters: picture the source arm at the target grip site at every replan.

The images come from a privileged render twin (twin.py). The adapter gives it the public
reset observation, the issued commands and each public observation to verify against.
"""
import atexit
from multiprocessing import Pipe
import os
from pathlib import Path
import subprocess

import numpy as np

from harness import Adapter

HERE = Path(__file__).resolve().parent
CAMERAS = ("agentview", "robot0_eye_in_hand")
SEED = 42710300
# Registered Kinova3 starts of the program panel: development, then locked.
LAYOUTS = (14, 15, 17, 36, 37, 38, 16, 31, 34, 35, 39, *range(40, 48))
CANDIDATES = [(layout, SEED + layout) for layout in LAYOUTS]


class Twin:
    """Client of the render twin process, started on first use."""

    def __init__(self):
        self.pipe = self.worker = None

    def call(self, op, value):
        if self.pipe is None:
            from ur5e_sim_worker import SIM_PYTHON
            self.pipe, peer = Pipe()
            trace = os.environ.get("VIEW_TRACE", "")
            try:
                self.worker = subprocess.Popen(
                    [str(SIM_PYTHON), "-Bu", str(HERE / "twin.py"), str(peer.fileno()), trace],
                    pass_fds=(peer.fileno(),),
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": "",
                         "FLUFFY_ALLOW_MESA": os.environ.get("VIEW_ALLOW_MESA", "0")})
            finally:
                peer.close()
            atexit.register(self.close)
        self.pipe.send((op, value))
        ok, result = self.pipe.recv()
        if not ok:
            raise RuntimeError(result)
        return result

    def close(self):
        if self.pipe is not None:
            self.pipe.close()
            self.pipe = None
            try:
                self.worker.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.worker.terminate()
                self.worker.wait()


TWIN = Twin()


class View(Adapter):
    mode = "control"
    cameras = CAMERAS

    def __init__(self, twin=None):
        self.twin = TWIN if twin is None else twin
        self.active, self.stats = False, []

    def reset(self, support, first):
        arm = first["arm"]
        if arm not in ("Kinova3", "Panda"):
            raise ValueError(f"Body view is registered for Kinova3 targets, not {arm}")
        # The source arm is already in view on a Panda start.
        self.active, self.stats = arm == "Kinova3", []
        if self.active:
            self.twin.call("start", dict(observation=first["observation"], candidates=CANDIDATES))

    def observation(self, raw):
        if not self.active:
            return raw
        reply = self.twin.call("view", dict(observation=raw, mode=self.mode, cameras=self.cameras))
        seen = dict(raw)
        for cam in self.cameras:
            key = cam + "_image"
            image = np.asarray(reply["images"][cam])
            if image.shape != raw[key].shape or image.dtype != raw[key].dtype:
                raise ValueError(f"Rendered {cam} does not match the native image format")
            seen[key] = image
        self.stats.append(reply["stats"])
        return seen

    def observe(self, before, issued, after):
        if self.active:
            self.twin.call("advance", issued)


class Kinova(View):
    name = "view_kinova"


class PandaExternal(View):
    name = "view_panda_external"
    mode, cameras = "panda", CAMERAS[:1]


class PandaBoth(View):
    name = "view_panda_both"
    mode = "panda"


ADAPTERS = [Kinova(), PandaExternal(), PandaBoth()]
