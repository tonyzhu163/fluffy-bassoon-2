"""Render twin for the body-view adapters. Runs in the simulator runtime; privileged.

`Session` is renderer-independent and CPU-tested. `Scene` is the MuJoCo backend built on
the natively qualified source-view siblings; it needs robosuite and is not run off cluster.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random
import sys
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
CAMERAS = ("agentview", "robot0_eye_in_hand")


def digest(public, keys=None):
    """Value digest of a public observation, independent of float width."""
    h = hashlib.sha256()
    for key in sorted(public if keys is None else keys):
        value = np.asarray(public[key])
        value = value if value.dtype == np.uint8 else value.astype(np.float64)
        h.update(key.encode() + str(value.shape).encode() + np.ascontiguousarray(value).tobytes())
    return h.hexdigest()


class Session:
    """Keep a second environment on the public trajectory of the episode the harness is running."""

    def __init__(self, backend):
        self.backend, self.known, self.live = backend, {}, False

    def start(self, public, candidates):
        self.live = False
        candidates = [tuple(c) for c in candidates]
        want, keys = digest(public), sorted(public)
        hit = [c for c in candidates if self.known.get(c) == want]
        # Later branches of a start hit the cache; a new start costs one reset per unseen earlier candidate.
        tries = hit[:1] or [c for c in candidates if c not in self.known]
        for count, start in enumerate(tries, 1):
            self.obs = self.backend.reset(*start)
            self.known[start] = digest(self.obs, keys)
            if self.known[start] == want:
                self.live, self.controls = True, 0
                return dict(resets=count)
        raise ValueError("No registered start reproduces the public reset observation")

    def advance(self, issued):
        if not self.live:
            raise ValueError("Advance without a matched start")
        issued = np.asarray(issued, dtype=np.float32)
        self.obs = self.backend.step(issued)
        self.controls += len(issued)
        return dict(controls=self.controls)

    def view(self, public, mode, cameras):
        if not self.live:
            raise ValueError("View without a matched start")
        if digest(public) != digest(self.obs, sorted(public)):
            self.live = False
            raise ValueError(f"Twin left the public trajectory by control {self.controls}")
        images, stats = self.backend.render(mode, tuple(cameras))
        return dict(images=images, stats=dict(stats, control=self.controls, mode=mode))


@contextmanager
def restore_rng():
    state, py_state = np.random.get_state(), random.getstate()
    try:
        yield
    finally:
        np.random.set_state(state)
        random.setstate(py_state)


class Scene:
    """Kinova3 drawer twin with Kinova and Panda render siblings (source_view, live.py conventions)."""

    def __init__(self, log=None):
        self.log, self.env, self.siblings = log, None, {}

    def close(self):
        from source_view import activate
        for sibling in self.siblings.values():
            sibling.close()
        self.siblings = {}
        if self.env is not None:
            activate(self.env._env.sim)
            self.env.close()
            self.env = None

    def reset(self, layout, seed):
        from arm_env import ArmEnv
        from source_view import Exposure
        self.close()
        self.layout, self.seed = int(layout), int(seed)
        self.env = ArmEnv(arm="Kinova3", suite="libero_goal", task_id=3, case_id="BASE")
        self.expo = Exposure()
        with self.expo.capture():
            self.row = self.env.reset(init_state_id=self.layout, seed=self.seed)
        return self.row["observation"]

    def step(self, issued):
        from source_view import Exposure, activate
        activate(self.env._env.sim)
        for action in issued[:-1]:
            self.env.step(action)
        # The exposure the policy sees is the one rendered by the last control of the window.
        self.expo = Exposure()
        with self.expo.capture():
            self.row = self.env.step(issued[-1])
        return self.row["observation"]

    def sibling(self, arm):
        from source_view import Sibling
        if arm not in self.siblings:
            with restore_rng():
                made = Sibling(arm, self.layout, self.seed)
            origin, mine = self.row_origin(), made.initial["environment_reproducibility"]
            for field in ("scene_fingerprint", "released_row_fingerprint", "bddl_sha256", "perturbation_plan_fingerprint"):
                if mine[field] != origin[field]:
                    raise ValueError(f"Render sibling differs in {field}")
            self.siblings[arm] = made
        return self.siblings[arm]

    def row_origin(self):
        if "environment_reproducibility" in self.row:
            self.origin = self.row["environment_reproducibility"]
        return self.origin

    def render(self, mode, cameras):
        from source_view import activate, compose, summarize_diff
        native, obs = self.env._env, self.row["observation"]
        self.row_origin()
        ed, timing = self.expo.read(native, obs)
        before = native.get_sim_state().copy()
        control = self.sibling("Kinova3")
        trace = dict(layout=self.layout, step=int(self.env._step), mode=mode, cameras_edited=list(cameras),
                     timing=timing, control=control.sync(native, ed), cameras={})
        images = {}
        try:
            panda = None
            if mode == "panda":
                panda = self.sibling("Panda")
                trace["panda"] = panda.sync(native, ed)
                if not trace["panda"]["admissible"]:
                    raise ValueError("Inadmissible Panda pose or arm collision")
            for cam in cameras:
                raw = obs[cam + "_image"]
                if panda is None:
                    full, _, mask = control.render(cam)
                    image = raw.copy()
                    image[mask] = full[mask]
                    stats = dict(raw_vs_full_control=summarize_diff(raw, full))
                else:
                    _, image, stats = compose(raw, control, panda, cam)
                stats["changed"] = int(np.any(raw != image, axis=-1).sum())
                trace["cameras"][cam] = stats
                if stats["raw_vs_full_control"]["changed"]:
                    raise ValueError("Kinova full render differs from target exposure")
                if panda is not None and (stats["clean_plate"]["changed"] or stats["clean_depth_max"]
                                          or stats["changed_outside_union"] or stats["visible_depth_behind_px"]):
                    raise ValueError("Scene-only RGB/depth or arm depth visibility failed")
                images[cam] = image
            if not np.array_equal(before, native.get_sim_state()):
                raise ValueError("Twin state changed while rendering")
        except Exception as exc:
            trace["invalid"] = str(exc)
            raise
        finally:
            if self.log is not None:
                with open(self.log, "a") as f:
                    f.write(json.dumps(trace, allow_nan=False) + "\n")
            activate(native.sim)
        return images, {cam: {k: v for k, v in s.items() if isinstance(v, int)} for cam, s in trace["cameras"].items()}


def serve(fd, log):
    from multiprocessing.connection import Connection
    import os
    from ur5e_sim_worker import _probe_egl

    _probe_egl(os.environ.get("FLUFFY_ALLOW_MESA") == "1")
    session = Session(Scene(log or None))
    with Connection(fd) as pipe:
        while True:
            try:
                op, value = pipe.recv()
            except EOFError:
                break
            try:
                if op == "start":
                    result = session.start(value["observation"], value["candidates"])
                elif op == "advance":
                    result = session.advance(value)
                elif op == "view":
                    result = session.view(value["observation"], value["mode"], value["cameras"])
                else:
                    raise ValueError(op)
                pipe.send((True, result))
            except Exception:
                session.live = False
                pipe.send((False, traceback.format_exc()))
    session.backend.close()


if __name__ == "__main__":
    sys.path[:0] = [str(HERE.parent / "frozen-vla-source-view-20261123-r1"),
                    str(HERE.parent / "frozen-vla-reachability-20261123-r1")]
    serve(int(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "")
