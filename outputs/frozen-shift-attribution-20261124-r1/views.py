"""Simulator worker that adds a Panda render of the same state to each boundary observation.

Adapted from ../frozen-vla-source-view-20261123-r1/live.py. The raw Kinova3 images are left
unchanged; `panda_agentview_image` and `panda_robot0_eye_in_hand_image` are added. Not executed
on CPU: it needs the pinned simulator runtime.
"""
import json
from multiprocessing import Pipe
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SOURCE_VIEW = HERE.parent / "frozen-vla-source-view-20261123-r1"


def _simulator_base():
    from ur5e_sim_worker import UR5eSimulator
    return UR5eSimulator


def Simulator(log, layouts):
    from ur5e_sim_worker import SIM_PYTHON

    class _Simulator(_simulator_base()):
        def __init__(self):
            self._closed = False
            self.pipe, peer = Pipe()
            try:
                self.worker = subprocess.Popen(
                    [str(SIM_PYTHON), "-Bu", str(SOURCE_VIEW / "entry.py"), str(HERE / "views.py"),
                     str(peer.fileno()), str(log), ",".join(map(str, layouts))],
                    pass_fds=(peer.fileno(),), env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
            finally:
                peer.close()

    return _Simulator()


def serve(fd, log, layouts):
    from unittest.mock import patch

    import numpy as np
    from arm_env import ArmEnv, CAMERAS
    from live import restore_rng
    from source_view import Exposure, Sibling, activate, compose
    from ur5e_sim_worker import serve as native_serve

    class Observed(ArmEnv):
        control = panda = None

        def reset(self, init_state_id, seed):
            self.layout = init_state_id
            expo = Exposure()
            with expo.capture():
                row = super().reset(init_state_id, seed)
            for sibling in (self.control, self.panda):
                if sibling is not None:
                    sibling.close()
            self.control = self.panda = None
            if self.arm != "Kinova3" or init_state_id not in layouts:
                return row  # calibration or another body: no sibling
            with restore_rng():
                self.control = Sibling("Kinova3", init_state_id, seed)
                self.panda = Sibling("Panda", init_state_id, seed)
            for sibling in (self.control, self.panda):
                for field in ("scene_fingerprint", "released_row_fingerprint", "bddl_sha256",
                              "perturbation_plan_fingerprint"):
                    if sibling.initial["environment_reproducibility"][field] != row["environment_reproducibility"][field]:
                        raise ValueError(f"Sibling reset differs in {field}")
            return self.add_view(row, expo)

        def step(self, action):
            if self.panda is None or (self._step + 1) % 5:
                return super().step(action)
            expo = Exposure()
            with expo.capture():
                row = super().step(action)
            return row if row["terminated"] or row["truncated"] else self.add_view(row, expo)

        def add_view(self, row, expo):
            obs = row["observation"]
            exposure, timing = expo.read(self._env, obs)
            before = self._env.get_sim_state().copy()
            trace = dict(layout=self.layout, step=self._step, timing=timing, cameras={})
            try:
                trace["control"] = self.control.sync(self._env, exposure)
                trace["panda"] = self.panda.sync(self._env, exposure)
                updated, valid = dict(obs), bool(trace["panda"]["admissible"])
                for cam in CAMERAS:
                    _, updated["panda_" + cam + "_image"], stats = compose(
                        obs[cam + "_image"], self.control, self.panda, cam)
                    trace["cameras"][cam] = stats
                    valid &= not (stats["raw_vs_full_control"]["changed"] or stats["clean_plate"]["changed"]
                                  or stats["clean_depth_max"] or stats["changed_outside_union"]
                                  or stats["visible_depth_behind_px"])
                if not np.array_equal(before, self._env.get_sim_state()):
                    raise ValueError("Physical state changed while rendering")
                # A boundary without an admissible, checked Panda render keeps the raw observation only;
                # the teacher then records the view conditions as missing there.
                trace["valid"] = valid
                if valid:
                    row["observation"] = updated
                return row
            except Exception as exc:
                trace["invalid"] = str(exc)
                raise
            finally:
                with open(log, "a") as stream:
                    stream.write(json.dumps(trace, allow_nan=False, default=str) + "\n")
                activate(self._env.sim)

        def close(self):
            for sibling in (self.control, self.panda):
                if sibling is not None:
                    sibling.close()
            self.control = self.panda = None
            if self._env is not None:
                activate(self._env.sim)
            super().close()

    with patch("arm_env.ArmEnv", Observed):
        native_serve(fd)


if __name__ == "__main__":
    serve(int(sys.argv[1]), sys.argv[2], [int(item) for item in sys.argv[3].split(",")])
