"""Native target feedback, with render siblings evaluated only at actual policy boundaries."""
from contextlib import contextmanager
import json
from multiprocessing import Pipe
import os
from pathlib import Path
import random
import subprocess
import sys
from unittest.mock import patch

import numpy as np
from ur5e_sim_worker import SIM_PYTHON, UR5eSimulator

HERE = Path(__file__).resolve().parent


class Simulator(UR5eSimulator):
    def __init__(self, mode, log, *, mesa=False):
        if mode not in ("control","panda"):
            raise ValueError(mode)
        self._closed = False
        self.pipe, peer = Pipe()
        try:
            self.worker = subprocess.Popen([str(SIM_PYTHON), "-Bu", str(HERE/"entry.py"), str(HERE/"live.py"),
                str(peer.fileno()), mode, str(log)], pass_fds=(peer.fileno(),),
                env={**os.environ,"FLUFFY_ALLOW_MESA":str(int(mesa)), "CUDA_VISIBLE_DEVICES":""})
        finally:
            peer.close()


@contextmanager
def restore_rng():
    state, py_state = np.random.get_state(), random.getstate()
    try:
        yield
    finally:
        np.random.set_state(state)
        random.setstate(py_state)


def serve(fd, mode, log):
    from arm_env import ArmEnv, CAMERAS
    from source_view import Exposure, Sibling, activate, compose, summarize_diff
    from ur5e_sim_worker import serve as native_serve

    def record(row):
        with log.open("a") as f:
            f.write(json.dumps(row,allow_nan=False)+"\n")

    class Observed(ArmEnv):
        control = panda = None

        def reset(self, layout, seed):
            self.layout = layout
            expo = Exposure()
            with expo.capture():
                row = super().reset(layout,seed)
            if (self.arm,self.suite,self.task_id,layout) not in [
                    ("Kinova3","libero_goal",3,i) for i in (16,17)]:
                raise ValueError("Unexpected score case")
            with restore_rng():
                self.control = Sibling("Kinova3", layout, seed)
                if mode == "panda":
                    self.panda = Sibling("Panda", layout, seed)
            for sibling in (self.control,self.panda):
                if sibling is not None:
                    for field in ("scene_fingerprint","released_row_fingerprint","bddl_sha256","perturbation_plan_fingerprint"):
                        if sibling.initial["environment_reproducibility"][field] != row["environment_reproducibility"][field]:
                            raise ValueError(f"Score reset differs in {field}")
            return self.apply_view(row,expo)

        def step(self, action):
            if (self._step+1) % 5:
                return super().step(action)
            expo = Exposure()
            with expo.capture():
                row = super().step(action)
            return row if row["terminated"] or row["truncated"] else self.apply_view(row,expo)

        def apply_view(self, row, expo):
            obs = row["observation"]
            ed, timing = expo.read(self._env,obs)
            pre = self._env.get_sim_state().copy()
            trace = dict(layout=self.layout, step=self._step, mode=mode, timing=timing,
                         control=self.control.sync(self._env,ed), cameras={})
            updated = dict(obs)
            try:
                if self.panda is not None:
                    trace["panda"] = self.panda.sync(self._env,ed)
                    if not trace["panda"]["admissible"]:
                        raise ValueError("Inadmissible Panda pose or arm collision")
                for cam in CAMERAS:
                    key = cam+"_image"
                    raw = obs[key]
                    if self.panda is None:
                        full, _, mask = self.control.render(cam)
                        sham = raw.copy()
                        sham[mask] = full[mask]
                        stats = dict(raw_vs_full_control=summarize_diff(raw,full))
                        updated[key] = sham
                    else:
                        _, updated[key], stats = compose(raw,self.control,self.panda,cam)
                    trace["cameras"][cam] = stats
                    if stats["raw_vs_full_control"]["changed"]:
                        raise ValueError("Kinova full render differs from target exposure")
                    if self.panda is not None and (stats["clean_plate"]["changed"] or stats["clean_depth_max"]
                                                   or stats["changed_outside_union"] or stats["visible_depth_behind_px"]):
                        raise ValueError("Scene-only RGB/depth or arm depth visibility failed")
                if not np.array_equal(pre,self._env.get_sim_state()):
                    raise ValueError("Physical state changed while rendering")
                row["observation"] = updated
                return row
            except Exception as exc:
                trace["invalid"] = str(exc)
                raise
            finally:
                record(trace)
                activate(self._env.sim)

        def close(self):
            for sibling in (self.control,self.panda):
                if sibling is not None:
                    sibling.close()
            self.control = self.panda = None
            if self._env is not None:
                activate(self._env.sim)
            super().close()

    with patch("arm_env.ArmEnv", Observed):
        native_serve(fd)


if __name__ == "__main__":
    serve(int(sys.argv[1]), sys.argv[2], Path(sys.argv[3]))
