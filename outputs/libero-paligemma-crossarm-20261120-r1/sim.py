"""Goal 5 BASE session shared by the collector and the VLM interpreter."""

from functools import lru_cache
from itertools import product
from multiprocessing import Pipe
from multiprocessing.connection import Connection
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

import numpy as np


BODIES = ("Panda", "Sawyer", "UR5e", "IIWA")
PRIOR = Path(__file__).resolve().parents[1] / "libero-crossarm-20261120-r1"
IIWA_XML_SHA = "7eb9d87f431f7fbfcb0e028e608956aa2c2f7dc456abb0dfb6f234fd8044a5dd"
_BOX_SIGNS = np.array(list(product((-1, 1), repeat=3)))


@lru_cache(maxsize=1)
def _arm_class():
    # Import only in the simulator interpreter. Share the prior module's registration
    # cache across direct Sessions, including an existing instance in this process.
    sys.path.insert(0, str(PRIOR))
    try:
        import arm_env
    finally:
        sys.path.remove(str(PRIOR))
    if Path(arm_env.__file__).resolve() != PRIOR / "arm_env.py":
        raise RuntimeError("Unexpected cross-arm simulator module")
    if arm_env.ARM_XML_SHA.setdefault("IIWA", IIWA_XML_SHA) != IIWA_XML_SHA:
        raise RuntimeError("Conflicting IIWA model identity")
    return arm_env.ArmEnv


def _public(packet):
    raw = packet["observation"]
    fingers = raw["robot0_gripper_qpos"]
    state = np.r_[raw["robot0_eef_pos"], raw["robot0_eef_quat_xyzw"], fingers[0] - fingers[1]]
    return dict(rgb=raw["agentview_image"], wrist=raw["robot0_eye_in_hand_image"],
                state=state.astype(np.float32), prompt=packet["language_instruction"])


class _PushGeometry:
    """Read-only views of the compiled BASE boxes, physical fingers and goal site."""

    def __init__(self, wrapper):
        import mujoco

        scene = wrapper.env
        goal, = scene.parsed_problem["goal_state"]
        if [str(value).lower() for value in goal] != ["on", "plate_1", "main_table_stove_front_region"]:
            raise ValueError(f"Unexpected BASE goal {goal!r}")
        self.model, self.data = wrapper.sim.model, wrapper.sim.data
        self.plate_id = scene.obj_body_id["plate_1"]
        self.goal_id = self.model.site_name2id(goal[2])
        self.table_id = self.model.site_name2id("table_top")
        robot = wrapper.robots[0]
        self.tcp_id = robot.eef_site_id
        fingers = robot.gripper.important_geoms
        groups = dict(plate=scene.objects_dict["plate_1"].contact_geoms,
                      obstacle=scene.objects_dict["cream_cheese_1"].contact_geoms,
                      fingers=fingers["left_finger"] + fingers["right_finger"])
        self.vertices = {}
        self.ids = {}
        for key, names in groups.items():
            ids = [self.model.geom_name2id(name) for name in names]
            if not ids:
                raise ValueError(f"Missing compiled collision geometry for {key}")
            self.ids[key] = set(ids)
            blocks = []
            for gid in ids:
                kind = self.model.geom_type[gid]
                if kind == mujoco.mjtGeom.mjGEOM_BOX:
                    local = _BOX_SIGNS * self.model.geom_size[gid]
                elif kind == mujoco.mjtGeom.mjGEOM_MESH:
                    mesh = self.model.geom_dataid[gid]
                    start = self.model.mesh_vertadr[mesh]
                    local = self.model.mesh_vert[start:start + self.model.mesh_vertnum[mesh]]
                else:
                    raise ValueError(f"Unexpected {key} collision geom type {kind}")
                blocks.append((gid, local))
            self.vertices[key] = blocks

    def _points(self, group):
        return np.concatenate([points @ self.data.geom_xmat[gid].reshape(3, 3).T + self.data.geom_xpos[gid]
                               for gid, points in self.vertices[group]])

    def read(self):
        plate = self._points("plate")
        obstacle = self._points("obstacle")
        fingers = self._points("fingers")
        center = self.data.body_xpos[self.plate_id]
        contacts = [(c.geom1, c.geom2) for c in self.data.contact[:self.data.ncon]]
        touching = any((a in self.ids["plate"] and b in self.ids["fingers"]) or
                       (b in self.ids["plate"] and a in self.ids["fingers"]) for a, b in contacts)
        return dict(plate=center.tolist(), plate_radius=float(np.linalg.norm(plate[:, :2] - center[:2], axis=1).max()),
                    rim_z=float(plate[:, 2].max()), obstacle_max=obstacle.max(axis=0).tolist(),
                    goal=self.data.site_xpos[self.goal_id].tolist(),
                    goal_half=self.model.site_size[self.goal_id].tolist(),
                    table_z=float(self.data.site_xpos[self.table_id, 2]),
                    finger_low=float(fingers[:, 2].min()), contact=touching,
                    tcp=self.data.site_xpos[self.tcp_id].tolist(),
                    tool_z=self.data.site_xmat[self.tcp_id].reshape(3, 3)[:, 2].tolist())


class _Native:
    def __init__(self, body, task_id):
        self.arm = _arm_class()(arm=body, suite="libero_goal", task_id=task_id)
        self.geometry = None

    def reset(self, pose, seed):
        self.geometry = None
        packet = self.arm.reset(init_state_id=pose, seed=seed)
        info = {k: packet[k] for k in ("model_identity", "environment_reproducibility", "max_steps")}
        return _public(packet), info

    def step(self, action):
        p = self.arm.step(action)
        return _public(p), p["sensor_motion"], p["reward"], p["terminated"] or p["truncated"], p["success"]

    def expert_scene(self):
        if self.geometry is None:
            self.geometry = _PushGeometry(self.arm._env)
        return self.geometry.read()

    def close(self):
        self.geometry = None
        self.arm.close()


# Pipe payloads use builtins. The two environments may have incompatible NumPy
# pickle module names. RGB is transmitted uncompressed without a text encoding.
def _pack(obs):
    return dict(rgb=obs["rgb"].tobytes(), wrist=obs["wrist"].tobytes(), shape=obs["rgb"].shape,
                state=obs["state"].tolist(), prompt=obs["prompt"])


def _unpack(obs):
    def camera(key):
        return np.frombuffer(obs[key], np.uint8).reshape(obs["shape"]).copy()
    return dict(rgb=camera("rgb"), wrist=camera("wrist"), state=np.array(obs["state"], np.float32),
                prompt=obs["prompt"])


def _serve(fd, body, task_id):
    with Connection(fd) as channel:
        native = None
        try:
            while True:
                op, value = channel.recv()
                if op == "close":
                    break
                try:
                    if native is None:
                        native = _Native(body, task_id)
                    if op == "reset":
                        obs, info = native.reset(*value)
                        result = _pack(obs), info
                    elif op == "step":
                        obs, motion, reward, done, success = native.step(value)
                        result = _pack(obs), motion.tolist(), reward, done, success
                    elif op == "expert_scene":
                        result = native.expert_scene()
                    else:
                        raise ValueError(f"Unknown operation {op!r}")
                except Exception:
                    channel.send((False, traceback.format_exc()))
                else:
                    channel.send((True, result))
        except (BrokenPipeError, EOFError):
            pass
        finally:
            if native is not None:
                native.close()


class Session:
    """Relative 20 Hz OSC session. reset_info holds audit metadata outside the observation."""

    def __init__(self, body, task_id=5):
        if body not in BODIES or task_id != 5:
            raise ValueError(f"Expected one of {BODIES} and BASE libero_goal/5")
        self.native = self.pipe = self.worker = self.reset_info = None
        self.closed = False
        python = os.environ.get("CROSSARM_SIM_PYTHON")
        # Preserve the venv path: different environments can symlink the same binary.
        if not python or os.path.abspath(shutil.which(python) or python) == os.path.abspath(sys.executable):
            self.native = _Native(body, task_id)
        else:
            self.pipe, peer = Pipe()
            try:
                self.worker = subprocess.Popen([python, "-Bu", str(Path(__file__).resolve()),
                                                str(peer.fileno()), body, str(task_id)], pass_fds=(peer.fileno(),))
            except BaseException:
                self.pipe.close()
                raise
            finally:
                peer.close()

    def _call(self, op, *args):
        if self.closed:
            raise RuntimeError("Session is closed")
        if self.native is not None:
            return getattr(self.native, op)(*args)
        value = args[0] if op == "step" else args
        self.pipe.send((op, value))
        ok, result = self.pipe.recv()
        if not ok:
            raise RuntimeError(result)
        return result

    def reset(self, pose, seed):
        self.reset_info = None
        obs, self.reset_info = self._call("reset", int(pose), int(seed))
        return obs if self.native is not None else _unpack(obs)

    def step(self, action):
        action = np.asarray(action)
        if action.shape != (7,) or action.dtype.kind not in "fiu" or not np.isfinite(action).all() or np.any(abs(action.astype(np.float64)) > 1):
            raise ValueError("Expected seven finite native actions in [-1, 1]")
        obs, motion, reward, done, success = self._call("step", action.astype(np.float32).tolist())
        if self.native is None:
            obs = _unpack(obs)
        return obs, np.asarray(motion, dtype=np.float32), float(reward), bool(done), bool(success)

    def expert_scene(self):
        """Privileged coordinates for demonstration generation only."""
        return self._call("expert_scene")

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.native is not None:
            self.native.close()
        else:
            try:
                self.pipe.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
            finally:
                self.pipe.close()
            try:
                self.worker.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.worker.kill()
                self.worker.wait()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


if __name__ == "__main__":
    _serve(int(sys.argv[1]), sys.argv[2], int(sys.argv[3]))
