"""Standard LIBERO in its qualified robosuite interpreter, on the job's EGL device."""
from multiprocessing import Pipe
from multiprocessing.connection import Connection
import hashlib
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback

import numpy as np


# examples/libero/main.py at OpenPI 215abfb. The 400 controls start AFTER settling.
SUITE = "libero_90"
HORIZON, SETTLE, RESOLUTION, NATIVE_CADENCE = 400, 10, 256, 5
CAMERAS = ("agentview", "robot0_eye_in_hand")
PUBLIC_SHAPES = {
    "agentview_image": (RESOLUTION, RESOLUTION, 3),
    "robot0_eye_in_hand_image": (RESOLUTION, RESOLUTION, 3),
    "robot0_eef_pos": (3,), "robot0_eef_quat": (4,), "robot0_gripper_qpos": (2,),
}


def fingerprint(array):
    value = np.ascontiguousarray(array)
    return hashlib.sha256(str((value.shape, value.dtype.str)).encode() + value.tobytes()).hexdigest()


def public_observation(raw):
    public = {key: np.asarray(raw[key]).copy() for key in PUBLIC_SHAPES}
    for key, value in public.items():
        valid = value.dtype == np.uint8 if key.endswith("_image") else np.isfinite(value).all()
        if value.shape != PUBLIC_SHAPES[key] or not valid:
            raise ValueError(f"Invalid public LIBERO channel {key}: {value.shape}, {value.dtype}")
    return public


class Simulator:
    """Keep standard LIBERO and OpenPI in separate, already installed environments."""

    def __enter__(self):
        self.pipe, peer = Pipe()
        try:
            self.proc = subprocess.Popen(
                [os.environ["LIBERO_SIM_PYTHON"], "-Bu", __file__, str(peer.fileno())],
                env=dict(os.environ, PYTHONPATH=os.environ.get("LIBERO_SIM_PYTHONPATH", os.environ["LIBERO_STD_ROOT"])),
                pass_fds=(peer.fileno(),),
            )
        except BaseException:
            self.pipe.close()
            raise
        finally:
            peer.close()
        return self

    def call(self, command, value):
        self.pipe.send((command, value))
        ok, result = self.pipe.recv()
        if not ok:
            raise RuntimeError(f"Standard LIBERO {command} failed:\n{result}")
        return result

    def __exit__(self, *exc):
        self.pipe.close()
        try:
            self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()


class StandardLibero:
    def __init__(self):
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        base = Path(os.environ["LIBERO_STD_ROOT"]).resolve() / "libero/libero"
        if Path(benchmark.__file__).resolve().parent != base / "benchmark":
            raise RuntimeError(f"Unexpected LIBERO import: {benchmark.__file__}")
        for key, name in (("assets", "assets"), ("bddl_files", "bddl_files"), ("init_states", "init_files")):
            if Path(get_libero_path(key)).resolve() != (base / name).resolve():
                raise RuntimeError(f"Unexpected standard LIBERO {key} root")
        self.suite = benchmark.get_benchmark_dict()[SUITE]()
        if self.suite.n_tasks != 90:
            raise RuntimeError(f"Standard LIBERO-90 has {self.suite.n_tasks} tasks in this runtime")
        self.env_type = OffScreenRenderEnv
        self.env = None
        self.task_id = None
        self.renderer = None

    def reset(self, case):
        task_id, state_id, seed = (case[key] for key in ("task_id", "init_state_id", "seed"))
        if self.task_id != task_id:
            self.states = self.suite.get_task_init_states(task_id)
            self.task = self.suite.get_task(task_id)
            self.bddl = Path(self.suite.get_task_bddl_file_path(task_id))
            self.bddl_sha = hashlib.sha256(self.bddl.read_bytes()).hexdigest()
            self.task_id = task_id
        if not 0 <= state_id < len(self.states):
            raise ValueError(f"Missing standard reset {state_id} for task {task_id}")
        state = np.asarray(self.states[state_id])
        if self.env is not None:
            self.env.close()
            self.env = None
        # The qualified standard runtime reconstructs fixtures on every seeded reset.
        random.seed(seed)
        np.random.seed(seed)
        self.env = self.env_type(bddl_file_name=str(self.bddl), camera_names=list(CAMERAS),
                                 camera_widths=RESOLUTION, camera_heights=RESOLUTION, render_gpu_device_id=0)
        self.env.seed(seed)
        self.env.reset()
        self.env.set_init_state(state)
        for _ in range(SETTLE):
            raw, _, _, _ = self.env.step([0.] * 6 + [-1.])
        if self.renderer is None:
            from OpenGL import GL

            self.env.sim._render_context_offscreen.gl_ctx.make_current()
            vendor, renderer = GL.glGetString(GL.GL_VENDOR), GL.glGetString(GL.GL_RENDERER)
            if not vendor or b"nvidia" not in vendor.lower() or not renderer:
                raise RuntimeError(f"Expected the allocated NVIDIA EGL renderer, found {vendor!r}, {renderer!r}")
            self.renderer = renderer.decode()
        self.count, self.done = 0, False
        obs = public_observation(raw)
        identity = dict(
            task_name=self.task.name, prompt=self.task.language, horizon=HORIZON, renderer=self.renderer,
            bank_size=len(self.states), bddl_sha256=self.bddl_sha, bank_state_sha256=fingerprint(state),
            physics=fingerprint(self.env.get_sim_state()), public={k: fingerprint(v) for k, v in obs.items()},
        )
        return obs, identity

    def step(self, commands):
        if self.env is None or self.done or not 0 < len(commands) <= HORIZON - self.count:
            raise ValueError("Attempt to step a closed episode or exceed the LIBERO-90 horizon")
        positions, quats, gripper, rewards = [], [], [], []
        for action in commands:
            raw, reward, success, _ = self.env.step(action.tolist())
            if not np.isfinite(reward) or bool(reward > 0) != bool(success):
                raise ValueError("Native sparse reward and LIBERO terminal success disagree")
            positions.append(raw["robot0_eef_pos"].copy())
            quats.append(raw["robot0_eef_quat"].copy())
            gripper.append(raw["robot0_gripper_qpos"].copy())
            rewards.append(float(reward))
            self.count += 1
            if success:
                break
        self.done = bool(success) or self.count == HORIZON
        return dict(observation=public_observation(raw), positions=positions, quats=quats, gripper=gripper,
                    rewards=rewards, success=bool(success), done=self.done)


def serve(fd):
    with Connection(fd) as pipe:
        sim = None
        try:
            sim = StandardLibero()
            while True:
                command, value = pipe.recv()
                if command == "reset":
                    result = sim.reset(value)
                elif command == "step":
                    result = sim.step(value)
                else:
                    raise ValueError(command)
                pipe.send((True, result))
        except EOFError:
            pass
        except Exception:
            pipe.send((False, traceback.format_exc()))
        finally:
            if sim is not None and sim.env is not None:
                sim.env.close()


if __name__ == "__main__":
    serve(int(sys.argv[1]))
