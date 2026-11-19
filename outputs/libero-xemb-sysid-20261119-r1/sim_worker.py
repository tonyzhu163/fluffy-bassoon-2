"""Persistent LIBERO-Pro2 subprocess simulator worker for cross-embodiment evaluation."""

from multiprocessing import Pipe
from multiprocessing.connection import Connection
from pathlib import Path
import os
import subprocess
import sys
import traceback

import numpy as np

from fluffy_bassoon.xemb.embodiment import (
    apply_embodiment_action,
    downsample_agentview,
    get_embodiment_spec,
    to_local_proprio,
)

ROOT = Path("/mowenjie/fluffy-bassoon-2")
PRO = ROOT / "sources/libero-pro2-e5500c69"
SIM_PYTHON = ROOT / "setups/libero-meta-learning-20261117-r1/env/bin/python"


class XEmbSimulator:
    """Client handle wrapping the isolated robosuite/LIBERO simulator subprocess."""

    def __init__(self, *, allow_mesa: bool = False):
        self._closed = False
        self.pipe, peer = Pipe()
        env = dict(os.environ)
        if allow_mesa:
            env["FLUFFY_ALLOW_MESA"] = "1"
        try:
            self.worker = subprocess.Popen(
                [str(SIM_PYTHON), "-Bu", str(Path(__file__).resolve()), str(peer.fileno())],
                pass_fds=(peer.fileno(),),
                env=env,
            )
        finally:
            peer.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def call(self, op: str, value=None):
        self.pipe.send((op, value))
        ok, result = self.pipe.recv()
        if not ok:
            raise RuntimeError(result)
        return result

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.pipe.close()
        try:
            self.worker.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.worker.terminate()
            self.worker.wait()


def _inspect_robots():
    import robosuite.robots as rs_robots
    import robosuite.models.grippers as rs_grippers

    robot_names = sorted(getattr(rs_robots, "ROBOT_CLASS_MAPPING", {}).keys())
    gripper_names = sorted(getattr(rs_grippers, "ALL_GRIPPERS", []))
    return dict(robots=robot_names, grippers=gripper_names)


def serve(fd: int):
    from OpenGL import GL
    from libero_remote.environment.libero import LiberoEnvironmentAdapter
    from libero_remote.environment.task_manifest import TaskManifest

    allow_mesa = os.environ.get("FLUFFY_ALLOW_MESA") == "1"
    env = LiberoEnvironmentAdapter()
    manifest = None
    spec = None
    anchor_eef_pos = None
    low = None
    high = None
    renderer_checked = False

    with Connection(fd) as pipe:
        try:
            while True:
                op, value = pipe.recv()
                if op == "inspect_robots":
                    result = _inspect_robots()
                elif op == "create":
                    env.close()
                    if manifest is None:
                        manifest = TaskManifest.load(str(PRO / "assets/task_manifest.json"))
                    variant = value.get("variant", "BASE")
                    key = value["suite"], int(value["task_id"]), variant
                    spec_name = value.get("embodiment", "nom")
                    spec = get_embodiment_spec(spec_name)
                    task_entry = manifest.require(*key)
                    max_steps = int(value.get("max_steps", task_entry["max_steps"]))
                    create_spec = dict(
                        suite=key[0],
                        task_id=key[1],
                        variant=variant,
                        init_state_id=int(value["init_state_id"]),
                        seed=int(value["seed"]),
                        camera_names=["agentview", "robot0_eye_in_hand"],
                        camera_height=256,
                        camera_width=256,
                        control_mode="relative",
                        max_steps=max_steps,
                        _bddl_path=str(manifest.resolve_bddl(*key)),
                    )
                    initial = env.create(create_spec)
                    if not renderer_checked:
                        vendor = GL.glGetString(GL.GL_VENDOR)
                        renderer = GL.glGetString(GL.GL_RENDERER)
                        if not vendor or not renderer:
                            raise RuntimeError(f"Missing OpenGL context: {vendor!r} / {renderer!r}")
                        if not allow_mesa and b"nvidia" not in vendor.lower():
                            raise RuntimeError(f"Expected NVIDIA EGL renderer on GPU, got {vendor!r} / {renderer!r}")
                        renderer_checked = True
                    raw = initial["observation"]
                    anchor_eef_pos = np.asarray(raw["robot0_eef_pos"], dtype=np.float32).copy()
                    bounds = initial["action_schema"]
                    low = np.asarray(bounds["low"], dtype=np.float32)
                    high = np.asarray(bounds["high"], dtype=np.float32)
                    local_raw = to_local_proprio(raw, anchor_eef_pos, spec)
                    initial_out = dict(
                        initial,
                        observation=local_raw,
                        agentview_64=downsample_agentview(raw["agentview_image"], 64),
                        anchor_eef_pos=anchor_eef_pos.copy(),
                        renderer=GL.glGetString(GL.GL_RENDERER).decode(),
                    )
                    result = create_spec, initial_out
                elif op == "chunk":
                    if spec is None or anchor_eef_pos is None:
                        raise RuntimeError("Call create before chunk")
                    actions = np.asarray(value, dtype=np.float32)
                    if actions.ndim != 2 or actions.shape[0] < 1 or actions.shape[1] != 7:
                        raise ValueError(f"Expected [n, 7] actions for chunk, got {actions.shape}")
                    positions, gripper_positions, rewards = [], [], []
                    local_raw = None
                    raw = None
                    final_success = False
                    any_success = False
                    terminated = False
                    truncated = False
                    for issued in actions:
                        phys = apply_embodiment_action(issued, spec, low, high)
                        step = env.step(phys)
                        raw = step["observation"]
                        local_raw = to_local_proprio(raw, anchor_eef_pos, spec)
                        positions.append(local_raw["robot0_eef_pos"].copy())
                        gripper_positions.append(np.asarray(raw["robot0_gripper_qpos"], dtype=np.float32).copy())
                        rewards.append(float(step["reward"]))
                        final_success = bool(step["success"])
                        any_success = any_success or final_success
                        terminated = bool(step["terminated"])
                        truncated = bool(step["truncated"])
                        if terminated or truncated:
                            break
                    result = dict(
                        observation=local_raw,
                        agentview_64=downsample_agentview(raw["agentview_image"], 64),
                        positions=positions,
                        gripper_positions=gripper_positions,
                        rewards=rewards,
                        steps=len(rewards),
                        success=final_success,
                        any_success=any_success,
                        terminated=terminated,
                        truncated=truncated,
                    )
                else:
                    raise ValueError(f"Unknown simulator op {op!r}")
                pipe.send((True, result))
        except EOFError:
            pass
        except Exception:
            pipe.send((False, traceback.format_exc()))
        finally:
            env.close()


if __name__ == "__main__":
    serve(int(sys.argv[1]))
