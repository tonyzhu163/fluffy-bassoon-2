"""Persistent multi-arm (Panda/Sawyer/UR5e) LIBERO-Pro2 subprocess simulator worker."""

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
SIM_PYTHON = ROOT / "setups/libero-meta-learning-20261117-r1/env/bin/python"
ALLOWED_ARMS = ("Panda", "Sawyer", "UR5e")


class UR5eSimulator:
    """Client handle wrapping the isolated multi-arm robosuite/LIBERO simulator subprocess."""

    def __init__(self, *, allow_mesa: bool = False):
        self._closed = False
        self.pipe, peer = Pipe()
        env = dict(os.environ)
        env["FLUFFY_ALLOW_MESA"] = "1" if allow_mesa else "0"
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


def _probe_egl(allow_mesa: bool) -> tuple[str, str]:
    from mujoco.egl import GLContext
    from OpenGL import GL

    context = GLContext(16, 16)
    try:
        context.make_current()
        vendor = GL.glGetString(GL.GL_VENDOR)
        renderer = GL.glGetString(GL.GL_RENDERER)
        if not vendor or not renderer:
            raise RuntimeError(f"Empty EGL vendor/renderer: {vendor!r} / {renderer!r}")
        if not allow_mesa and b"nvidia" not in vendor.lower():
            raise RuntimeError(f"Expected NVIDIA EGL vendor, got {vendor!r} / {renderer!r}")
        return vendor.decode(), renderer.decode()
    finally:
        context.free()


def serve(fd: int):
    from arm_env import ArmEnv

    allow_mesa = os.environ.get("FLUFFY_ALLOW_MESA") == "1"
    renderer_vendor, renderer_name = _probe_egl(allow_mesa)

    env = None
    spec = None
    nominal_wire = True
    anchor_eef_pos = None
    low = np.full((7,), -1.0, dtype=np.float32)
    high = np.full((7,), 1.0, dtype=np.float32)
    done = True
    controls = 0
    max_steps = 0

    with Connection(fd) as pipe:
        try:
            while True:
                op, value = pipe.recv()
                if op == "create":
                    if env is not None:
                        env.close()
                        env = None
                    arm = str(value["arm"])
                    if arm not in ALLOWED_ARMS:
                        raise ValueError(f"Unknown arm {arm!r}; expected one of {ALLOWED_ARMS}")
                    variant = str(value.get("variant", "BASE"))
                    if variant != "BASE":
                        raise ValueError(f"Multi-arm worker only supports BASE variant, got {variant!r}")
                    suite = str(value["suite"])
                    task_id = int(value["task_id"])
                    init_state_id = int(value["init_state_id"])
                    seed = int(value["seed"])
                    spec_name = str(value.get("embodiment", "nom"))
                    spec = get_embodiment_spec(spec_name)
                    nominal_wire = (spec.yaw_deg, spec.xy_scale, spec.gripper_sign) == (0.0, 1.0, 1)

                    env = ArmEnv(arm=arm, suite=suite, task_id=task_id)
                    first = env.reset(init_state_id=init_state_id, seed=seed)
                    identity = first["model_identity"]
                    expected_dof = 6 if arm == "UR5e" else 7
                    if int(identity["joint_count"]) != expected_dof:
                        raise ValueError(
                            f"Expected joint_count={expected_dof} on {arm}, got {identity['joint_count']}"
                        )

                    max_steps = int(first["max_steps"])
                    if value.get("max_steps", max_steps) != max_steps:
                        raise ValueError("Use max_controls on the motor to cap a rollout at the manifest horizon")

                    raw = first["observation"]
                    anchor_eef_pos = np.asarray(raw["robot0_eef_pos"], dtype=np.float32).copy()
                    local_raw = raw if spec.yaw_deg == 0 else to_local_proprio(raw, anchor_eef_pos, spec)
                    action_schema = dict(
                        shape=[7],
                        dtype="float32",
                        controller=str(identity["controller"]),
                        control_mode="relative",
                        control_freq=int(identity["control_freq"]),
                        low=[-1.0] * 7,
                        high=[1.0] * 7,
                    )
                    create_spec = dict(
                        arm=arm,
                        suite=suite,
                        task_id=task_id,
                        variant=variant,
                        init_state_id=init_state_id,
                        seed=seed,
                        embodiment=spec.name,
                        max_steps=max_steps,
                    )
                    controls, done = 0, False
                    initial_out = dict(
                        observation=local_raw,
                        agentview_64=downsample_agentview(raw["agentview_image"], 64),
                        anchor_eef_pos=anchor_eef_pos.copy(),
                        language_instruction=str(first["language_instruction"]),
                        model_identity=identity,
                        environment_reproducibility=first["environment_reproducibility"],
                        action_schema=action_schema,
                        worker_step_index=int(first["worker_step_index"]),
                        max_steps=max_steps,
                        renderer=renderer_name,
                        renderer_vendor=renderer_vendor,
                    )
                    result = create_spec, initial_out
                elif op == "chunk":
                    if env is None or done:
                        raise RuntimeError("Call create before stepping a new or completed episode")
                    actions = np.asarray(value, dtype=np.float32)
                    if actions.ndim != 2 or actions.shape[0] < 1 or actions.shape[1] != 7:
                        raise ValueError(f"Expected [n, 7] actions for chunk, got {actions.shape}")
                    if not np.isfinite(actions).all() or np.any(actions < low) or np.any(actions > high):
                        raise ValueError("Issued action is nonfinite or outside [-1, 1]")

                    positions, gripper_positions, joint_positions, rewards = [], [], [], []
                    local_raw = None
                    raw = None
                    step = None
                    final_success = False
                    any_success = False
                    terminated = False
                    truncated = False
                    for issued in actions:
                        phys = issued if nominal_wire else apply_embodiment_action(issued, spec, low, high)
                        step = env.step(phys)
                        controls += 1
                        if int(step["worker_step_index"]) != controls:
                            raise ValueError(f"Unexpected native control index at {controls}")
                        raw = step["observation"]
                        local_raw = raw if spec.yaw_deg == 0 else to_local_proprio(raw, anchor_eef_pos, spec)
                        positions.append(np.asarray(local_raw["robot0_eef_pos"], dtype=np.float32).copy())
                        gripper_positions.append(np.asarray(raw["robot0_gripper_qpos"], dtype=np.float32).copy())
                        joint_positions.append(np.asarray(raw["robot0_joint_pos"], dtype=np.float32).copy())
                        rewards.append(float(step["reward"]))
                        final_success = bool(step["success"])
                        any_success = any_success or final_success
                        terminated = bool(step["terminated"])
                        truncated = bool(step["truncated"])
                        done = terminated or truncated
                        if controls == max_steps and not done:
                            raise ValueError("ArmEnv exhausted its horizon without a terminal flag")
                        if done:
                            break
                    result = dict(
                        observation=local_raw,
                        agentview_64=downsample_agentview(raw["agentview_image"], 64),
                        positions=positions,
                        gripper_positions=gripper_positions,
                        joint_positions=joint_positions,
                        rewards=rewards,
                        steps=len(rewards),
                        success=final_success,
                        any_success=any_success,
                        terminated=terminated,
                        truncated=truncated,
                        language_instruction=str(step["language_instruction"]),
                    )
                else:
                    raise ValueError(f"Unknown simulator op {op!r}")
                pipe.send((True, result))
        except EOFError:
            pass
        except Exception:
            pipe.send((False, traceback.format_exc()))
        finally:
            if env is not None:
                env.close()


if __name__ == "__main__":
    serve(int(sys.argv[1]))
