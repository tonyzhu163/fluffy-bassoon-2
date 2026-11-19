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


def _inspect_robots():
    import robosuite.robots as rs_robots
    import robosuite.models.grippers as rs_grippers

    robot_names = sorted(getattr(rs_robots, "ROBOT_CLASS_MAPPING", {}).keys())
    gripper_names = sorted(name for name in getattr(rs_grippers, "ALL_GRIPPERS", []) if name is not None)
    return dict(robots=robot_names, grippers=gripper_names)


def validate_reset(initial, create_spec, entry):
    from libero.libero.envs.bddl_utils import get_problem_info
    from libero_remote.environment.observations import describe_observation
    from libero_remote.reproducibility import canonical_fingerprint

    identity = initial["environment_reproducibility"]
    plan = identity["perturbation_plan"]
    if (initial["worker_step_index"] != 0 or not identity["initial_state_fingerprint"]
            or plan["case_id"] != entry["case_id"] or plan["seed"] != create_spec["seed"]
            or identity["perturbation_plan_fingerprint"] != canonical_fingerprint(plan)):
        raise ValueError(f"Unexpected native reset identity for {create_spec}")
    instruction = initial["language_instruction"]
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Missing released reset instruction")
    if entry["case_id"] == "BASE":
        expected = get_problem_info(create_spec["_bddl_path"])["language_instruction"]
        if instruction != expected:
            raise ValueError(f"Released BASE reset prompt {instruction!r} differs from BDDL {expected!r}")

    schema = initial["action_schema"]
    expected_action = dict(version=entry["action_schema_version"], shape=[7], dtype="float32",
                           controller="OSC_POSE", control_mode="relative", translation_frame="world",
                           rotation_frame="world", rotation_representation="axis_angle",
                           gripper_semantics="negative_open_positive_close_zero_hold",
                           action_repeat=1, control_freq=20, low=[-1.0] * 7, high=[1.0] * 7)
    for field, expected in expected_action.items():
        if schema[field] != expected:
            raise ValueError(f"Unexpected action_schema.{field}: {schema[field]!r}, expected {expected!r}")
    raw = initial["observation"]
    if (initial["observation_schema_version"] != entry["observation_schema_version"]
            or initial["observation_schema"] != describe_observation(raw)):
        raise ValueError("Released observation schema disagrees with manifest or reset arrays")
    shapes = dict(agentview_image=(256, 256, 3), robot0_eye_in_hand_image=(256, 256, 3),
                  robot0_joint_pos=(7,), robot0_joint_vel=(7,), robot0_eef_pos=(3,),
                  robot0_eef_quat_xyzw=(4,), robot0_gripper_qpos=(2,))
    if raw.keys() != shapes.keys():
        raise ValueError(f"Unexpected public reset fields: {sorted(raw)}")
    for field, shape in shapes.items():
        dtype = np.uint8 if field.endswith("_image") else np.float32
        if raw[field].shape != shape or raw[field].dtype != dtype or not np.isfinite(raw[field]).all():
            raise ValueError(f"Invalid native reset array {field}")


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
    done = True
    controls = 0

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
                    nominal_wire = (spec.yaw_deg, spec.xy_scale, spec.gripper_sign) == (0.0, 1.0, 1)
                    task_entry = manifest.require(*key)
                    max_steps = int(task_entry["max_steps"])
                    if value.get("max_steps", max_steps) != max_steps:
                        raise ValueError("Use max_controls on the motor to cap a rollout at the manifest horizon")
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
                    validate_reset(initial, create_spec, task_entry)
                    vendor = GL.glGetString(GL.GL_VENDOR)
                    renderer = GL.glGetString(GL.GL_RENDERER)
                    if not vendor or not renderer or (not allow_mesa and b"nvidia" not in vendor.lower()):
                        raise RuntimeError(f"Unexpected EGL vendor/renderer: {vendor!r} / {renderer!r}")
                    raw = initial["observation"]
                    anchor_eef_pos = np.asarray(raw["robot0_eef_pos"], dtype=np.float32).copy()
                    bounds = initial["action_schema"]
                    low = np.asarray(bounds["low"], dtype=np.float32)
                    high = np.asarray(bounds["high"], dtype=np.float32)
                    local_raw = raw if spec.yaw_deg == 0 else to_local_proprio(raw, anchor_eef_pos, spec)
                    controls, done = 0, False
                    initial_out = dict(
                        initial,
                        observation=local_raw,
                        agentview_64=downsample_agentview(raw["agentview_image"], 64),
                        anchor_eef_pos=anchor_eef_pos.copy(),
                        renderer=renderer.decode(),
                        renderer_vendor=vendor.decode(),
                    )
                    result = create_spec, initial_out
                elif op == "chunk":
                    if done:
                        raise RuntimeError("Call create before stepping a new or completed episode")
                    actions = np.asarray(value, dtype=np.float32)
                    if actions.ndim != 2 or actions.shape[0] < 1 or actions.shape[1] != 7:
                        raise ValueError(f"Expected [n, 7] actions for chunk, got {actions.shape}")
                    if not np.isfinite(actions).all() or np.any(actions < low) or np.any(actions > high):
                        raise ValueError("Issued action is nonfinite or outside the advertised bounds")
                    positions, gripper_positions, rewards = [], [], []
                    local_raw = None
                    raw = None
                    final_success = False
                    any_success = False
                    terminated = False
                    truncated = False
                    for issued in actions:
                        phys = issued if nominal_wire else apply_embodiment_action(issued, spec, low, high)
                        step = env.step(phys)
                        controls += 1
                        if step["worker_step_index"] != controls:
                            raise ValueError(f"Unexpected native control index at {controls}")
                        raw = step["observation"]
                        local_raw = raw if spec.yaw_deg == 0 else to_local_proprio(raw, anchor_eef_pos, spec)
                        positions.append(local_raw["robot0_eef_pos"].copy())
                        gripper_positions.append(np.asarray(raw["robot0_gripper_qpos"], dtype=np.float32).copy())
                        rewards.append(float(step["reward"]))
                        final_success = bool(step["success"])
                        any_success = any_success or final_success
                        terminated = bool(step["terminated"])
                        truncated = bool(step["truncated"])
                        done = terminated or truncated
                        if controls == max_steps and not done:
                            raise ValueError("Native adapter exhausted its horizon without a terminal flag")
                        if done:
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
                        language_instruction=step["info"]["language_instruction"],
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
