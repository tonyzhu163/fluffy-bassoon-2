"""Persistent LIBERO-Pro2 subprocess simulator worker for physical MuJoCo TCP offsets."""

from multiprocessing import Pipe
from multiprocessing.connection import Connection
from pathlib import Path
import os
import subprocess
import sys
import traceback

import numpy as np

from fluffy_bassoon.xemb.embodiment import downsample_agentview
from fluffy_bassoon.xemb.tcp import apply_mujoco_tcp_offset

ROOT = Path("/mowenjie/fluffy-bassoon-2")
PRO = ROOT / "sources/libero-pro2-e5500c69"
SIM_PYTHON = ROOT / "setups/libero-meta-learning-20261117-r1/env/bin/python"
PROPRIO_KEYS = (
    "robot0_joint_pos",
    "robot0_joint_vel",
    "robot0_eef_pos",
    "robot0_eef_quat_xyzw",
    "robot0_gripper_qpos",
)


class TcpSimulator:
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


def validate_reset(initial, create_spec, entry):
    from libero.libero.envs.bddl_utils import get_problem_info
    from libero_remote.environment.observations import describe_observation
    from libero_remote.reproducibility import canonical_fingerprint

    identity = initial["environment_reproducibility"]
    plan = identity["perturbation_plan"]
    if (
        initial["worker_step_index"] != 0
        or not identity["initial_state_fingerprint"]
        or plan["case_id"] != entry["case_id"]
        or plan["seed"] != create_spec["seed"]
        or identity["perturbation_plan_fingerprint"] != canonical_fingerprint(plan)
    ):
        raise ValueError(f"Unexpected native reset identity for {create_spec}")
    instruction = initial["language_instruction"]
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Missing released reset instruction")
    if entry["case_id"] == "BASE":
        expected = get_problem_info(create_spec["_bddl_path"])["language_instruction"]
        if instruction != expected:
            raise ValueError(f"Released BASE reset prompt {instruction!r} differs from BDDL {expected!r}")

    schema = initial["action_schema"]
    expected_action = dict(
        version=entry["action_schema_version"],
        shape=[7],
        dtype="float32",
        controller="OSC_POSE",
        control_mode="relative",
        translation_frame="world",
        rotation_frame="world",
        rotation_representation="axis_angle",
        gripper_semantics="negative_open_positive_close_zero_hold",
        action_repeat=1,
        control_freq=20,
        low=[-1.0] * 7,
        high=[1.0] * 7,
    )
    for field, expected in expected_action.items():
        if schema[field] != expected:
            raise ValueError(f"Unexpected action_schema.{field}: {schema[field]!r}, expected {expected!r}")

    raw = initial["observation"]
    if (
        initial["observation_schema_version"] != entry["observation_schema_version"]
        or initial["observation_schema"] != describe_observation(raw)
    ):
        raise ValueError("Released observation schema disagrees with manifest or reset arrays")
    shapes = dict(
        agentview_image=(256, 256, 3),
        robot0_eye_in_hand_image=(256, 256, 3),
        robot0_joint_pos=(7,),
        robot0_joint_vel=(7,),
        robot0_eef_pos=(3,),
        robot0_eef_quat_xyzw=(4,),
        robot0_gripper_qpos=(2,),
    )
    if raw.keys() != shapes.keys():
        raise ValueError(f"Unexpected public reset fields: {sorted(raw)}")
    for field, shape in shapes.items():
        dtype = np.uint8 if field.endswith("_image") else np.float32
        if raw[field].shape != shape or raw[field].dtype != dtype or not np.isfinite(raw[field]).all():
            raise ValueError(f"Invalid native reset array {field}")


def _pack_views(raw: dict, *, need_rgb: bool, need_64: bool) -> dict:
    obs = raw if need_rgb else {k: raw[k] for k in PROPRIO_KEYS}
    return dict(
        observation=obs,
        has_full_rgb=bool(need_rgb),
        agentview_64=downsample_agentview(raw["agentview_image"], 64) if need_64 else None,
        wrist_64=downsample_agentview(raw["robot0_eye_in_hand_image"], 64) if need_64 else None,
    )


def serve(fd: int):
    from OpenGL import GL
    from libero_remote.environment.libero import LiberoEnvironmentAdapter
    from libero_remote.environment.task_manifest import TaskManifest

    allow_mesa = os.environ.get("FLUFFY_ALLOW_MESA") == "1"
    env = LiberoEnvironmentAdapter()
    manifest = None
    low = None
    high = None
    done = True
    controls = 0
    max_steps = 0
    last_raw = None
    last_prompt = None

    with Connection(fd) as pipe:
        try:
            while True:
                op, value = pipe.recv()
                if op == "create":
                    env.close()
                    if manifest is None:
                        manifest = TaskManifest.load(str(PRO / "assets/task_manifest.json"))
                    variant = value.get("variant", "BASE")
                    key = value["suite"], int(value["task_id"]), variant
                    dx_m = float(value.get("dx_m", 0.0))
                    dy_m = float(value.get("dy_m", 0.0))
                    dz_m = float(value.get("dz_m", 0.0))
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
                    shift_meta = apply_mujoco_tcp_offset(env._env.env, dx_m, dy_m, dz_m)
                    vendor = GL.glGetString(GL.GL_VENDOR)
                    renderer = GL.glGetString(GL.GL_RENDERER)
                    if not vendor or not renderer or (not allow_mesa and b"nvidia" not in vendor.lower()):
                        raise RuntimeError(f"Unexpected EGL vendor/renderer: {vendor!r} / {renderer!r}")
                    last_raw = initial["observation"]
                    last_prompt = initial["language_instruction"]
                    bounds = initial["action_schema"]
                    low = np.asarray(bounds["low"], dtype=np.float32)
                    high = np.asarray(bounds["high"], dtype=np.float32)
                    controls, done = 0, False
                    initial_out = dict(
                        initial,
                        agentview_64=downsample_agentview(last_raw["agentview_image"], 64),
                        wrist_64=downsample_agentview(last_raw["robot0_eye_in_hand_image"], 64),
                        shift_meta=shift_meta,
                        renderer=renderer.decode(),
                        renderer_vendor=vendor.decode(),
                    )
                    result = create_spec, initial_out
                elif op == "step":
                    if done:
                        raise RuntimeError("Call create before stepping a new or completed episode")
                    if isinstance(value, dict):
                        act = np.asarray(value["action"], dtype=np.float32)
                        need_rgb = bool(value.get("need_rgb", False))
                        need_64 = bool(value.get("need_64", False))
                    else:
                        act = np.asarray(value, dtype=np.float32)
                        need_rgb = False
                        need_64 = False
                    if act.shape != (7,) or not np.isfinite(act).all() or np.any(act < low) or np.any(act > high):
                        raise ValueError(f"Issued action invalid or out of bounds: {act!r}")
                    step = env.step(act)
                    controls += 1
                    if step["worker_step_index"] != controls:
                        raise ValueError(f"Unexpected native control index at {controls}")
                    last_raw = step["observation"]
                    last_prompt = step["info"]["language_instruction"]
                    terminated = bool(step["terminated"])
                    truncated = bool(step["truncated"])
                    done = terminated or truncated
                    if controls == max_steps and not done:
                        raise ValueError("Native adapter exhausted its horizon without a terminal flag")
                    packed = _pack_views(last_raw, need_rgb=need_rgb, need_64=need_64)
                    result = dict(
                        **packed,
                        reward=float(step["reward"]),
                        success=bool(step["success"]),
                        terminated=terminated,
                        truncated=truncated,
                        controls=controls,
                        language_instruction=last_prompt,
                    )
                elif op == "observe":
                    if last_raw is None:
                        raise RuntimeError("Call create before observe")
                    result = dict(
                        **_pack_views(last_raw, need_rgb=True, need_64=True),
                        language_instruction=last_prompt,
                    )
                elif op == "chunk":
                    if done:
                        raise RuntimeError("Call create before stepping a new or completed episode")
                    actions = np.asarray(value, dtype=np.float32)
                    if actions.ndim != 2 or actions.shape[0] < 1 or actions.shape[1] != 7:
                        raise ValueError(f"Expected [n, 7] actions for chunk, got {actions.shape}")
                    if not np.isfinite(actions).all() or np.any(actions < low) or np.any(actions > high):
                        raise ValueError("Issued action is nonfinite or outside the advertised bounds")
                    rewards = []
                    final_success = False
                    terminated = False
                    truncated = False
                    for act in actions:
                        step = env.step(act)
                        controls += 1
                        if step["worker_step_index"] != controls:
                            raise ValueError(f"Unexpected native control index at {controls}")
                        last_raw = step["observation"]
                        last_prompt = step["info"]["language_instruction"]
                        rewards.append(float(step["reward"]))
                        final_success = bool(step["success"])
                        terminated = bool(step["terminated"])
                        truncated = bool(step["truncated"])
                        done = terminated or truncated
                        if controls == max_steps and not done:
                            raise ValueError("Native adapter exhausted its horizon without a terminal flag")
                        if done:
                            break
                    result = dict(
                        **_pack_views(last_raw, need_rgb=True, need_64=True),
                        rewards=rewards,
                        steps=len(rewards),
                        success=final_success,
                        terminated=terminated,
                        truncated=truncated,
                        controls=controls,
                        language_instruction=last_prompt,
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
