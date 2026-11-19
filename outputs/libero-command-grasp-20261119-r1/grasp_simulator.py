"""Native Goal 2 actuator and evaluator. Run under the qualified Pro2 interpreter."""

from multiprocessing.connection import Connection
from pathlib import Path
import sys
import traceback

import numpy as np
from OpenGL import GL
from libero.libero.envs.bddl_utils import robosuite_parse_problem
from libero_remote.environment.libero import LiberoEnvironmentAdapter
from libero_remote.environment.task_manifest import TaskManifest


def robot_state(raw):
    return np.concatenate([raw[key] for key in (
        "robot0_eef_pos", "robot0_eef_quat_xyzw", "robot0_gripper_qpos",
    )])


def serve(channel, release):
    manifest = TaskManifest.load(str(release / "assets/task_manifest.json"))
    base_problem = robosuite_parse_problem(str(manifest.resolve_bddl("libero_goal", 2, "BASE")))
    words = base_problem["language_instruction"]
    prompt = " ".join(words) if isinstance(words, list) else words
    env = LiberoEnvironmentAdapter()
    try:
        while True:
            op, value = channel.recv()
            if op == "reset":
                pose, seed, condition, sign = value
                if condition not in ("BASE", "D17") or sign not in (-1, 1):
                    raise ValueError("Unknown native condition or wire convention")
                env.close()
                key = "libero_goal", 2, condition
                entry = manifest.require(*key)
                horizon = entry["max_steps"]
                initial = env.create(dict(
                    suite=key[0], task_id=key[1], variant=condition, init_state_id=pose, seed=seed,
                    _bddl_path=str(manifest.resolve_bddl(*key)), max_steps=horizon, control_mode="relative",
                    camera_names=["agentview", "robot0_eye_in_hand"], camera_height=256, camera_width=256,
                ))
                world = env._env.env
                runtime = world.perturbation_runtime
                if (runtime.plan.case_id != condition or env._env.check_success()
                        or world.parsed_problem["goal_state"] != base_problem["goal_state"]
                        or initial["worker_step_index"] != 0 or initial["language_instruction"] != prompt):
                    raise ValueError("Invalid native goal, condition, reset index or released instruction")
                renderer = GL.glGetString(GL.GL_RENDERER)
                vendor = GL.glGetString(GL.GL_VENDOR)
                if not renderer or not vendor or b"nvidia" not in vendor.lower():
                    raise RuntimeError(f"Expected the allocated NVIDIA EGL renderer, got {renderer!r}")
                index, done = 0, False
                result = dict(initial=initial, prompt=prompt, horizon=horizon, renderer=renderer.decode(),
                              manifest=manifest.sha256, bddl=entry["bddl_sha256"])
            elif op == "step":
                if done or index >= horizon:
                    raise RuntimeError("Native episode already ended")
                native_input = value.copy()
                native_input[0] *= sign
                step = env.step(native_input)
                index += 1
                if step["worker_step_index"] != index:
                    raise ValueError("Missing or duplicated native control")
                done = step["terminated"] or step["truncated"]
                raw = step["observation"]
                # In the pinned release transform_action stores its output in previous_action.
                # This readback is evaluator telemetry, downstream of the simulated actuator.
                applied = runtime.previous_action.copy()
                if not np.array_equal(applied[:6], native_input[:6]) or (
                    condition == "BASE" and not np.array_equal(applied, native_input)
                ):
                    raise ValueError("Unexpected transformation inside the released controller")
                if applied[-1] != native_input[-1] and applied[-1] != -1:
                    raise ValueError("Unexpected D17 gripper readback")
                result = dict(state=robot_state(raw), native_input=native_input, applied=applied,
                              observation=raw if index % 5 == 0 or done else None,
                              **{k: step[k] for k in ("reward", "success", "terminated", "truncated")})
            elif op == "diagnostics":
                if not done:
                    raise RuntimeError("Diagnostics are available only after native termination")
                result = runtime.diagnostics()
            else:
                raise ValueError(f"Unknown simulator operation {op}")
            channel.send((True, result))
    except EOFError:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    with Connection(int(sys.argv[1])) as pipe:
        try:
            serve(pipe, Path(sys.argv[2]))
        except Exception:
            pipe.send((False, traceback.format_exc()))
            raise
