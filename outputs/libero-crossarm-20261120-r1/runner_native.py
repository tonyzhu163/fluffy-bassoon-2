"""Transport bounded OSC chunks between the OpenPI and LIBERO interpreters."""
from multiprocessing import Pipe
from multiprocessing.connection import Connection
import os
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np


class Simulator:
    def __init__(self):
        self.pipe, peer = Pipe()
        try:
            self.worker = subprocess.Popen(
                [os.environ["CROSSARM_SIM_PYTHON"], "-Bu", str(Path(__file__).resolve()), str(peer.fileno())],
                pass_fds=(peer.fileno(),),
            )
        finally:
            peer.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.pipe.close()
        try:
            self.worker.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.worker.terminate()
            self.worker.wait(timeout=15)

    def call(self, op, value):
        self.pipe.send((op, value))
        ok, result = self.pipe.recv()
        if not ok:
            raise RuntimeError(result)
        return result


def serve(fd):
    with Connection(fd) as channel:
        env = None
        try:
            from arm_env import ArmEnv
            from OpenGL import GL

            while True:
                op, value = channel.recv()
                if op == "reset":
                    if env is not None:
                        env.close()
                        env = None
                    env = ArmEnv(arm=value["body"], suite=value["suite"], task_id=value["task_id"])
                    result = env.reset(init_state_id=value["init_state_id"], seed=value["seed"])
                    vendor = GL.glGetString(GL.GL_VENDOR)
                    renderer = GL.glGetString(GL.GL_RENDERER)
                    if not vendor or b"nvidia" not in vendor.lower() or not renderer:
                        raise RuntimeError(f"Expected allocated NVIDIA EGL device, observed {vendor!r}, {renderer!r}")
                    result = dict(result, renderer=[vendor.decode(), renderer.decode()])
                    controls = int(result["worker_step_index"])
                    horizon = int(result["max_steps"])
                    prompt = result["language_instruction"]
                    if controls != 0 or horizon <= 0 or not isinstance(prompt, str) or not prompt.strip():
                        raise ValueError("Invalid BASE episode start")
                    done = False
                elif op == "chunk":
                    if env is None or done:
                        raise RuntimeError("Reset the simulator before issuing another control")
                    actions = np.asarray(value)
                    if (actions.ndim != 2 or actions.shape[1] != 7 or not 1 <= len(actions) <= 5
                            or not np.isfinite(actions).all() or np.max(np.abs(actions)) > 1
                            or controls + len(actions) > horizon):
                        raise ValueError("Invalid bounded native control chunk")
                    commands, motions, rewards = [], [], []
                    for action in actions:
                        step = env.step(action)
                        controls += 1
                        if controls != step["worker_step_index"] or step["language_instruction"] != prompt:
                            raise ValueError("Unexpected native control counter or BASE instruction")
                        command = np.asarray(step["issued_action"], dtype=np.float32)
                        motion = np.asarray(step["sensor_motion"], dtype=np.float32)
                        reward = float(step["reward"])
                        if (not np.array_equal(command, action) or motion.shape != (7,)
                                or not np.isfinite(motion).all() or not np.isfinite(reward)):
                            raise ValueError("Invalid action, sensor response, or native reward")
                        commands.append(command)
                        motions.append(motion)
                        rewards.append(reward)
                        done = bool(step["terminated"] or step["truncated"])
                        if step["success"] and not done:
                            raise ValueError("Native success must end the episode")
                        if done:
                            break
                    if controls == horizon and not done:
                        raise ValueError("Episode passed its native horizon")
                    result = dict(observation=step["observation"], commands=np.stack(commands),
                                  motion=np.stack(motions), rewards=rewards, success=bool(step["success"]),
                                  terminated=bool(step["terminated"]), truncated=bool(step["truncated"]))
                else:
                    raise ValueError(f"Unknown simulator operation {op!r}")
                channel.send((True, result))
        except EOFError:
            pass
        except Exception:
            try:
                channel.send((False, traceback.format_exc()))
            except (BrokenPipeError, EOFError):
                pass
        finally:
            if env is not None:
                env.close()


if __name__ == "__main__":
    serve(int(sys.argv[1]))
