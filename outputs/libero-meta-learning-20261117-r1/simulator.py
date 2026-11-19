"""Keep Pro2's robosuite version in its own interpreter on the allocated GPU."""
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from pathlib import Path
import subprocess
import sys
import traceback

from job import PRO, PYTHON


class Simulator:
    def __init__(self):
        self._closed = False
        self.pipe, peer = Pipe()
        try:
            self.worker = subprocess.Popen(
                [PYTHON, '-Bu', Path(__file__).resolve(), str(peer.fileno())],
                pass_fds=(peer.fileno(),))
        finally:
            peer.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def call(self, op, value):
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


def serve(fd):
    from libero_remote.environment.libero import LiberoEnvironmentAdapter
    from libero_remote.environment.task_manifest import TaskManifest

    env = LiberoEnvironmentAdapter()
    manifest = None
    with Connection(fd) as pipe:
        try:
            while True:
                op, value = pipe.recv()
                if op == 'create':
                    env.close()
                    if manifest is None:
                        manifest = TaskManifest.load(str(PRO/'assets/task_manifest.json'))
                    key = value['suite'], value['task_id'], value['variant']
                    spec = dict(value, camera_names=['agentview', 'robot0_eye_in_hand'],
                                camera_height=256, camera_width=256, control_mode='relative',
                                max_steps=manifest.require(*key)['max_steps'],
                                _bddl_path=str(manifest.resolve_bddl(*key)))
                    result = spec, env.create(spec)
                elif op == 'step':
                    result = env.step(value)
                elif op == 'chunk':
                    actions = list(value)
                    if not actions:
                        raise ValueError('Require at least one action for chunk')
                    positions, rewards = [], []
                    raw = None
                    final_success = False
                    any_success = False
                    terminated = False
                    truncated = False
                    for action in actions:
                        step = env.step(action)
                        raw = step['observation']
                        positions.append(raw['robot0_eef_pos'].copy())
                        rewards.append(float(step['reward']))
                        final_success = bool(step['success'])
                        any_success = any_success or final_success
                        terminated = bool(step['terminated'])
                        truncated = bool(step['truncated'])
                        if terminated or truncated:
                            break
                    result = dict(observation=raw, positions=positions, rewards=rewards,
                                  steps=len(rewards), success=final_success,
                                  any_success=any_success, terminated=terminated,
                                  truncated=truncated)
                else:
                    raise ValueError(op)
                pipe.send((True, result))
        except EOFError:
            pass
        except Exception:
            pipe.send((False, traceback.format_exc()))
        finally:
            env.close()


if __name__ == '__main__':
    serve(int(sys.argv[1]))
