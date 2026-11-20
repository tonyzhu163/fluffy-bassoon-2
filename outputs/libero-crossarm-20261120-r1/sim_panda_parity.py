"""Compare released Panda frames, full reset physics and native transitions on the allocated Linux CPU."""
import hashlib
import json
import sys

import numpy as np

from arm_env import ArmEnv, CAMERAS
from libero_remote.environment.libero import LiberoEnvironmentAdapter


def public_hashes(obs):
    return {k: (v.shape, v.dtype.str, hashlib.sha256(v.tobytes(order="C")).hexdigest()) for k, v in obs.items()}


def trajectory(env, first, commands):
    outcome = dict(start=public_hashes(first["observation"]), prompt=first["language_instruction"],
                   state=first["environment_reproducibility"]["initial_state_fingerprint"], steps=[])
    for action in commands:
        step = env.step(action)
        outcome["steps"].append(dict(observation=public_hashes(step["observation"]),
                                     reward=step["reward"], success=step["success"],
                                     terminated=step["terminated"], truncated=step["truncated"]))
        if step["terminated"] or step["truncated"]:
            break
    return outcome


def check():
    args = dict(suite="libero_goal", task_id=4)
    start = dict(init_state_id=0, seed=20261520)
    commands = np.array([[0.25, 0, 0, 0, 0, 0, -1], [0, 0, 0, 0, 0.25, 0, 1],
                         [-0.25, 0, 0, 0, -0.25, 0, 0]], dtype=np.float32)
    with ArmEnv(arm="Panda", **args) as env:
        first = env.reset(**start)
        local = trajectory(env, first, commands)
        path = str(env.bddl)
    native = LiberoEnvironmentAdapter()
    try:
        released = native.create(dict(**args, **start, variant="BASE", _bddl_path=path,
                                     camera_names=list(CAMERAS), camera_height=256, camera_width=256,
                                     control_mode="relative", max_steps=first["max_steps"]))
        reference = trajectory(native, released, commands)
    finally:
        native.close()
    if local != reference:
        raise ValueError(f"Physical Panda differs from the released adapter: {local!r} vs {reference!r}")
    return dict(passed=True, **args, **start, **local)


if __name__ == "__main__":
    if sys.platform != "linux":
        raise SystemExit("Run in the pinned simulator interpreter on the allocated Linux CPU")
    print(json.dumps(check(), indent=2))
