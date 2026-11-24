"""Instrument the installed native step while retaining the pinned worker's serve loop."""
from multiprocessing.connection import Connection
import sys
from unittest.mock import patch

import numpy as np

import arm_env
import ur5e_sim_worker as worker
from robosuite.models.grippers.panda_gripper import PandaGripper
from ur5e_motor import public_digest


def serve(fd):
    # Confirm the actual installed model maps the scalar command to opposing finger targets.
    gripper = object.__new__(PandaGripper)
    for sign in (-1, 1):
        gripper.current_action = np.zeros(2)
        actual = gripper.format_action(np.array([sign]))
        np.testing.assert_array_equal(actual, np.array([-sign, sign]) * gripper.speed)
    assert "-1 => open, 1 => closed" in PandaGripper.format_action.__doc__

    original = arm_env.ArmEnv.step
    receipt = []
    final_public = None

    def measured(env, command):
        nonlocal final_public
        actual_actions = []
        native_step = env._env.step

        def outgoing(value):
            assert value.dtype == np.float32 and value.shape == (7,)
            actual_actions.append(value.copy())
            return native_step(value)

        with patch.object(env._env, "step", outgoing):
            result = original(env, command)
        actual, = actual_actions
        assert actual.tobytes() == result["issued_action"].tobytes() == command.tobytes()
        receipt.append(dict(action=actual.tolist(), index=result["worker_step_index"],
                            **{k:result[k] for k in ("reward","success","terminated","truncated")}))
        final_public = result["observation"]
        return result

    class Pipe(Connection):
        def send(self, message):
            nonlocal final_public
            ok, obj = message
            if ok and isinstance(obj, dict) and "steps" in obj:
                assert len(receipt) == obj["steps"] and [r["reward"] for r in receipt] == obj["rewards"]
                assert all(obj[k] == receipt[-1][k] for k in ("success","terminated","truncated"))
                after = public_digest(final_public)
                assert after == public_digest(obj["observation"])
                obj = dict(obj, native_steps=receipt.copy(), after_public=after)
                receipt.clear()
                final_public = None
            super().send((ok, obj))

    with patch.object(arm_env.ArmEnv, "step", measured), patch.object(worker, "Connection", Pipe):
        worker.serve(fd)


if __name__ == "__main__":
    serve(int(sys.argv[1]))
