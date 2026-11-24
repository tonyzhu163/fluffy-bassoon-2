"""Attach actual physical commands at ArmEnv's native simulator boundary to the original PR7 replies."""
from multiprocessing.connection import Connection
import sys
from unittest.mock import patch

import arm_env
import ur5e_sim_worker as worker


def main(fd):
    original = arm_env.ArmEnv.step
    receipt = []

    def measured(env, physical):
        step = env._env.step
        observed = []

        def capture(action):
            observed.append(action.copy())
            return step(action)

        with patch.object(env._env, "step", capture):
            row = original(env, physical)
        value, = observed
        if value.tobytes() != physical.tobytes() or row["issued_action"].tobytes() != value.tobytes():
            raise ValueError("Physical action changed at native boundary")
        receipt.append(dict(action=value.tolist(), index=row["worker_step_index"],
                            **{k:row[k] for k in ("reward", "success", "terminated", "truncated")}))
        return row

    class Wire(Connection):
        def send(self, message):
            ok, obj = message
            if ok and isinstance(obj, dict) and "steps" in obj:
                if len(receipt) != obj["steps"] or [s["reward"] for s in receipt] != obj["rewards"]:
                    raise ValueError("Native action/reward count")
                obj = dict(obj, native_steps=receipt.copy())
                receipt.clear()
            super().send((ok, obj))

    with patch.object(arm_env.ArmEnv, "step", measured), patch.object(worker, "Connection", Wire):
        worker.serve(fd)


if __name__ == "__main__":
    main(int(sys.argv[1]))
