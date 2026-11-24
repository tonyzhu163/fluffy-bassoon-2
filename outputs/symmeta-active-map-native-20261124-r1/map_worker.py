"""Inject the full map once at the simulator boundary, on the nominal embodiment wire."""
from multiprocessing.connection import Connection
import sys
from unittest.mock import patch

import numpy as np
import arm_env
import ur5e_sim_worker as worker


def serve(fd):
    step_arm = arm_env.ArmEnv.step
    records = []
    truth = None

    class Wire(Connection):
        def recv(self):
            nonlocal truth
            op, value = super().recv()
            if op == "create":
                assert not records
                value = dict(value)
                truth = np.asarray(value.pop("forward_map"), dtype=np.float64)
                assert value["embodiment"] == "nom" and truth.shape == (2,2)
                assert np.isfinite(truth).all() and np.linalg.norm(truth, np.inf) <= 1
            return op, value

        def send(self, message):
            ok, obj = message
            if ok and isinstance(obj, dict) and "steps" in obj:
                assert obj["steps"] == len(records) and obj["rewards"] == [r["reward"] for r in records]
                obj = dict(obj, native_steps=records.copy())
                records.clear()
            super().send((ok, obj))

    def measured(env, issued):
        assert issued.dtype == np.float32 and np.max(np.abs(issued)) <= 1
        physical = issued.copy()
        physical[:2] = truth @ issued[:2]
        seen = []
        native_step = env._env.step

        def observe(actual):
            seen.append(actual.copy())
            return native_step(actual)

        with patch.object(env._env, "step", observe):
            result = step_arm(env, physical)
        actual, = seen
        assert actual.tobytes() == physical.tobytes() == result["issued_action"].tobytes()
        records.append(dict(issued=issued.tolist(), physical=actual.tolist(), index=result["worker_step_index"],
                            **{k:result[k] for k in ("reward", "success", "terminated", "truncated")}))
        return result

    with patch.object(worker, "Connection", Wire), patch.object(arm_env.ArmEnv, "step", measured):
        worker.serve(fd)


if __name__ == "__main__":
    serve(int(sys.argv[1]))
