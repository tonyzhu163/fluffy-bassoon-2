"""CPU IPC/wiring fixture on real Panda and Kinova3 physics, software rendering; no VLA outcome."""
import json
from pathlib import Path
import numpy as np

from native_rollout import Trace, source_tracking
from panel import cases
from ur5e_motor import run_episode
from ur5e_sim_worker import UR5eSimulator
from fluffy_bassoon.xemb.embodiment import get_embodiment_spec


class FixedPolicy:
    def infer(self, item, *, noise):
        assert item["observation/image"].shape == (224, 224, 3)
        assert item["observation/state"].shape == (8,) and noise.shape == (10, 32)
        a = np.zeros((10, 7), np.float32)
        a[:, :3] = [0.15, -0.12, 0.05]
        return {"actions": a}


class Cap:
    """Five-control fixture stops the privileged tracker without running a full episode."""
    def __init__(self, sim):
        self.sim = sim
        self.i = 0

    def call(self, op, value):
        response = self.sim.call(op, value)
        if op == "chunk":
            self.i += response["steps"]
            if self.i == 5:
                response["truncated"] = True
        return response


def main():
    path = Path("/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-motion-models-20261123-r1/cpu01/matrices.json")
    m = json.loads(path.read_text())["Kinova3/libero_goal/3"]
    src = next(cases("Panda"))
    target = dict(src, arm="Kinova3")
    jobs = [(src, None), (target, None), (target, m["matrix"]), (target, m["planar_matrix"])]
    traces = []
    with UR5eSimulator(allow_mesa=True) as sim:
        for case, matrix in jobs:
            trace = Trace(sim, np.asarray(matrix) if matrix else None)
            rec = run_episode(trace, FixedPolicy(), case, get_embodiment_spec("nom"), max_controls=5)
            assert rec["controls"] == len(rec["native_rewards"]) == 5 and rec["success"] is False
            traces.append((trace, rec))
        tracked = source_tracking(Cap(sim), target, traces[0][0].chunks, 5)
    assert tracked["controls"] == 5
    assert len({r["reset"]["noise_sha256"] for _, r in traces}) == 1
    assert len({r["reset"]["physics"] for _, r in traces[1:]}) == 1
    assert len({r["opening_proposal_sha256"] for _, r in traces[1:]}) == 1
    assert all(np.allclose(np.asarray(traces[i][0].chunks[0]["cmd"])[:, :3],
                           np.asarray(traces[1][0].chunks[0]["cmd"])[:, :3] @ traces[i][0].matrix)
               for i in (2, 3))
    print(json.dumps(dict(status="qualified", trajectories=len(traces), controls_per_trajectory=5,
                          native_reward_tapes=[r["native_rewards"] for _, r in traces],
                          privileged_tracker_five_controls=tracked["tracking_error_m_rad"],
                          finding="Real reset/action IPC; fixed policy and five-control cap, no task success evaluation.")))


if __name__ == "__main__":
    main()
