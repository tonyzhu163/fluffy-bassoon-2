"""Exercise uncapped native stopping with a stationary CPU motor in each registered task."""
import os
from pathlib import Path
import sys

import numpy as np
from fluffy_bassoon.xemb.embodiment import get_embodiment_spec
from matched import contract
from panel import require, save
from ur5e_motor import run_episode
from ur5e_sim_worker import UR5eSimulator


class Stationary:
    def infer(self, item, *, noise):
        return {"actions": np.zeros((10, 7), np.float32)}


def main(path):
    require(os.environ["CUDA_VISIBLE_DEVICES"] == "" and len(os.sched_getaffinity(0)) <= 2, "CPU fixture")
    data, cases, _ = contract()
    detail = []
    with UR5eSimulator(allow_mesa=True) as sim:
        for suite, task in data["tasks"].items():
            case = next(c for c in cases if c["suite"] == suite and c["arm"] == "Panda")
            r = run_episode(sim, Stationary(), case, get_embodiment_spec("nom"), canonicalize_quat=False)
            require(r["controls"] == task["bddl"]["max_steps"] == r["reset"]["horizon"]
                    and r["truncated"] and not r["terminated"] and r["native_rewards"] == [0.0]*r["controls"]
                    and not r["success"], "uncapped stationary timeout")
            detail.append(dict(case=case, controls=r["controls"], truncated=r["truncated"], reset=r["reset"],
                               summed_native_rewards=sum(r["native_rewards"]), stationary_action=[0.0]*7))
    save(path, dict(fixture=True, motor="stationary", cpu_affinity=sorted(os.sched_getaffinity(0)), episodes=detail))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
