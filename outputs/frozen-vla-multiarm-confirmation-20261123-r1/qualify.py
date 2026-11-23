"""Exercise actual native IPC with a scripted policy on Mesa; no frozen VLA inference."""
import argparse
import os
from pathlib import Path
import socket

import numpy as np

from native_rollout import Trace
from ur5e_motor import run_episode
from ur5e_sim_worker import UR5eSimulator
from fluffy_bassoon.xemb.embodiment import get_embodiment_spec
from screen import ARMS, METHODS, JointChecked, calibrations, cases, check_case, check_pair, historical, key, require, save


class WiringPolicy:
    def __init__(self):
        self.calls = 0

    def infer(self, item, *, noise):
        require(all(item[k].shape == (224, 224, 3) for k in ("observation/image", "observation/wrist_image"))
                and item["observation/state"].shape == (8,) and noise.shape == (10, 32), "OpenPI inputs")
        commands = ((.25, -.17, .09, .03, -.04, .06, -.4), (-.11, .23, -.18, -.06, .07, -.02, .4))
        action = np.asarray(commands[self.calls], dtype=np.float32)
        self.calls += 1
        return {"actions": np.tile(action, (10, 1))}


def main(arm, out):
    cpu_ids = sorted(os.sched_getaffinity(0))
    require(len(cpu_ids) == 1 and os.environ["CUDA_VISIBLE_DEVICES"] == "", "one CPU lane per arm")
    rows = calibrations(arm)
    history = historical(arm)
    summary = []
    with UR5eSimulator(allow_mesa=True) as simulator:
        checker = JointChecked(simulator, arm)
        for case in cases(arm):
            # One historical IIWA failure and one historically working task use full paired action checks.
            paired = (case["suite"], case["task_id"]) in (("libero_10", 8), ("libero_goal", 6))
            row = rows[key(case)]
            methods = METHODS if paired else ("raw",)
            results = {}
            for method in methods:
                field = METHODS[method]
                matrix = np.asarray(row[field], dtype=np.float32) if field else None
                trace = Trace(checker, matrix)
                policy = WiringPolicy()
                cap = 10 if paired else 1
                rec = run_episode(trace, policy, case, get_embodiment_spec("nom"), max_controls=cap, canonicalize_quat=False)
                require(rec["controls"] == cap == len(rec["native_rewards"]) and
                        all(r in (0, 1) for r in rec["native_rewards"]), "native CPU controls")
                require("llvmpipe" in rec["reset"]["renderer"].lower(), "software-only rendering")
                obj = dict(case=case, record=rec, chunks=trace.chunks, model=trace.first["model_identity"],
                           full_reset=trace.first["environment_reproducibility"])
                check_case(obj, case, terminal=False)
                require(row["support"]["physics"] != rec["reset"]["physics"], "calibration/query reset")
                if field:
                    check_pair(results["raw"], obj, row, method)
                    for actual, reference in zip(obj["chunks"], results["raw"]["chunks"], strict=True):
                        expected = np.asarray(reference["cmd"], dtype=np.float32)
                        expected[:, :3] = np.clip(expected[:, :3] @ matrix, -1, 1)
                        require(np.array_equal(np.asarray(actual["cmd"], dtype=np.float32), expected), "scripted action wiring")
                results[method] = obj
            save(out / f"{arm}-{key(case)}.json", results)
            summary.append(dict(case=case, methods=list(methods), controls_per_method=cap,
                                history_layout_13_raw=history[(case["suite"], case["task_id"])],
                                joint_count=checker.width, arm_xml=obj["model"]["xml_sha256"],
                                native_interface=rec["reset"]["action_schema"], renderer=rec["reset"]["renderer"],
                                support_sha256=row["support"]["sha256"], support_controls=row["support"]["controls"],
                                query_physics=rec["reset"]["physics"], support_physics=row["support"]["physics"]))
            print(arm, key(case), list(methods), cap, flush=True)
    save(out / f"{arm}-qualified.json", dict(status="passed", arm=arm, host=socket.gethostname(), cpus=cpu_ids,
            cases=summary, scope="Native CPU wiring with scripted inputs; no full policy episode or success estimate."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=ARMS)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    main(args.arm, args.out)
