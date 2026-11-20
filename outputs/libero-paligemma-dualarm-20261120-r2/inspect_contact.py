"""Replay a recorded native failure and identify the load-bearing contacts."""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "libero-paligemma-crossarm-20261120-r1"))
from sim import Session


def inspect(env):
    raw = env.native.arm._env
    robot = raw.robots[0]
    model, data = raw.sim.model._model, raw.sim.data._data
    geom = env.native.geometry
    contacts = []
    for idx, c in enumerate(data.contact):
        names = [model.geom(int(g)).name or model.body(int(model.geom_bodyid[g])).name for g in (c.geom1, c.geom2)]
        if not any(name.startswith(("robot0", "gripper0")) for name in names):
            continue
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, idx, wrench)
        contacts.append((round(float(wrench[0]), 2), names, c.pos.round(4).tolist()))
    indices = robot._ref_joint_indexes
    q = np.array(data.qpos[robot._ref_joint_pos_indexes])
    bounds = model.jnt_range[indices]
    ranges = np.minimum(q - bounds[:, 0], bounds[:, 1] - q)
    return dict(contacts=sorted(contacts, reverse=True)[:8], joint_margin=ranges.round(3).tolist(),
                finger_bounds=np.stack((geom._points("fingers").min(0), geom._points("fingers").max(0))).round(3).tolist(),
                tau=np.asarray(robot.torques).round(2).tolist(), actual_controls=data.ctrl.round(2).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tape", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    f = json.loads(args.tape.read_text())
    case = f["attempt"]
    records = []
    with Session(case["body"]) as env:
        env.reset(case["pose"], case["seed"])
        env.expert_scene()
        for t, a in enumerate(f["actions"]):
            if t in (0, 45, 65, 76, 82, 90, 110, 150, 220, 297):
                scene = env.expert_scene()
                plate_err = np.linalg.norm(np.subtract(scene["plate"], f["pre_action_scene"][t]["plate"]))
                row = dict(t=t, phase=f["pre_action_scene"][t]["phase"], plate=scene["plate"],
                           finger_low=scene["finger_low"], replay_plate_error_m=plate_err, **inspect(env))
                records.append(row)
                print(json.dumps(row), flush=True)
            _, _, _, done, _ = env.step(a)
            if done:
                break
    args.output.write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
