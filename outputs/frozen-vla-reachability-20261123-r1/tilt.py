"""Allow wrist tilt at the identical TCP position on Kinova's missed recorded orientations."""
import argparse
import json
import os
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from arm_env import ArmEnv
from feasibility import install
from geometry import Geometry
from sim_reset import POSITION_TOL_M


def closest_attitude(g, goal, seed, intended):
    def pose_and_jac(q):
        g.d.qpos[g.adr] = q
        mujoco.mj_kinematics(g.m, g.d)
        mujoco.mj_comPos(g.m, g.d)
        mujoco.mj_jacSite(g.m, g.d, g.jacp, g.jacr, g.site)
        return g.tcp()

    def objective(q):
        pose = pose_and_jac(q)
        tilt = Rotation.from_matrix(pose[:3,:3] @ goal[:3,:3].T).as_rotvec()
        return float(tilt @ tilt)/2, tilt @ g.jacr[:, g.dofs]

    def position(q):
        return pose_and_jac(q)[:3,3] - goal[:3,3]

    def pos_jac(q):
        pose_and_jac(q)
        return g.jacp[:, g.dofs].copy()

    fit = minimize(objective, seed, jac=True, bounds=list(zip(g.lo, g.hi)), method="SLSQP",
                   constraints=[dict(type="eq", fun=position, jac=pos_jac)],
                   options={"maxiter": 150, "ftol": 1e-13})
    g.d.qpos[g.adr] = fit.x
    g.forward()
    tcp = g.tcp()
    dp = np.linalg.norm(tcp[:3,3] - goal[:3,3])
    tilt = Rotation.from_matrix(tcp[:3,:3] @ goal[:3,:3].T).as_rotvec()
    return dict(q=fit.x.tolist(), tcp=tcp.tolist(), position_error_m=float(dp),
                position_ok=bool(dp <= POSITION_TOL_M), actual_rotation_correction_world_rad=tilt.tolist(),
                angle_deg=float(np.rad2deg(np.linalg.norm(tilt))), solver_status=int(fit.status),
                min_limited_margin_rad=float(np.minimum(g.hi-fit.x, fit.x-g.lo).min()), contacts=g.contacts(intended))


def run(path, out):
    d = json.loads(path.read_text())
    reset = d["native_resets"]["Kinova3"]
    rows = []
    with ArmEnv("Kinova3", reset["suite"], reset["task_id"]) as env:
        assert env.reset(reset["init_state_id"], reset["seed"])["environment_reproducibility"] == reset
        g = Geometry(env._env)
        for demo in d["demonstrations"]:
            for w in demo["waypoints"]:
                fits = w["search"]["Kinova3"]["recorded"]
                if any(fit["pose_ok"] for fit in fits):
                    continue
                install(g, w["stage"])
                sols = [closest_attitude(g, np.asarray(w["goal"]), fit["q"], w["body"]) for fit in fits]
                rows.append(dict(donor=demo["demo"], phase=w["phase"], body=w["body"], frame=w["frame"], fits=sols))
    (out / (path.stem+"-tilt.json")).write_text(json.dumps(rows, indent=1, allow_nan=False)+"\n")
    print(d["case"], len(rows), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cases", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert len(os.sched_getaffinity(0)) <= 2
    args.out.mkdir(exist_ok=True, parents=True)
    for f in args.cases:
        run(f, args.out)
