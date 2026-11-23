"""Independent FK/contact readback and a necessary fixed-orientation reach bound."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from arm_env import ArmEnv
from feasibility import ARMS, TURN, install
from geometry import Geometry
from sim_reset import POSITION_TOL_M, ROTATION_TOL_RAD


def chain_points(g, shifts):
    return g.d.xanchor[g.joints] + shifts[:, None] * g.d.xaxis[g.joints]


def lengths(points):
    return np.linalg.norm(np.diff(points, axis=0), axis=1)


def reach_bound(g, goal):
    """Every hinge-axis point belongs to both neighboring links. The polygonal chain's length is fixed."""
    g.d.qpos[g.adr] = g.qreset
    g.forward()
    reference = g.tcp()
    # Offsets only choose the points for a necessary inequality. The inequality holds for every choice.
    origins = g.d.xanchor[g.joints].copy()
    axes = g.d.xaxis[g.joints].copy()

    def measure(offsets):
        pts = origins + offsets[:, None] * axes
        tcp_to_last = reference[:3, :3].T @ (pts[-1] - reference[:3, 3])
        required = goal[:3, 3] + goal[:3, :3] @ tcp_to_last
        distance = np.linalg.norm(required - pts[0])
        chain = lengths(pts).sum()
        rotation_slack = 2 * np.linalg.norm(tcp_to_last) * np.sin(ROTATION_TOL_RAD / 2)
        return float(distance), float(chain), float(rotation_slack)

    def objective(offsets):
        dist, chain, slack = measure(offsets)
        return chain + slack - dist

    init = np.zeros(len(g.joints))
    fit = minimize(objective, init, method="SLSQP", bounds=[(-1., 1.)]*len(init),
                   options={"ftol": 1e-12, "maxiter": 240})
    offsets = fit.x if objective(fit.x) < objective(init) else init
    distance, chain, slack = measure(offsets)
    pts = chain_points(g, offsets)
    seg = lengths(pts)
    tool_offset = reference[:3, :3].T @ (pts[-1] - reference[:3, 3])
    # Check each adjacent length and both rigid endpoint constraints at independent FK configurations.
    max_error = 0.
    for q in [g.qreset + np.linspace(-.31, .47, len(init)), g.qreset - np.linspace(.29, .53, len(init))]:
        g.d.qpos[g.adr] = np.clip(q, g.lo, g.hi)
        g.forward()
        test_pts = chain_points(g, offsets)
        test_tcp = g.tcp()
        errs = [np.max(abs(lengths(test_pts)-seg)), np.linalg.norm(test_pts[0]-pts[0]),
                np.linalg.norm(test_tcp[:3,:3].T @ (test_pts[-1]-test_tcp[:3,3])-tool_offset)]
        max_error = max(max_error, *errs)
    if max_error > 1e-10:
        raise ValueError("Hinge point certificate changed under FK")
    return dict(axis_offsets_m=offsets.tolist(), required_end_distance_m=distance, chain_length_m=chain,
                rotation_slack_m=slack, lower_bound_position_error_m=max(0., distance-chain-slack),
                exceeds_waypoint_tolerance=bool(distance-chain-slack > POSITION_TOL_M + 1e-9),
                max_rigidity_readback_error_m=float(max_error))


def inspect(case_file):
    data = json.loads(case_file.read_text())
    rows = []
    counts = {}
    for arm in ARMS:
        record = data["native_resets"][arm]
        with ArmEnv(arm, record["suite"], record["task_id"]) as env:
            actual = env.reset(record["init_state_id"], record["seed"])
            assert actual["environment_reproducibility"] == record
            g = Geometry(env._env)
            counts[arm] = dict()
            for role, body, geom in g.geoms:
                if role in ("arm", "hand", "finger"):
                    idx = g.m.geom(geom).id if not geom.startswith("geom#") else int(geom[5:])
                    if g.m.geom_contype[idx] or g.m.geom_conaffinity[idx]:
                        counts[arm][body] = counts[arm].get(body, 0) + 1
            for donor in data["demonstrations"]:
                for waypoint in donor["waypoints"]:
                    install(g, waypoint["stage"])
                    for mode, sols in waypoint["search"][arm].items():
                        goal = np.asarray(waypoint["goal"])
                        if mode == "half_turn":
                            goal = goal @ TURN
                        for sol in sols:
                            g.d.qpos[g.adr] = sol["q"]
                            g.forward()
                            tcp = g.tcp()
                            dp = np.linalg.norm(tcp[:3,3]-goal[:3,3])
                            dr = Rotation.from_matrix(tcp[:3,:3].T @ goal[:3,:3]).magnitude()
                            assert abs(dp-sol["pos_error_m"]) < 1e-12 and abs(dr-sol["rot_error_rad"]) < 1e-12
                            assert g.contacts(waypoint["body"]) == sol["contacts"]
                            assert np.all(g.d.qpos[g.adr] >= g.lo) and np.all(g.d.qpos[g.adr] <= g.hi)
                        row = dict(arm=arm, donor=donor["demo"], body=waypoint["body"], phase=waypoint["phase"],
                                   mode=mode, candidates_verified=len(sols))
                        if not any(s["pose_ok"] for s in sols):
                            row["bound"] = reach_bound(g, goal)
                        rows.append(row)
    return dict(case=data["case"], collision_enabled_geoms_by_link=counts, rows=rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cases", type=Path, nargs="+")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    assert len(os.sched_getaffinity(0)) <= 2
    args.out.mkdir(exist_ok=True, parents=True)
    for path in args.cases:
        report = inspect(path)
        (args.out / (path.stem + "-readback.json")).write_text(json.dumps(report, indent=1, allow_nan=False)+"\n")
        print(path.stem, sum(r["candidates_verified"] for r in report["rows"]), flush=True)
