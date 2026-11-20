"""Scratch-pose screening and native OSC execution for the simulated Goal 5 Sawyer."""

import json

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


# Positive tilt inclines the grip site's +Z finger axis toward world -Y.
# Yaw is the world heading of the level tool X axis before the tilt.
CANDIDATES = ((32, 0), (48, 0), (32, 90), (48, 90), (32, -90),
              (48, -90), (32, 180), (18, 0), (0, 0), (-25, 0))
POS_TOL = 0.004
ANG_TOL = 0.045


def _bounded(vector, limit):
    return vector * min(1., limit / max(np.linalg.norm(vector), 1e-12))


def _rot(matrix):
    return Rotation.from_matrix(np.asarray(matrix).reshape(3, 3))


class _Planner:
    def __init__(self, env, geometry):
        self.m = env.sim.model._model
        self.d = mujoco.MjData(self.m)
        live = env.sim.data._data
        self.d.qpos[:] = live.qpos
        self.d.mocap_pos[:] = live.mocap_pos
        self.d.mocap_quat[:] = live.mocap_quat
        robot = env.robots[0]
        self.q = np.asarray(robot._ref_joint_pos_indexes)
        self.v = np.asarray(robot._ref_joint_vel_indexes)
        self.limits = self.m.jnt_range[robot._ref_joint_indexes].T
        self.site = robot.eef_site_id
        self.start = self.d.qpos[self.q].copy()
        self.d.qpos[robot._ref_gripper_joint_pos_indexes] = 0.  # closed signed slides, scratch only
        mujoco.mj_kinematics(self.m, self.d)
        self.xyz = self.d.site_xpos[self.site].copy()
        self.rot = _rot(self.d.site_xmat[self.site])
        world = np.concatenate([v @ self.d.geom_xmat[i].reshape(3, 3).T + self.d.geom_xpos[i]
                                for i, v in geometry.vertices["fingers"]])
        self.finger_local = self.rot.inv().apply(world - self.xyz)
        self.arm = set()
        self.cabinet = set()
        self.elbow = set()
        for i in range(self.m.ngeom):
            if not (self.m.geom_contype[i] or self.m.geom_conaffinity[i]):
                continue
            name = self.m.geom(i).name or ""
            body = self.m.body(int(self.m.geom_bodyid[i])).name or ""
            if body.startswith(("robot0_", "gripper0_")):
                self.arm.add(i)
            if body.startswith("wooden_cabinet_1") or name.startswith("wooden_cabinet_1"):
                self.cabinet.add(i)
            if name.startswith("robot0_right_l4") or body.startswith("robot0_right_l4"):
                self.elbow.add(i)
        known = env.sim.model.geom_name2id("robot0_right_l4_2")
        if known not in self.elbow or known not in self.arm or not self.cabinet:
            raise ValueError("Missing Sawyer elbow or wooden cabinet collision geoms")
        self.jp, self.jr = (np.zeros((3, self.m.nv)) for _ in range(2))

    def solve(self, target, orientation):
        # Continue the reset IK branch with bounded damped least-squares increments.
        for step in range(33):
            mujoco.mj_kinematics(self.m, self.d)
            p = target - self.d.site_xpos[self.site]
            w = (orientation * _rot(self.d.site_xmat[self.site]).inv()).as_rotvec()
            dp, dr = float(np.linalg.norm(p)), float(np.linalg.norm(w))
            if (dp < POS_TOL / 2 and dr < ANG_TOL / 2) or step == 32:
                return dp, dr
            mujoco.mj_comPos(self.m, self.d)
            mujoco.mj_jacSite(self.m, self.d, self.jp, self.jr, self.site)
            J = np.vstack((self.jp[:, self.v], 0.18 * self.jr[:, self.v]))
            dq = J.T @ np.linalg.solve(J @ J.T + 0.00015 * np.eye(6), np.r_[p, 0.18 * w])
            self.d.qpos[self.q] = np.clip(self.d.qpos[self.q] + _bounded(dq, 0.22), *self.limits)

    def clearance(self):
        mujoco.mj_fwdPosition(self.m, self.d)
        pairs = []
        for index in range(self.d.ncon):
            c = self.d.contact[index]
            a, b = int(c.geom1), int(c.geom2)
            if a in self.cabinet:
                a, b = b, a
            if a in self.arm and b in self.cabinet:
                pairs.append((a, b))
        # Exact convex-geom distance, capped at 10 cm. Bounding spheres only cull
        # pairs incapable of improving the current minimum.
        clearance = 0.10
        for a in self.elbow:
            for b in self.cabinet:
                gap = np.linalg.norm(self.d.geom_xpos[a] - self.d.geom_xpos[b])
                if gap - self.m.geom_rbound[a] - self.m.geom_rbound[b] < clearance:
                    gap = mujoco.mj_geomDistance(self.m, self.d, a, b, 0.10, None)
                    clearance = min(clearance, float(gap))
        return pairs, clearance

    def candidate(self, scene, goal, tilt, yaw):
        target_rot = (Rotation.from_euler("x", -tilt, degrees=True)
                      * Rotation.from_euler("z", yaw, degrees=True)
                      * Rotation.from_euler("x", 180, degrees=True))
        cloud = target_rot.apply(self.finger_local)
        low = cloud[:, 2].min()
        tip_xy = cloud[cloud[:, 2] <= low + 0.002, :2].mean(axis=0)
        plate = np.array(scene["plate"][:2])
        direction = goal - plate
        direction /= max(np.linalg.norm(direction), 1e-9)
        rim = direction * (scene["plate_radius"] + 0.008)
        seat = scene["table_z"] + 0.002 - low
        center = plate - tip_xy
        above = np.r_[center, max(scene["table_z"] + 0.15 - low, seat + 0.10)]
        halfway = Rotation.from_rotvec(0.5 * (target_rot * self.rot.inv()).as_rotvec()) * self.rot
        path = [("approach", (self.xyz + above) / 2, halfway), ("align", above, target_rot)]
        for label, z in (("descend", seat + 0.065), ("preseat", seat + 0.025), ("seat", seat)):
            path.append((label, np.r_[center, z], target_rot))
        for label, xy in (("engage", plate + rim), ("push_mid", (plate + goal) / 2 + rim),
                          ("push_goal", goal + rim)):
            path.append((label, np.r_[xy - tip_xy, seat], target_rot))
        self.d.qpos[self.q] = self.start
        worst_p = worst_r = 0.
        minimum = 0.10
        colliding = 0
        hit = set()
        for _, target, rot in path:
            previous = self.d.qpos[self.q].copy()
            dp, dr = self.solve(target, rot)
            worst_p, worst_r = max(worst_p, dp), max(worst_r, dr)
            end = self.d.qpos[self.q].copy()
            # Sample the joint-space connector as well as each destination.
            samples = max(2, int(np.ceil(np.max(np.abs(end - previous)) / 0.15)))
            for i in range(1, samples + 1):
                self.d.qpos[self.q] = previous + (end - previous) * (i / samples)
                pairs, gap = self.clearance()
                minimum = min(minimum, gap)
                colliding += bool(pairs)
                hit.update(pairs)
        error = max(worst_p / POS_TOL, worst_r / ANG_TOL)
        feasible = colliding == 0 and minimum > 0 and error <= 1
        names = [[self.m.geom(a).name, self.m.geom(b).name] for a, b in sorted(hit)]
        score = (int(not feasible), colliding, max(0., error - 1), -minimum, error)
        metrics = dict(tilt_deg=tilt, yaw_deg=yaw, feasible=feasible, score=list(score),
                       elbow_clearance_m=minimum, cabinet_contact_samples=colliding,
                       contact_pairs=names, max_xyz_error_m=worst_p, max_angle_error_rad=worst_r)
        return metrics, path


class ClearanceExpert:
    """Plan once after reset. Call action with fresh scene observations at 20 Hz."""

    def __init__(self, session, scene):
        if session.native is None or session.native.arm._env is None:
            raise ValueError("Use a reset direct native Sawyer Session in the simulator interpreter")
        env = session.native.arm._env
        planner = _Planner(env, session.native.geometry)
        self.live = env.sim.data._data
        self.site = planner.site
        self.model = planner.m
        self.debug_pair = (env.sim.model.geom_name2id("robot0_right_l4_2"),
                           env.sim.model.geom_name2id("wooden_cabinet_1_g9"))
        goal = np.asarray(scene["goal"][:2]) + 0.55 * np.asarray(scene["goal_half"][:2])
        self.points = [goal]
        ranked = sorted((planner.candidate(scene, goal, *c) for c in CANDIDATES),
                        key=lambda row: row[0]["score"])
        self.scores = [metrics for metrics, _ in ranked]
        best, self.path = ranked[0]
        self.chosen = best if best["feasible"] else None
        self.i = 0
        self.phase = self.path[0][0] if self.chosen else "no_feasible_path"
        self.seat_low = None
        self.integral_xyz = np.zeros(3)
        print(json.dumps(dict(candidates=self.scores, chosen=self.chosen)), flush=True)

    def measured_elbow_gap(self):
        """Live signed distance for the specific pair seen at controls 40, 65 and 84."""
        return float(mujoco.mj_geomDistance(self.model, self.live, *self.debug_pair, 0.10, None))

    def action(self, scene):
        if self.chosen is None:
            raise RuntimeError("No sampled cabinet-clear path met pose tolerances")
        xyz = np.asarray(scene["tcp"])
        current = _rot(self.live.site_xmat[self.site])
        while True:
            self.phase, target, orientation = self.path[self.i]
            dp = target - xyz
            w = (orientation * current.inv()).as_rotvec()
            seated = (scene["finger_low"] <= scene["table_z"] + 0.004 or
                      (scene["contact"] and scene["finger_low"] <= scene["rim_z"] + 0.003))
            if self.phase == "seat":
                reached = seated and np.linalg.norm(dp[:2]) < 0.015
            else:
                reached = np.linalg.norm(dp) < 0.016
            if self.seat_low is not None:
                dp[2] = self.seat_low - scene["finger_low"]
                reached = np.linalg.norm(dp[:2]) < 0.015
            if not reached or np.linalg.norm(w) > 0.08 or self.i + 1 == len(self.path):
                break
            if self.phase == "seat":
                self.seat_low = max(scene["table_z"] + 0.001, scene["finger_low"] - 0.002)
            self.i += 1
            self.integral_xyz.fill(0)
        # Native impedance produced a stationary 9.6 mm approach error on Sawyer.
        # Limit the integral correction and reset it at every waypoint.
        self.integral_xyz = np.clip(self.integral_xyz + 0.06 * dp, -0.014, 0.014)
        linear = _bounded(dp + self.integral_xyz, 0.014 if self.seat_low is not None else 0.035)
        return np.clip(np.r_[1.5 * linear / 0.05, _bounded(w, 0.25) / 0.5, 1.],
                       -1., 1.).astype(np.float32)
