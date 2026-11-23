"""Static FK, contact classification and bounded pose IK in the compiled native scene."""
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from sim_reset import POSITION_TOL_M, ROTATION_TOL_RAD


def transform(xyz, matrix):
    pose = np.eye(4)
    pose[:3, :3] = np.reshape(matrix, (3, 3))
    pose[:3, 3] = xyz
    return pose


def body_pose(model, data, name):
    i = model.body(name).id
    return transform(data.xpos[i], data.xmat[i])


def put_free(model, data, joint, pose):
    i = model.joint(joint).qposadr[0]
    data.qpos[i:i+3] = pose[:3, 3]
    data.qpos[i+3:i+7] = Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]]


class Geometry:
    def __init__(self, native, model=None):
        self.m = native.sim.model._model if model is None else model
        robot = native.robots[0]
        self.d = mujoco.MjData(self.m)
        self.d.qpos[:] = native.sim.data.qpos
        self.d.mocap_pos[:] = native.sim.data.mocap_pos
        self.d.mocap_quat[:] = native.sim.data.mocap_quat
        self.site = robot.eef_site_id
        self.joints = np.asarray(robot._ref_joint_indexes)
        self.adr = self.m.jnt_qposadr[self.joints]
        self.dofs = self.m.jnt_dofadr[self.joints]
        self.fingers = np.asarray(robot._ref_gripper_joint_pos_indexes)
        self.limited = self.m.jnt_limited[self.joints].astype(bool)
        lo, hi = self.m.jnt_range[self.joints].T.copy()
        lo[~self.limited], hi[~self.limited] = -np.inf, np.inf
        self.lo, self.hi = lo, hi
        self.qreset = self.d.qpos[self.adr].copy()
        self.initial_qpos = self.d.qpos.copy()
        self.geoms = []
        for i, bid in enumerate(self.m.geom_bodyid):
            body = self.m.body(bid).name or f"body#{bid}"
            if body.startswith("gripper0_"):
                role = "finger" if "finger" in body else "hand"
            elif body.startswith("robot0_"):
                role = "arm" if self.m.body_weldid[bid] != 0 else "base"
            elif body.startswith("mount0_"):
                role = "base"
            else:
                role = "scene"
            self.geoms.append((role, body, self.m.geom(i).name or f"geom#{i}"))
        self.jacp = np.empty((3, self.m.nv))
        self.jacr = self.jacp.copy()
        self.forward()

    def forward(self):
        mujoco.mj_fwdPosition(self.m, self.d)

    def tcp(self):
        return transform(self.d.site_xpos[self.site], self.d.site_xmat[self.site])

    def contacts(self, intended):
        """Minimum signed distance for each enabled, penetrating robot-involving pair."""
        pairs = {}
        for c in self.d.contact:
            if c.dist >= 0 or c.exclude != 0:
                continue
            a, b = sorted((self.geoms[c.geom1], self.geoms[c.geom2]), key=lambda v: v[0] == "scene")
            roles = (a[0], b[0])
            if not any(r in ("arm", "finger", "hand") for r in roles):
                continue
            if "scene" not in roles:
                kind = "intra_gripper" if all(r in ("finger", "hand") for r in roles) else "arm_self_or_base"
            elif a[0] == "arm":
                kind = "arm_scene"
            elif a[0] == "finger" and b[1] == intended:
                kind = "intended_finger"
            else:
                kind = "other_hand_scene"
            key = (kind, a[1], a[2], b[1], b[2])
            pairs[key] = min(pairs.get(key, 0), float(c.dist))
        return [dict(kind=k, body1=a, geom1=g, body2=b, geom2=h, distance_m=dist)
                for (k, a, g, b, h), dist in sorted(pairs.items())]

    def solve(self, goal, seed, intended):
        # d log(R R_goal^T) = J_left(phi)^-1 omega_world for a world angular perturbation.
        def error(q):
            self.d.qpos[self.adr] = q
            mujoco.mj_kinematics(self.m, self.d)
            pose = self.tcp()
            phi = Rotation.from_matrix(pose[:3, :3] @ goal[:3, :3].T).as_rotvec()
            return np.r_[(pose[:3, 3] - goal[:3, 3]) / POSITION_TOL_M, phi / ROTATION_TOL_RAD]

        def jac(q):
            phi = error(q)[3:] * ROTATION_TOL_RAD
            mujoco.mj_comPos(self.m, self.d)
            mujoco.mj_jacSite(self.m, self.d, self.jacp, self.jacr, self.site)
            x, y, z = phi
            skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
            theta = np.linalg.norm(phi)
            coef = 1 / 12 if theta < 1e-4 else (1 - 0.5 * theta / np.tan(0.5 * theta)) / theta**2
            left_inv = np.eye(3) - skew / 2 + coef * skew @ skew
            return np.vstack((self.jacp[:, self.dofs] / POSITION_TOL_M,
                              left_inv @ self.jacr[:, self.dofs] / ROTATION_TOL_RAD))

        seed = np.clip(seed, self.lo + 1e-8, self.hi - 1e-8)
        fit = least_squares(error, seed, jac=jac, bounds=(self.lo, self.hi), tr_solver="lsmr",
                            max_nfev=120, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        self.d.qpos[self.adr] = fit.x
        self.forward()
        pose = self.tcp()
        dp = float(np.linalg.norm(pose[:3, 3] - goal[:3, 3]))
        dr = float(Rotation.from_matrix(pose[:3, :3] @ goal[:3, :3].T).magnitude())
        margins = np.minimum(fit.x - self.lo, self.hi - fit.x)
        if np.any(margins[self.limited] < -1e-12):
            raise ValueError("Solver violated a modeled joint limit")
        return dict(q=fit.x.tolist(), pos_error_m=dp, rot_error_rad=dr, nfev=fit.nfev,
                    pose_ok=dp <= POSITION_TOL_M and dr <= ROTATION_TOL_RAD,
                    min_limited_margin_rad=float(np.min(margins[self.limited])),
                    nearest_limit=self.m.joint(self.joints[np.argmin(margins)]).name,
                    contacts=self.contacts(intended))
