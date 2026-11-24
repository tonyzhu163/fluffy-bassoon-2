"""CPU diagnostic: cross-arm native JOINT_POSITION decoders on archived Panda OSC proposals."""
import json
import math
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

from arm_env import OSC_SCALE, _register_arm, _reset_wrapper
from libero.libero.envs.env_wrapper import ControlEnv
from native import DAMPING, JOINT, JOINT_SCALE, Pair
from sim_reset import scene_state, tool_state, transfer_scene

ARCHIVE = Path("/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-native-controller-20261123-r1/gpu01")
SEEDS = {"libero_spatial": 171117, "libero_goal": 171118}
LAYOUTS = (0, 2, 3)
ARMS = ("Panda", "Kinova3", "IIWA", "UR5e")
DECODERS = ("ik_naive", "ik_dir_scaled", "ik_pd_vel", "ik_pose_track")

# Analytical 50ms critically damped PD constants for kp=50, T=0.05s
OMEGA = math.sqrt(50.0)
DT = 0.05
PD_ALPHA = 1.0 - (1.0 + OMEGA * DT) * math.exp(-OMEGA * DT)  # ~0.04964
PD_BETA = math.exp(-OMEGA * DT)  # ~0.70217


class GeneralChain:
    """Fixed-base kinematic chain for any d-DoF hinge arm (6 or 7 DoF, continuous or limited)."""

    def __init__(self, env):
        robot = env.robots[0]
        self.m = env.sim.model._model
        self.d = mujoco.MjData(self.m)
        self.site = robot.eef_site_id
        self.qidx = robot._ref_joint_pos_indexes
        self.vidx = robot._ref_joint_vel_indexes
        ids = robot._ref_joint_indexes
        self.dof = len(ids)
        bounded = self.m.jnt_limited[ids].astype(bool)
        self.limits = self.m.jnt_range[ids].copy()
        self.limits[~bounded] = [-np.inf, np.inf]

    def fk(self, q):
        self.d.qpos[self.qidx] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    def jac(self, q):
        self.fk(q)
        mujoco.mj_comPos(self.m, self.d)
        jp, jr = np.zeros((2, 3, self.m.nv))
        mujoco.mj_jacSite(self.m, self.d, jp, jr, self.site)
        return np.vstack([jp[:, self.vidx], jr[:, self.vidx]])

    def bounds(self, q):
        low = np.clip((self.limits[:, 0] - q) / JOINT_SCALE, -1.0, 0.0)
        high = np.clip((self.limits[:, 1] - q) / JOINT_SCALE, 0.0, 1.0)
        return low, high

    def ik_naive(self, q, twist):
        low, high = self.bounds(q)
        mat = self.jac(q) * JOINT_SCALE / OSC_SCALE[:, None]
        des = twist / OSC_SCALE
        res = lsq_linear(
            np.vstack([mat, DAMPING * np.eye(self.dof)]),
            np.r_[des, np.zeros(self.dof)],
            bounds=(low, high),
            method="bvls",
        )
        return np.clip(res.x, low, high)

    def _dls_dir_scaled(self, q, mat, des, damping=DAMPING):
        low, high = self.bounds(q)
        # Unconstrained damped least squares
        A = mat.T @ mat + (damping ** 2) * np.eye(self.dof)
        b = mat.T @ des
        u = np.linalg.solve(A, b)
        # Uniformly scale to preserve Cartesian direction when bounds bind
        scale = 1.0
        for i in range(self.dof):
            if u[i] > high[i] and u[i] > 1e-12:
                scale = min(scale, max(0.0, high[i]) / u[i])
            elif u[i] < low[i] and u[i] < -1e-12:
                scale = min(scale, max(0.0, -low[i]) / (-u[i]))
        return np.clip(u * scale, low, high)

    def ik_dir_scaled(self, q, twist):
        mat = self.jac(q) * JOINT_SCALE / OSC_SCALE[:, None]
        des = twist / OSC_SCALE
        return self._dls_dir_scaled(q, mat, des)

    def ik_pd_vel(self, q, v, twist, osc_gain=0.35):
        """Compensate PD velocity carryover: J*(alpha*delta + beta*T*v) = osc_gain*twist."""
        J = self.jac(q)
        # Desired 50ms Cartesian displacement minus passive velocity carryover
        passive_dx = J @ (PD_BETA * DT * v)
        rem_dx = osc_gain * twist - passive_dx
        mat = (J * (PD_ALPHA * JOINT_SCALE)) / OSC_SCALE[:, None]
        des = rem_dx / OSC_SCALE
        return self._dls_dir_scaled(q, mat, des, damping=DAMPING * 0.5)


class CrossArmDecoder:
    def __init__(self, chain, q, method):
        self.chain = chain
        self.method = method
        p0, r0 = chain.fk(q)
        self.goal_pos = p0.copy()
        self.goal_ori = r0.copy()

    def action(self, q, v, canonical):
        x = np.asarray(canonical, dtype=np.float64)
        pos, ori = self.chain.fk(q)
        if np.any(x[3:6]):
            dr = x[3:6] * OSC_SCALE[3:]
            self.goal_ori = Rotation.from_rotvec(dr).as_matrix() @ ori
        else:
            dr = Rotation.from_matrix(self.goal_ori @ ori.T).as_rotvec()
        twist = np.r_[x[:3] * OSC_SCALE[:3], dr]

        if self.method == "ik_naive":
            u = self.chain.ik_naive(q, twist)
        elif self.method == "ik_dir_scaled":
            u = self.chain.ik_dir_scaled(q, twist)
        elif self.method == "ik_pd_vel":
            u = self.chain.ik_pd_vel(q, v, twist)
        elif self.method == "ik_pose_track":
            # Maintain a bounded persistent Cartesian pose target (mirrors OSC_POSE semantics)
            # Clip target lead so it stays within 1.5x OSC_SCALE of current pose
            self.goal_pos = pos + np.clip((self.goal_pos - pos) + x[:3] * OSC_SCALE[:3] * 0.35, -0.075, 0.075)
            err_pos = self.goal_pos - pos
            err_ori = Rotation.from_matrix(self.goal_ori @ ori.T).as_rotvec()
            pose_twist = np.r_[err_pos, err_ori]
            u = self.chain.ik_pd_vel(q, v, pose_twist, osc_gain=1.0)
        else:
            raise ValueError(self.method)
        return np.r_[u, x[-1]]


def build_arm_env(bddl, arm, seed):
    if arm != "Panda":
        _register_arm(arm)
    np.random.seed(seed)
    env = ControlEnv(
        bddl_file_name=str(bddl),
        robots=[arm],
        controller=JOINT,
        gripper_types="PandaGripper",
        control_freq=20,
        use_camera_obs=False,
        has_offscreen_renderer=False,
    )
    _reset_wrapper(env, seed)
    return env


def run_case(out_dir: Path):
    rows = []
    for suite, base_seed in SEEDS.items():
        for layout in LAYOUTS:
            seed = base_seed + 100 * layout
            osc_tape = json.loads((ARCHIVE / f"{suite}-{layout}-OSC_POSE.json").read_text())
            cmds = np.array(osc_tape["canonical"], dtype=np.float64)
            # Build paired Panda source to get exact settled scene & OSC reference TCP path
            pair = Pair(suite, layout, seed, cameras=False)
            try:
                # Replay OSC_POSE on Panda to collect exact reference TCP positions
                pair.start("OSC_POSE")
                scratch_src = mujoco.MjData(pair.envs["OSC_POSE"].sim.model._model)
                ref_xyz = [tool_state(pair.envs["OSC_POSE"], scratch_src)["tcp_xyz"]]
                for c in cmds:
                    _, r = pair.step("OSC_POSE", c)
                    ref_xyz.append(tool_state(pair.envs["OSC_POSE"], scratch_src)["tcp_xyz"])
                    if r > 0:
                        break
                ref_xyz = np.array(ref_xyz)
                # Reset source env to initial settled state for transfer_scene
                pair.start("OSC_POSE")
                src_env = pair.envs["OSC_POSE"]
                src_scene = scene_state(src_env)

                for arm in ARMS:
                    env = build_arm_env(pair.bddl, arm, seed)
                    try:
                        chain = GeneralChain(env)
                        scratch = mujoco.MjData(env.sim.model._model)
                        for method in DECODERS:
                            _reset_wrapper(env, seed)
                            transfer_scene(src_env, env, src_scene)
                            env.robots[0].gripper.current_action = src_env.robots[0].gripper.current_action.copy()
                            env.set_perturbation_seed(seed)
                            env.begin_perturbation_rollout()
                            sensors = env.env._observables
                            q0 = sensors["robot0_joint_pos"]._sensor({})
                            dec = CrossArmDecoder(chain, q0, method)
                            xyzs = [tool_state(env, scratch)["tcp_xyz"]]
                            sats = 0
                            success = False
                            controls = 0
                            for t, c in enumerate(cmds):
                                q = sensors["robot0_joint_pos"]._sensor({})
                                v = sensors["robot0_joint_vel"]._sensor({})
                                act = dec.action(q, v, c)
                                if np.any(np.abs(act[:-1]) >= 0.999):
                                    sats += 1
                                _, rew, done, _ = env.step(act)
                                controls += 1
                                xyzs.append(tool_state(env, scratch)["tcp_xyz"])
                                if rew > 0 or done:
                                    success = bool(env.check_success())
                                    break
                            xyzs = np.array(xyzs)
                            n_cmp = min(len(xyzs), len(ref_xyz))
                            err_mm = np.linalg.norm(xyzs[:n_cmp] - ref_xyz[:n_cmp], axis=1) * 1000.0
                            row = dict(
                                suite=suite,
                                layout=layout,
                                arm=arm,
                                dof=chain.dof,
                                method=method,
                                success=success,
                                controls=controls,
                                src_controls=len(cmds),
                                sat_frac=sats / max(1, controls),
                                mean_tcp_err_mm=float(np.mean(err_mm)),
                                max_tcp_err_mm=float(np.max(err_mm)),
                                final_tcp_err_mm=float(err_mm[-1]),
                            )
                            rows.append(row)
                            print(
                                f"{suite}/{layout} {arm:7s} {method:14s} succ={int(success)} "
                                f"ctrl={controls:3d}/{len(cmds):3d} sat={row['sat_frac']:.2f} "
                                f"tcp_err={row['mean_tcp_err_mm']:5.1f}mm",
                                flush=True,
                            )
                    finally:
                        env.close()
            finally:
                pair.close()

    # Aggregate summary by arm x method
    summary = {}
    for arm in ARMS:
        summary[arm] = {}
        for method in DECODERS:
            sub = [r for r in rows if r["arm"] == arm and r["method"] == method]
            summary[arm][method] = dict(
                successes=sum(r["success"] for r in sub),
                total=len(sub),
                mean_sat=float(np.mean([r["sat_frac"] for r in sub])),
                mean_tcp_err_mm=float(np.mean([r["mean_tcp_err_mm"] for r in sub])),
                mean_controls=float(np.mean([r["controls"] for r in sub])),
            )
    (out_dir / "results.json").write_text(json.dumps(dict(summary=summary, rows=rows), indent=2) + "\n")
    print("\n=== SUMMARY (successes/6, mean TCP err mm, sat frac) ===", flush=True)
    for arm in ARMS:
        line = f"{arm:8s} |"
        for method in DECODERS:
            s = summary[arm][method]
            line += f" {method}: {s['successes']}/{s['total']} ({s['mean_tcp_err_mm']:5.1f}mm, sat={s['mean_sat']:.2f}) |"
        print(line, flush=True)


if __name__ == "__main__":
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    run_case(out_dir)
    print(f"Elapsed: {time.time() - t0:.1f}s", flush=True)
