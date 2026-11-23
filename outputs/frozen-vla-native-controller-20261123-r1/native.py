"""One BASE Panda task with independent factory-constructed native controllers."""
import hashlib
import os
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

from arm_env import CAMERAS, OSC_SCALE, _manifest, _public, _reset_wrapper
from libero.libero import benchmark
from libero.libero.envs.env_wrapper import ControlEnv
from libero_remote.environment.libero import _settle_environment
from libero_remote.reproducibility import numerical_state_fingerprint
from sim_reset import load_panda_row, scene_state

OSC = "OSC_POSE"
JOINT = "JOINT_POSITION"
JOINT_SCALE = 0.05
STATE = mujoco.mjtState.mjSTATE_INTEGRATION
DAMPING = 0.02


def digest(x):
    a = np.ascontiguousarray(x)
    return hashlib.sha256(str((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()


def delta(p0, r0, p1, r1):
    return np.r_[p1 - p0, Rotation.from_matrix(r1 @ r0.T).as_rotvec()]


def snapshot(env):
    m, d = env.sim.model._model, env.sim.data._data
    x = np.empty(mujoco.mj_stateSize(m, STATE))
    mujoco.mj_getState(m, d, x, STATE)
    r = env.robots[0]
    ori = getattr(r.controller, "goal_ori", None)
    clocks = {k: (v._time_since_last_sample, v._sampled, v._current_delay)
              for k, v in env.env._observables.items()}
    return dict(x=x, clocks=clocks, grip=r.gripper.current_action.copy(), goal_ori=None if ori is None else ori.copy(),
                timestep=env.env.timestep, cur_time=env.env.cur_time, perturb=env.env._perturbation_step)


def restore(env, snap):
    m, d = env.sim.model._model, env.sim.data._data
    mujoco.mj_setState(m, d, snap["x"], STATE)
    mujoco.mj_forward(m, d)
    # Recompute the derived fields, preserving the integration warm start.
    mujoco.mj_setState(m, d, snap["x"], STATE)
    assert env.env._observables.keys() == snap["clocks"].keys()
    for k, observable in env.env._observables.items():
        observable._time_since_last_sample, observable._sampled, observable._current_delay = snap["clocks"][k]
    env.env._obs_cache = {}
    robot = env.robots[0]
    robot.gripper.current_action = snap["grip"].copy()
    if robot.controller.name == OSC and snap["goal_ori"] is not None:
        robot.controller.goal_ori = snap["goal_ori"].copy()
    robot.controller.new_update = True
    env.env.timestep, env.env.cur_time = snap["timestep"], snap["cur_time"]
    env.env.done = False
    env.env._perturbation_step = snap["perturb"]


class Chain:
    """Use only encoder readings and the known fixed-base arm/grip-site geometry."""
    def __init__(self, env):
        robot = env.robots[0]
        self.m = env.sim.model._model
        self.d = mujoco.MjData(self.m)
        self.site = robot.eef_site_id
        self.qidx = robot._ref_joint_pos_indexes
        self.vidx = robot._ref_joint_vel_indexes
        ids = robot._ref_joint_indexes
        assert len(ids) == 7 and np.all(self.m.jnt_limited[ids])
        self.limits = self.m.jnt_range[ids].copy()

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

    def solve(self, q, matrix, desired, damping=DAMPING):
        # Normalize by componentwise native scales so angular and linear errors are comparable.
        low = np.clip((self.limits[:, 0] - q) / JOINT_SCALE, -1, 0)
        high = np.clip((self.limits[:, 1] - q) / JOINT_SCALE, 0, 1)
        result = lsq_linear(np.vstack([matrix, damping * np.eye(7)]),
                            np.r_[desired, np.zeros(7)], bounds=(low, high), method="bvls")
        if not result.success:
            raise RuntimeError(result.message)
        return np.clip(result.x, low, high)

    def ik(self, q, twist):
        return self.solve(q, self.jac(q) * JOINT_SCALE / OSC_SCALE[:, None], twist / OSC_SCALE)


class Pair:
    def __init__(self, suite, layout, seed, *, cameras=False):
        self.cameras = cameras
        self.envs = {}
        manifest = _manifest(Path(os.environ["CROSSARM_RELEASE"]) / "assets/task_manifest.json")
        entry = manifest.require(suite, 1, "BASE")
        self.horizon = entry["max_steps"]
        bddl = manifest.resolve_bddl(suite, 1, "BASE")
        self.bddl = Path(bddl)
        row = benchmark.get_benchmark_dict()[suite]().get_task_init_states(1)[layout]
        self.row_hash = numerical_state_fingerprint(row)
        try:
            for name in (OSC, JOINT):
                np.random.seed(seed)
                env = ControlEnv(bddl_file_name=str(bddl), robots=["Panda"], controller=name,
                                 gripper_types="PandaGripper", control_freq=20,
                                 use_camera_obs=cameras, has_offscreen_renderer=cameras,
                                 camera_heights=256, camera_widths=256, camera_names=list(CAMERAS))
                self.envs[name] = env
                _reset_wrapper(env, seed)
                load_panda_row(env, row)
                low, high = env.env.action_spec
                robot = env.robots[0]
                ctrl = robot.controller
                dim = 7 if name == OSC else 8
                assert ctrl.name == name and env.env.control_freq == 20
                assert np.array_equal(low, -np.ones(dim)) and np.array_equal(high, np.ones(dim))
                assert np.allclose(ctrl.output_max, OSC_SCALE if name == OSC else [JOINT_SCALE] * 7)
                assert np.allclose(ctrl.output_min, -ctrl.output_max) and ctrl.impedance_mode == "fixed"
                assert type(robot.gripper).__name__ == "PandaGripper"
                assert ctrl.eef_name == robot.gripper.important_sites["grip_site"]
                assert env.get_perturbation_plan()["case_id"] == "BASE"
            source, joint = self.envs.values()
            assert scene_state(source) == scene_state(joint)
            assert source.robots[0].robot_joints == joint.robots[0].robot_joints
            _settle_environment(source)
            for env in (source, joint):
                env.set_perturbation_seed(seed)
            self.initial = snapshot(source)
            self.chain = Chain(joint)
            self.prompt = source.language_instruction
            self.first = {}
            for name, env in self.envs.items():
                restore(env, self.initial)
                raw = env.begin_perturbation_rollout()
                assert not env.check_success() and env.language_instruction == self.prompt
                self.first[name] = _public(raw, 7) if cameras else raw
            assert np.array_equal(snapshot(source)["x"], snapshot(joint)["x"])
            assert scene_state(source) == scene_state(joint)
            self.reset_hash = numerical_state_fingerprint(source.get_sim_state())
            self.full_hash = digest(self.initial["x"])
            for key in ("robot0_joint_pos", "robot0_eef_pos", *(CAMERA_KEYS if cameras else ())):
                assert np.array_equal(self.first[OSC][key], self.first[JOINT][key]), key
            if cameras:
                assert all(np.array_equal(v, self.first[JOINT][k]) for k, v in self.first[OSC].items())
            self.opening = {k: snapshot(e) for k, e in self.envs.items()}
        except BaseException:
            self.close()
            raise

    def start(self, name):
        env = self.envs[name]
        restore(env, self.initial)
        raw = env.begin_perturbation_rollout()
        return _public(raw, 7) if self.cameras else raw

    def step(self, name, action):
        env = self.envs[name]
        lo, hi = env.env.action_spec
        a = np.asarray(action)
        if a.shape != lo.shape or not np.isfinite(a).all() or np.any(a < lo) or np.any(a > hi):
            raise ValueError(f"Illegal {name} native action: {a}")
        raw, reward, done, _ = env.step(a)
        success = bool(env.check_success())
        if float(reward) != float(success) or bool(done) != success:
            raise ValueError("Native BASE reward, done and BDDL predicate disagree")
        return (_public(raw, 7) if self.cameras else raw), float(reward)

    def close(self):
        for env in self.envs.values():
            env.close()
        self.envs.clear()


CAMERA_KEYS = ("agentview_image", "robot0_eye_in_hand_image")


class Decoder:
    def __init__(self, chain, q, method, calibration):
        self.chain = chain
        self.method = method
        self.calibration = calibration
        _, self.goal_ori = chain.fk(q)

    def action(self, q, canonical):
        x = np.asarray(canonical)
        assert x.shape == (7,) and np.all(np.abs(x) <= 1)
        if self.method == OSC:
            return x.copy()
        if self.method == "hold":
            return np.r_[np.zeros(7), x[-1]]
        _, ori = self.chain.fk(q)
        # Native OSC retains the previous orientation target on an exactly zero rotation command.
        if np.any(x[3:6]):
            dr = x[3:6] * OSC_SCALE[3:]
            self.goal_ori = Rotation.from_rotvec(dr).as_matrix() @ ori
        else:
            dr = Rotation.from_matrix(self.goal_ori @ ori.T).as_rotvec()
        twist = np.r_[x[:3] * OSC_SCALE[:3], dr]
        if self.method == "ik":
            a = self.chain.ik(q, twist)
        elif self.method == "calibrated":
            c = self.calibration
            # Both matrices and offset are one-control realized world motion in canonical scale units.
            desired = np.array(c["osc_matrix"]) @ (twist / OSC_SCALE) + c["offset"]
            a = self.chain.solve(q, np.array(c["joint_matrix"]), desired, damping=c["damping"])
        else:
            raise ValueError(self.method)
        return np.r_[a, x[-1]]
