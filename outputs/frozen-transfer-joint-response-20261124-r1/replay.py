"""Replay known Panda proposals through actual Panda and Kinova3 joint servos."""
import hashlib
import json
import os
import platform
from importlib.metadata import version
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from arm_env import ARM_XML_SHA, CAMERAS, GRIPPER_XML_SHA, _register_arm, _reset_wrapper
from libero.libero.envs.env_wrapper import ControlEnv
from libero_remote.reproducibility import canonical_fingerprint, numerical_state_fingerprint
from native import Chain, Decoder, JOINT, Pair, digest
from sim_reset import pose_errors, scene_body_poses, scene_state, tool_state, transfer_scene

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "frozen-vla-native-controller-20261123-r1"
ARCHIVE = Path("/mowenjie/fluffy-bassoon-2/outputs") / PRIOR.name / "gpu01"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    def scalar(value):
        if isinstance(value, (np.ndarray, np.generic)):
            return value.tolist()
        raise TypeError(type(value).__name__)
    path.write_text(json.dumps(data, indent=2, allow_nan=False, default=scalar) + "\n")


class TargetChain(Chain):
    """The prior seven-axis solver with the four continuous Kinova hinges unbounded."""
    def __init__(self, env):
        robot = env.robots[0]
        self.m = env.sim.model._model
        self.d = mujoco.MjData(self.m)
        self.site = robot.eef_site_id
        self.qidx, self.vidx = robot._ref_joint_pos_indexes, robot._ref_joint_vel_indexes
        ids = robot._ref_joint_indexes
        bounded = self.m.jnt_limited[ids].astype(bool)
        assert list(bounded) == [False, True, False, True, False, True, False]
        self.limits = self.m.jnt_range[ids].copy()
        self.limits[~bounded] = [-np.inf, np.inf]


def identity(env, arm):
    robot = env.robots[0]
    ctrl = robot.controller
    model = env.sim.model
    assert type(robot.robot_model).__name__ in ["Mounted"+arm, "OnTheGround"+arm]
    assert sha(robot.robot_model.file) == ARM_XML_SHA[arm]
    assert sha(robot.gripper.file) == GRIPPER_XML_SHA and type(robot.gripper).__name__ == "PandaGripper"
    assert len(robot.robot_joints) == ctrl.control_dim == 7
    assert np.all(model.jnt_type[robot._ref_joint_indexes] == mujoco.mjtJoint.mjJNT_HINGE)
    assert ctrl.name == JOINT and ctrl.impedance_mode == "fixed" and ctrl.interpolator is None
    assert ctrl.eef_name == robot.gripper.important_sites["grip_site"]
    assert env.env.control_freq == 20 and np.all(ctrl.kp == 50) and np.allclose(ctrl.kd, 2*np.sqrt(ctrl.kp))
    lo, hi = env.env.action_spec
    assert np.array_equal(lo, -np.ones(8)) and np.array_equal(hi, np.ones(8))
    assert np.all(ctrl.output_max == .05) and np.all(ctrl.output_min == -.05)
    assert env.get_perturbation_plan()["case_id"] == "BASE"
    return dict(arm_class=type(robot.robot_model).__name__, arm_xml=sha(robot.robot_model.file),
                gripper_xml=sha(robot.gripper.file), joints=robot.robot_joints,
                limited=model.jnt_limited[robot._ref_joint_indexes], ctrl=ctrl.name,
                kp=ctrl.kp, kd=ctrl.kd, input_box=[lo, hi], output_rad=ctrl.output_max,
                torque_ranges=model.actuator_ctrlrange[robot._ref_joint_actuator_indexes],
                timestep=model.opt.timestep, frequency=env.env.control_freq)


def boundary(env, scratch, obs):
    robot = env.robots[0]
    sensors = env.env._observables
    q = sensors["robot0_joint_pos"]._sensor({})
    v = sensors["robot0_joint_vel"]._sensor({})
    actual = tool_state(env, scratch)
    return dict(q=q, v=v, xyz=actual["tcp_xyz"], rot=actual["tcp_matrix"],
                emitted_q=obs["robot0_joint_pos"].copy(), emitted_v=obs["robot0_joint_vel"].copy(),
                emitted_xyz=obs["robot0_eef_pos"].copy(), time=float(env.sim.data.time),
                fingers=env.sim.data.qpos[robot._ref_gripper_joint_pos_indexes].copy())


def read_contacts(env, robot_geoms):
    data = env.sim.data._data
    return sum((c.geom1 in robot_geoms) != (c.geom2 in robot_geoms) for c in data.contact)


def rollout(out, arm, env, chain, obs, tape, base):
    schema = identity(env, arm)
    scratch = mujoco.MjData(env.sim.model._model)
    kin = boundary(env, scratch, obs)
    decoder = Decoder(chain, obs["robot0_joint_pos"].astype(np.float32), "ik", {})
    robot = env.robots[0]
    names = set(robot.robot_model.contact_geoms) | set(robot.gripper.contact_geoms)
    robot_geoms = {env.sim.model.geom_name2id(n) for n in names}
    assert robot_geoms and not env.check_success()
    rows = {k: [v] for k, v in kin.items()}
    J, actions, rewards, contacts, clipped_torque, decoded_difference = [], [], [], [], [], []
    cmds = np.array(tape["canonical"], dtype=np.float32)
    archived = np.array(tape["native"])
    for t, proposal in enumerate(cmds):
        jac = chain.jac(kin["q"])
        pred_action = decoder.action(obs["robot0_joint_pos"].astype(np.float32), proposal)
        action = archived[t].copy() if arm == "Panda" else pred_action
        assert action.shape == (8,) and np.all(np.isfinite(action)) and np.all(abs(action) <= 1)
        assert action[-1] == proposal[-1]
        if arm == "Panda":
            decoded_difference.append(float(np.max(abs(pred_action-action))))
        obs, reward, done, _ = env.step(action)
        assert bool(done) == bool(env.check_success()) and float(reward) == float(done)
        kin = boundary(env, scratch, obs)
        assert np.isclose(kin["time"] - rows["time"][-1], .05, atol=1e-12, rtol=0)
        for k, v in kin.items():
            rows[k].append(v)
        J.append(jac)
        actions.append(action)
        rewards.append(float(reward))
        contacts.append(read_contacts(env, robot_geoms))
        ids = robot._ref_joint_actuator_indexes
        torque = env.sim.data.ctrl[ids]
        limits = env.sim.model.actuator_ctrlrange[ids]
        clipped_torque.append(np.any((torque <= limits[:, 0]) | (torque >= limits[:, 1])))
        if done:
            break
    n = len(actions)
    # The archived source proposal stays fixed even if later CPU contacts diverge.
    archive_reward_match = rewards == tape["rewards"] if arm == "Panda" else None
    initial_q = np.array(rows["q"][0])
    J0 = J[0]
    h = 1e-6
    diffs = []
    for e in np.eye(7):
        p0, r0 = chain.fk(initial_q - h*e)
        p1, r1 = chain.fk(initial_q + h*e)
        diffs.append(np.r_[p1-p0, Rotation.from_matrix(r1 @ r0.T).as_rotvec()]/(2*h))
    fd = np.max(abs(np.array(diffs).T - J0))
    assert fd < 1e-7
    arrays = dict(**rows, J=J, actions=actions, proposal=cmds[:n], rewards=rewards,
                  external_contacts=contacts, torque_limited=clipped_torque)
    assert all(np.isfinite(a).all() for a in arrays.values())
    np.savez_compressed(out/f"{base}-{arm}.npz", **arrays)
    return dict(identity=schema, controls=n, success=bool(rewards[-1]), jacobian_fd_max=fd,
                archive_action_max_abs=max(decoded_difference) if decoded_difference else None,
                archive_native_reward_match=archive_reward_match,
                q_clock_max_rad=np.max(abs(np.subtract(rows["q"], rows["emitted_q"]))),
                velocity_clock_max_rad_s=np.max(abs(np.subtract(rows["v"], rows["emitted_v"]))),
                tcp_clock_max_m=np.max(np.linalg.norm(np.subtract(rows["xyz"], rows["emitted_xyz"]),axis=1)),
                actual_q_range_rad=np.ptp(rows["q"], axis=0), raw_sha256=sha(out/f"{base}-{arm}.npz"))


def collect(out):
    _register_arm("Kinova3")
    assert version("robosuite") == "1.4.0" and mujoco.__version__ == "3.2.3"
    recorded = json.loads((PRIOR/"evidence/gpu01/readback.json").read_text())
    cpu_panel = json.loads((PRIOR/"evidence/panel/panel.json").read_text())["cases"]
    cpu_states = {(r["suite"], r["layout"]): r["state"] for r in cpu_panel}
    for case in recorded["cases"]:
        suite, layout, seed = (case[k] for k in ("suite", "layout", "seed"))
        key = f"{suite}-{layout}"
        path = ARCHIVE/f"{key}-ik.json"
        assert sha(path) == case["methods"]["ik"]["raw_file_sha256"]
        tape = json.loads(path.read_text())
        assert tape["case"] == dict(suite=suite, layout=layout, seed=seed) and not tape["fixture"]
        assert tape["schema"]["controller"] == JOINT and len(tape["native"]) == tape["controls"]
        pair = Pair(suite, layout, seed)
        target = None
        try:
            assert pair.row_hash == tape["meta"]["row"] and pair.prompt == tape["meta"]["prompt"]
            assert pair.full_hash == cpu_states[suite, layout], (key, pair.full_hash, cpu_states[suite, layout])
            source = pair.envs[JOINT]
            src_obs = pair.start(JOINT)
            proprio = {k: digest(src_obs[k.removesuffix("_xyzw")].astype(np.float32))
                      for k in tape["public"] if "image" not in k}
            assert len(proprio) == 5 and all(v == tape["public"][k] for k, v in proprio.items())
            np.random.seed(seed)
            target = ControlEnv(bddl_file_name=str(pair.bddl), robots=["Kinova3"], controller=JOINT,
                                gripper_types="PandaGripper", control_freq=20, has_offscreen_renderer=False,
                                use_camera_obs=False, camera_names=list(CAMERAS))
            _reset_wrapper(target, seed)
            align = transfer_scene(source, target, scene_state(source))
            target.robots[0].gripper.current_action = source.robots[0].gripper.current_action.copy()
            target.set_perturbation_seed(seed)
            tgt_obs = target.begin_perturbation_rollout()
            chain = TargetChain(target)
            assert target.language_instruction == pair.prompt and scene_state(target) == scene_state(source)
            error = pose_errors(scene_body_poses(target), scene_body_poses(source))
            assert error["max_position_m"] < 1e-6 and error["max_rotation_rad"] < 1e-6
            start = {}
            for arm, env, c, obs in [("Panda", source, pair.chain, src_obs),
                                     ("Kinova3", target, chain, tgt_obs)]:
                b = boundary(env, mujoco.MjData(env.sim.model._model), obs)
                start[arm] = dict(**b, twist=c.jac(b["q"]) @ b["v"],
                                  hand_quat=obs["robot0_eef_quat"], jaw_obs=obs["robot0_gripper_qpos"])
            a, b = start.values()
            assert np.array_equal(a["fingers"], b["fingers"])
            assert np.linalg.norm(a["xyz"] - b["xyz"]) <= .001
            assert Rotation.from_matrix(a["rot"] @ b["rot"].T).magnitude() <= .01
            receipt = dict(case=tape["case"], prompt=pair.prompt, archive=str(path), sha256=sha(path),
                           archive_noise=tape["noise_sha256"], archive_opening=tape["opening"],
                           scene_sha=canonical_fingerprint(scene_state(source)), cpu_state=pair.full_hash,
                           archive_state=tape["meta"]["state"], cpu_physics=pair.reset_hash, source_proprio=proprio,
                           archive_row=pair.row_hash, archive_physics=tape["meta"]["physics"],
                           target_physics=numerical_state_fingerprint(target.get_sim_state()),
                           bddl_sha=sha(pair.bddl), scene_error=error, alignment=align, start=start)
            for arm, env, c, obs in [("Panda", source, pair.chain, src_obs),
                                     ("Kinova3", target, chain, tgt_obs)]:
                receipt[arm] = rollout(out, arm, env, c, obs, tape, key)
                print(key, arm, receipt[arm]["controls"], flush=True)
            write(out/f"{key}.json", receipt)
        finally:
            if target is not None:
                target.close()
            pair.close()


if __name__ == "__main__":
    print(platform.node(), sorted(os.sched_getaffinity(0)), flush=True)
    collect(Path(sys.argv[1]))
