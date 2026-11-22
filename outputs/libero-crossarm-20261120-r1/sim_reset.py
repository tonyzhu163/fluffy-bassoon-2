"""Transfer one published Panda BASE row into a differently dimensioned MuJoCo model."""

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


# Two percent of a full 0.05 m / 0.5 rad OSC component.
POSITION_TOL_M = 0.001
ROTATION_TOL_RAD = 0.01


def load_panda_row(env, row):
    row = np.asarray(row, dtype=np.float64)
    model = env.sim.model
    if model.na or row.shape != (1 + model.nq + model.nv,) or not np.isfinite(row).all():
        raise ValueError("Released initial state does not fit the compiled donor Panda model")
    env.set_init_state(row)


def scene_state(env):
    """Read the movable/fixture joint coordinates and the separate model-level fixture poses."""
    model, data = env.sim.model, env.sim.data
    robot = env.robots[0]
    excluded = set(robot.robot_joints) | set(robot.gripper_joints)
    names = set(model.joint_names)
    if None in names or len(names) != model.njnt or not excluded <= names:
        raise ValueError("Named joints do not cover the complete compiled model")
    joints = {}
    for name in sorted(names - excluded):
        idx = model.joint_name2id(name)
        joints[name] = dict(
            type=int(model.jnt_type[idx]),
            body=model.body_id2name(int(model.jnt_bodyid[idx])),
            qpos=np.asarray(data.get_joint_qpos(name)).tolist(),
            qvel=np.asarray(data.get_joint_qvel(name)).tolist(),
        )
    fixtures = {}
    for obj in env.env.fixtures_dict.values():
        idx = model.body_name2id(obj.root_body)
        fixtures[obj.root_body] = dict(pos=model.body_pos[idx].tolist(), quat=model.body_quat[idx].tolist())
    return dict(joints=joints, fixtures=fixtures)


def fresh_kinematics(env, scratch):
    """Read current FK without changing robosuite's live observable clock or dynamics caches."""
    model, data = env.sim.model._model, env.sim.data._data
    scratch.qpos[:] = data.qpos
    scratch.mocap_pos[:] = data.mocap_pos
    scratch.mocap_quat[:] = data.mocap_quat
    mujoco.mj_kinematics(model, scratch)


def tool_state(env, scratch):
    fresh_kinematics(env, scratch)
    robot = env.robots[0]
    site = robot.eef_site_id
    hand = env.sim.model.body_name2id(robot.robot_model.eef_name)
    links = env.sim.model.jnt_bodyid[robot._ref_joint_indexes]
    fingers = scratch.qpos[robot._ref_gripper_joint_pos_indexes]
    return dict(
        tcp_xyz=scratch.site_xpos[site].copy(),
        tcp_matrix=scratch.site_xmat[site].reshape(3, 3).copy(),
        hand_xyz=scratch.xpos[hand].copy(),
        hand_quat_xyzw=scratch.xquat[hand, [1, 2, 3, 0]].copy(),
        link_xyz=scratch.xpos[links].copy(),
        link_quat_xyzw=scratch.xquat[links][:, [1, 2, 3, 0]].copy(),
        jaw_opening=float(fingers[0] - fingers[1]),
    )


def tool_delta(before, after):
    relative = after["tcp_matrix"] @ before["tcp_matrix"].T
    return np.r_[after["tcp_xyz"] - before["tcp_xyz"], Rotation.from_matrix(relative).as_rotvec(),
                 after["jaw_opening"] - before["jaw_opening"]].astype(np.float32)


def _match_tool_pose(env, reference):
    """Bounded position-level IK at reset. Only the alternate arm's control joints may move."""
    robot = env.robots[0]
    model = env.sim.model
    scratch = mujoco.MjData(model._model)
    fresh_kinematics(env, scratch)
    indices = robot._ref_joint_pos_indexes
    ids = robot._ref_joint_indexes
    limited = np.asarray(model.jnt_limited[ids], dtype=bool)
    limits = model.jnt_range[ids].copy()
    if type(robot.robot_model).__name__ in ("MountedKinova3", "OnTheGroundKinova3"):
        # Stock Gen3 joints 1, 3, 5 and 7 are continuous hinges; their MJCF
        # range slots are zero placeholders. Leave the three hardware bounds intact.
        continuous = {name for name, has_limit in zip(robot.robot_joints, limited) if not has_limit}
        if continuous != {f"robot0_Actuator{i}" for i in (1, 3, 5, 7)} or not np.all(
                model.jnt_type[ids] == mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError("Unexpected Kinova3 hinge or limit identity")
        limits[~limited] = [-np.inf, np.inf]
    elif not np.all(limited):
        raise ValueError("Expected physical joint limits on the alternate arm")
    site = robot.eef_site_id

    def error(q):
        scratch.qpos[indices] = q
        mujoco.mj_kinematics(model._model, scratch)
        dp = scratch.site_xpos[site] - reference["tcp_xyz"]
        dr = scratch.site_xmat[site].reshape(3, 3) @ reference["tcp_matrix"].T
        return np.r_[dp / POSITION_TOL_M, Rotation.from_matrix(dr).as_rotvec() / ROTATION_TOL_RAD]

    rest = scratch.qpos[indices].copy()
    best = least_squares(error, rest, bounds=(limits[:, 0], limits[:, 1]), max_nfev=600)
    dp = float(np.linalg.norm(best.fun[:3]) * POSITION_TOL_M)
    dr = float(np.linalg.norm(best.fun[3:]) * ROTATION_TOL_RAD)
    if not (dp <= POSITION_TOL_M and dr <= ROTATION_TOL_RAD):
        # Deterministic multi-start fallback around stock rest pose for alternate 7-DoF arms (e.g. IIWA)
        # when single-start TRF hits a local joint-limit boundary while staying within exact tolerances.
        rng = np.random.default_rng(20261122)
        lo, hi = limits[:, 0], limits[:, 1]
        finite = np.isfinite(lo) & np.isfinite(hi)
        span = np.where(finite, np.minimum(hi - lo, 2.0), 1.0)
        for _ in range(8):
            q0 = rest + rng.normal(0.0, 0.25, size=rest.shape) * span
            q0 = np.clip(q0, lo + 1e-4, hi - 1e-4)
            cand = least_squares(error, q0, bounds=(lo, hi), max_nfev=600)
            cdp = float(np.linalg.norm(cand.fun[:3]) * POSITION_TOL_M)
            cdr = float(np.linalg.norm(cand.fun[3:]) * ROTATION_TOL_RAD)
            if cdp <= POSITION_TOL_M and cdr <= ROTATION_TOL_RAD:
                best, dp, dr = cand, cdp, cdr
                break
            if (cdp / POSITION_TOL_M + cdr / ROTATION_TOL_RAD) < (dp / POSITION_TOL_M + dr / ROTATION_TOL_RAD):
                best, dp, dr = cand, cdp, cdr
    result = best
    if not (dp <= POSITION_TOL_M and dr <= ROTATION_TOL_RAD):
        raise ValueError(f"Released grip-site pose unreachable from stock IK seed: {dp:.6g} m, {dr:.6g} rad")
    env.sim.data.qpos[indices] = result.x
    env.sim.data.qvel[robot._ref_joint_vel_indexes] = 0
    return dict(method="joint_limited_grip_site_ik", seed_qpos=rest.tolist(), qpos=result.x.tolist(),
                position_error_m=dp, rotation_error_rad=dr)


def transfer_scene(donor, target, scene):
    """Transfer time, fingers, every task joint and seeded fixture pose, then solve target-arm IK."""
    actual = scene_state(target)
    source_robot, robot = donor.robots[0], target.robots[0]
    model, data = target.sim.model, target.sim.data
    if (scene["joints"].keys() != actual["joints"].keys() or scene["fixtures"].keys() != actual["fixtures"].keys()
            or list(robot.gripper_joints) != list(source_robot.gripper_joints) or model.na):
        raise ValueError("Non-arm joint, common finger or fixture identity differs across BASE models")

    for name, values in scene["joints"].items():
        if any(values[key] != actual["joints"][name][key] for key in ("body", "type")):
            raise ValueError(f"Scene joint {name!r} has a different parent or type")
        data.set_joint_qpos(name, values["qpos"])
        data.set_joint_qvel(name, values["qvel"])
    for name in robot.gripper_joints:
        data.set_joint_qpos(name, donor.sim.data.get_joint_qpos(name))
        data.set_joint_qvel(name, donor.sim.data.get_joint_qvel(name))
    for name, pose in scene["fixtures"].items():
        idx = model.body_name2id(name)
        model.body_pos[idx] = pose["pos"]
        model.body_quat[idx] = pose["quat"]
    data.time = donor.sim.data.time
    reference = tool_state(donor, mujoco.MjData(donor.sim.model._model))
    alignment = _match_tool_pose(target, reference)
    # set_init_state restores the captured model arrays before it loads qpos and qvel.
    target.env.perturbation_runtime.capture_model_baseline(target.env)
    target.set_init_state(target.get_sim_state())
    # The controller was built at the stock joint pose. A zero angular OSC action holds its last goal.
    robot.controller.update_initial_joints(data.qpos[robot._ref_joint_pos_indexes])
    for name in robot.gripper_joints:
        if (data.get_joint_qpos(name) != donor.sim.data.get_joint_qpos(name)
                or data.get_joint_qvel(name) != donor.sim.data.get_joint_qvel(name)):
            raise ValueError(f"Common finger state differs at {name}")
    if data.time != donor.sim.data.time:
        raise ValueError("Target simulation time differs from the released row")
    if scene_state(target) != scene:
        raise ValueError("Target readback differs from released scene before settling")
    return alignment


def scene_body_poses(env):
    """World poses of all named bodies of every task fixture and movable object, including articulated links."""
    model = env.sim.model
    scratch = mujoco.MjData(model._model)
    fresh_kinematics(env, scratch)
    roots = set(env.env.obj_body_id.values())
    bodies = set(roots)
    for idx in range(1, model.nbody):
        if int(model.body_parentid[idx]) in bodies:
            bodies.add(idx)
    # A fixed arena root may contain an unnamed visual child. Jointed descendants must be named.
    poses = {}
    for idx in sorted(bodies):
        name = model.body_id2name(idx)
        if name is None:
            if model.body_jntnum[idx]:
                raise ValueError("An articulated task link is unnamed")
            continue
        poses[name] = dict(xyz=scratch.xpos[idx].tolist(), quat_xyzw=scratch.xquat[idx, [1, 2, 3, 0]].tolist())
    return poses


def pose_errors(actual, reference):
    if not actual or actual.keys() != reference.keys():
        raise ValueError("Task body identities differ across models")
    errors = []
    for name, pose in actual.items():
        a, b = pose, reference[name]
        dp = np.linalg.norm(np.subtract(a["xyz"], b["xyz"]))
        dq = Rotation.from_quat(a["quat_xyzw"]) * Rotation.from_quat(b["quat_xyzw"]).inv()
        errors.append((name, float(dp), float(dq.magnitude())))
    farthest = max(errors, key=lambda item: item[1])
    most_rotated = max(errors, key=lambda item: item[2])
    return dict(max_position_m=farthest[1], position_body=farthest[0],
                max_rotation_rad=most_rotated[2], rotation_body=most_rotated[0])
