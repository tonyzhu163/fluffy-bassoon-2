"""Opt-in expert-planner repair; no VLA sampling or inference action changes.

The native wrapper drops non-arm DOFs before calling an articulation planner.
Provide its full measured qpos so padding cannot import stale other-arm state.
GPU positive controls are required before using this to qualify a scene bank.
"""
import numpy as np


def full_qpos(wrapper, current):
    value = np.asarray(current).copy()
    assert value.ndim == 1 and len(value) == len(wrapper.all_joints)
    assert len(set(wrapper.all_joints)) == len(wrapper.all_joints) and np.isfinite(value).all()
    indices = [wrapper.all_joints.index(name) for name in wrapper.active_joints_name]
    assert len(indices) == len(set(indices))
    # Retain upstream arm rounding, while preserving every other measured DOF.
    value[indices] = np.round(value[indices], 5)
    return value


def plan_path(self, curr_joint_pos, target_gripper_pose, constraint_pose=None, arms_tag=None):
    import mplib
    from envs.robot.planner import MplibPlanner
    target = mplib.pymp.Pose(list(target_gripper_pose.p), list(target_gripper_pose.q))
    result = MplibPlanner.plan_path(self, now_qpos=full_qpos(self, curr_joint_pos),
        target_pose=target, use_point_cloud=False, use_attach=False, arms_tag=arms_tag, log=False)
    if result.get('status') == 'Success':
        for key in ('position', 'velocity'): result[key] = np.asarray(result[key])
    return result


def plan_batch(self, curr_joint_pos, target_gripper_pose_list, constraint_pose=None, arms_tag=None):
    results = [plan_path(self, curr_joint_pos, target, constraint_pose=constraint_pose, arms_tag=arms_tag)
        for target in target_gripper_pose_list]
    return dict(status=np.asarray(['Success' if r['status'] == 'Success' else 'Failure' for r in results], dtype=object),
        position=np.asarray([r['position'] if r['status'] == 'Success' else np.array([]) for r in results], dtype=object),
        velocity=np.asarray([r['velocity'] if r['status'] == 'Success' else np.array([]) for r in results], dtype=object))


def install():
    from envs.robot.planner import MplibWrapperPlanner
    if getattr(MplibWrapperPlanner, '_fb2_full_measured_qpos', False): return
    MplibWrapperPlanner.plan_path = plan_path
    MplibWrapperPlanner.plan_batch = plan_batch
    MplibWrapperPlanner._fb2_full_measured_qpos = True


def plan_path_sync(self, curr_joint_pos, target_gripper_pose, constraint_pose=None, arms_tag=None):
    # MPlib's documented SAPIEN API updates articulations and objects from the scene.
    # Preserve every collision check and use the existing full-qpos route afterward.
    self.planner.update_from_simulation()
    return plan_path(self, curr_joint_pos, target_gripper_pose, constraint_pose, arms_tag)


def plan_batch_sync(self, curr_joint_pos, target_gripper_pose_list, constraint_pose=None, arms_tag=None):
    results = [plan_path_sync(self, curr_joint_pos, target, constraint_pose, arms_tag)
        for target in target_gripper_pose_list]
    return dict(status=np.asarray(['Success' if r['status'] == 'Success' else 'Failure' for r in results], dtype=object),
        position=np.asarray([r['position'] if r['status'] == 'Success' else np.array([]) for r in results], dtype=object),
        velocity=np.asarray([r['velocity'] if r['status'] == 'Success' else np.array([]) for r in results], dtype=object))


def install_sync():
    from envs.robot.planner import MplibWrapperPlanner
    MplibWrapperPlanner.plan_path = plan_path_sync
    MplibWrapperPlanner.plan_batch = plan_batch_sync
    MplibWrapperPlanner._fb2_full_measured_qpos = True
