"""Physical stock arms in released LIBERO-Pro2 tasks (BASE by default).

Import in the LIBERO interpreter. The motor and the learned reader run in their own process.
"""

from functools import lru_cache
import hashlib
from importlib.metadata import version
import os
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.env_wrapper import ControlEnv
from libero.libero.envs.robots import MountedPanda, OnTheGroundPanda
from libero_remote.environment import task_manifest
from libero_remote.environment.libero import _set_controller_mode, _settle_environment
from libero_remote.environment.observations import normalize_standard_observation
from libero_remote.reproducibility import canonical_fingerprint, numerical_state_fingerprint
from robosuite.models.robots import manipulators
from robosuite.models.robots.robot_model import REGISTERED_ROBOTS
from robosuite.robots import ROBOT_CLASS_MAPPING, SingleArm

from sim_reset import (
    POSITION_TOL_M, ROTATION_TOL_RAD, load_panda_row, pose_errors, scene_body_poses, scene_state,
    tool_delta, tool_state, transfer_scene,
)


CAMERAS = ("agentview", "robot0_eye_in_hand")
SOURCE_PIN = "e5500c690c1498eca1b7050350654964190cdf70"
MANIFEST_SHA = "71a6e4e25a2a4a162844d77136e7bff8e6f69c839d2d8194620c4afeabde2bba"
ARM_XML_SHA = {
    "Panda": "51e3a4a6d54bcaf77e9d36a27372b871728590eedf53d4fd0e195525950eb988",
    "Sawyer": "a0a9743d868c23e157e57b3415c582c2223379aefaebf54c5dbff3653998fbaf",
    "UR5e": "1c9818566a7bbba89a2c848523e6e1eb72db204d91b8a05d0bff5bbb2a4ad316",
    "IIWA": "7eb9d87f431f7fbfcb0e028e608956aa2c2f7dc456abb0dfb6f234fd8044a5dd",
    "Kinova3": "2884d72a0427223d333dc00fa2c199c3f86406184c234f436436c7d149002cd7",
}
GRIPPER_XML_SHA = "d9d68123909d6910488310d0fae4e358ed4ac831c6d0957f3e30893a5412baac"
OSC_SCALE = np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5])


@lru_cache(maxsize=1)
def _manifest(path):
    result = task_manifest.TaskManifest.load(str(path))
    if result.sha256 != MANIFEST_SHA:
        raise ValueError("Expected the published LIBERO-Pro2 e5500c69 manifest")
    return result


@lru_cache(maxsize=None)
def _register_arm(arm):
    if arm == "Panda":
        return  # LIBERO registered its two physical Panda variants during import.
    if arm in ("IIWA", "Kinova3"):
        # LIBERO integrations can expose only a subset of the wheel's stock factory.
        expected = getattr(manipulators, arm)
        if (REGISTERED_ROBOTS.setdefault(arm, expected) is not expected
                or ROBOT_CLASS_MAPPING.setdefault(arm, SingleArm) is not SingleArm):
            raise RuntimeError(f"Conflicting robosuite factory for stock {arm}")
    stock = REGISTERED_ROBOTS.get(arm)
    if stock is None or ROBOT_CLASS_MAPPING.get(arm) is not SingleArm:
        raise RuntimeError(f"Missing stock single arm {arm} in robosuite 1.4.0")
    for placement in (MountedPanda, OnTheGroundPanda):
        name = placement.__name__.removesuffix("Panda") + arm
        if name in REGISTERED_ROBOTS or name in ROBOT_CLASS_MAPPING:
            raise RuntimeError(f"Refusing to replace an existing simulator model {name}")
        # RobotModelMeta registers this stock subclass in the XML model factory.
        # LIBERO supplies only the arena/mount geometry. The arm inherits its own MJCF and rest joints.
        type(name, (stock,), {"default_mount": placement.default_mount, "base_xpos_offset": placement.base_xpos_offset})
        ROBOT_CLASS_MAPPING[name] = SingleArm


def _identity(env, arm, case_id="BASE"):
    robot = env.robots[0]
    model = env.sim.model
    arm_model = robot.robot_model
    name = type(arm_model).__name__
    xml_sha = hashlib.sha256(Path(arm_model.file).read_bytes()).hexdigest()
    finger_sha = hashlib.sha256(Path(robot.gripper.file).read_bytes()).hexdigest()
    joints = list(robot.robot_joints)
    if (xml_sha != ARM_XML_SHA[arm] or name not in ("Mounted" + arm, "OnTheGround" + arm)
            or len(joints) != (6 if arm == "UR5e" else 7) or arm_model.naming_prefix != "robot0_"):
        raise RuntimeError(f"Compiled arm does not match requested stock model {arm}: {name}, {joints}, {xml_sha}")
    if (type(robot.gripper).__name__ != "PandaGripper" or finger_sha != GRIPPER_XML_SHA
            or robot.gripper.dof != 1 or list(robot.gripper_joints) != ["gripper0_finger_joint1", "gripper0_finger_joint2"]):
        raise RuntimeError("Expected one command driving the two signed physical Panda finger slides")

    hand_id = model.body_name2id(arm_model.eef_name)
    wrist_id = model.camera_name2id(CAMERAS[1])
    view_id = model.camera_name2id(CAMERAS[0])
    wrist_fovy = 75.0
    if case_id == "S16":
        plan = env.get_perturbation_plan()
        if CAMERAS[1] in plan["targets"]:
            wrist_fovy = np.clip(wrist_fovy * float(plan["params"]["fov_scale"]), 20, 100)
    if (model.cam_bodyid[wrist_id] != hand_id or not np.isclose(model.cam_fovy[wrist_id], wrist_fovy)
            or model.cam_bodyid[view_id] == hand_id):
        raise RuntimeError("Camera parent or field of view differs from the physical LIBERO camera contract")
    c = robot.controller
    site = robot.gripper.important_sites["grip_site"]
    low, high = env.env.action_spec
    if (c.name != "OSC_POSE" or not c.use_delta or not c.use_ori or c.impedance_mode != "fixed"
            or c.eef_name != site or robot.eef_site_id != model.site_name2id(site)
            or env.env.control_freq != 20 or not np.array_equal(low, -np.ones(7))
            or not np.array_equal(high, np.ones(7)) or not np.allclose(c.output_max, OSC_SCALE)
            or not np.allclose(c.output_min, -OSC_SCALE)):
        raise RuntimeError("Expected world relative 6D OSC at the observed grip site plus one gripper command")
    links = [model.body_id2name(int(model.jnt_bodyid[model.joint_name2id(j)])) for j in joints]
    # Only mesh geoms physically attached to the arm itself qualify a robot mesh identity.
    arm_bodies = set(arm_model.bodies)
    mesh_ids = {int(model.geom_dataid[i]) for i in range(model.ngeom)
                if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
                and model.body_id2name(int(model.geom_bodyid[i])) in arm_bodies}
    meshes = sorted(model.mesh_id2name(i) for i in mesh_ids)
    if not meshes:
        raise RuntimeError(f"No compiled mesh geometry on {name}")
    return dict(arm=arm, model_class=name, joint_count=len(joints), joint_names=joints, joint_links=links,
                model_nq=int(model.nq), model_nv=int(model.nv), xml_sha256=xml_sha, arm_meshes=meshes,
                mount=type(arm_model.mount).__name__, base_xyz=env.sim.data.get_body_xpos(arm_model.root_body).tolist(),
                gripper="PandaGripper", gripper_xml_sha256=finger_sha, gripper_joints=list(robot.gripper_joints),
                tcp_site=site, hand_link=arm_model.eef_name, cameras=list(CAMERAS), wrist_parent=arm_model.eef_name,
                wrist_fovy_deg=float(model.cam_fovy[wrist_id]), controller=c.name, control_freq=20,
                osc_output_max=c.output_max.tolist(), robosuite=version("robosuite"), mujoco=version("mujoco"))


def _public(raw, joint_count):
    obs = normalize_standard_observation(raw, CAMERAS)
    for key in ("agentview_image", "robot0_eye_in_hand_image"):
        if raw[key].dtype != np.uint8 or obs[key].shape != (256, 256, 3):
            raise ValueError(f"Unexpected native RGB output for {key}")
    shapes = dict(robot0_joint_pos=joint_count, robot0_joint_vel=joint_count, robot0_eef_pos=3,
                  robot0_eef_quat_xyzw=4, robot0_gripper_qpos=2)
    for key, size in shapes.items():
        if obs[key].shape != (size,) or not np.isfinite(obs[key]).all():
            raise ValueError(f"Invalid native public observation {key}")
    if not np.isclose(np.linalg.norm(obs["robot0_eef_quat_xyzw"]), 1.0, atol=1e-5):
        raise ValueError("Invalid hand-link quaternion")
    return obs


def _reset_wrapper(env, seed):
    env.seed(seed)
    env.set_perturbation_seed(seed)
    env.reset()


def _measured_pose(observation):
    fingers = observation["robot0_gripper_qpos"]
    return dict(tcp_xyz=observation["robot0_eef_pos"].copy(),
                tcp_matrix=Rotation.from_quat(observation["robot0_eef_quat_xyzw"]).as_matrix(),
                jaw_opening=float(fingers[0] - fingers[1]))


class ArmEnv:
    """One real arm and task. A fresh compiled simulation is owned by each successful reset."""

    def __init__(self, arm, suite, task_id, *, case_id="BASE"):
        if arm not in ARM_XML_SHA:
            raise ValueError(f"Unknown study arm {arm!r}, expected one of {tuple(ARM_XML_SHA)}")
        if version("robosuite") != "1.4.0" or version("mujoco") != "3.2.3":
            raise RuntimeError("Use the pinned robosuite 1.4.0 / MuJoCo 3.2.3 simulator interpreter")
        _register_arm(arm)
        release = Path(os.environ.get("CROSSARM_RELEASE", Path(task_manifest.__file__).resolve().parents[2]))
        manifest = _manifest(release / "assets/task_manifest.json")
        if case_id not in ("BASE", "S13", "S14", "S15", "S16", "S19"):
            raise ValueError(f"Unsupported cross-arm case {case_id!r}")
        self.case_id = case_id
        self.entry = manifest.require(suite, task_id, case_id)
        self.bddl = manifest.resolve_bddl(suite, task_id, case_id)
        self.base_bddl = manifest.resolve_bddl(suite, task_id, "BASE")
        self.arm = arm
        self.task_id = int(task_id)
        self.suite = suite
        self._states = benchmark.get_benchmark_dict()[suite]().get_task_init_states(self.task_id)
        self._env = None
        self._scratch = None
        self._tool = None
        self._previous_public = None
        self.model_identity = None
        self._step = 0
        self._done = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._done = True
        self._scratch = self._tool = self.model_identity = None
        self._previous_public = None
        if self._env is not None:
            env, self._env = self._env, None
            env.close()

    def reset(self, init_state_id, seed):
        if not 0 <= init_state_id < len(self._states):
            raise ValueError("init_state_id outside released task state bank")
        self.close()
        seed = int(seed)
        np.random.seed(seed)
        common = dict(bddl_file_name=str(self.bddl), controller="OSC_POSE", gripper_types="PandaGripper",
                      initialization_noise=None, control_freq=20, camera_names=list(CAMERAS),
                      camera_heights=256, camera_widths=256)
        donor = None
        try:
            env = self._env = OffScreenRenderEnv(robots=[self.arm], **common)
            _reset_wrapper(env, seed)
            row = self._states[init_state_id]
            if self.arm == "Panda":
                source = env
            else:
                donor_config = dict(common, bddl_file_name=str(self.base_bddl))
                donor = ControlEnv(robots=["Panda"], use_camera_obs=False, has_offscreen_renderer=False, **donor_config)
                _reset_wrapper(donor, seed)
                source = donor
            load_panda_row(source, row)
            scene = scene_state(source)
            alignment = dict(method="released_panda_state")
            if donor is not None:
                if self.case_id != "BASE":
                    # transfer_scene captures the transferred fixture baseline. Restore clean
                    # visual model arrays first so native set_init_state applies the case once.
                    env.env.perturbation_runtime.prepare_reset(env.env)
                alignment = transfer_scene(donor, env, scene)
            initial_scene_hash = canonical_fingerprint(scene_state(env))
            if initial_scene_hash != canonical_fingerprint(scene):
                raise RuntimeError("Pre-settle target scene does not match the donor")
            identity = _identity(env, self.arm, self.case_id)
            _settle_environment(env)
            env.set_perturbation_seed(seed)
            raw = env.begin_perturbation_rollout()
            _set_controller_mode(env, "relative")
            poses = scene_body_poses(env)
            self._scratch = mujoco.MjData(env.sim.model._model)
            self._tool = tool_state(env, self._scratch)
            if donor is not None:
                _settle_environment(donor)
                donor.set_perturbation_seed(seed)
                donor.begin_perturbation_rollout()
                errors = pose_errors(poses, scene_body_poses(donor))
                donor_tool = tool_state(donor, mujoco.MjData(donor.sim.model._model))
                settled_delta = tool_delta(donor_tool, self._tool)
                alignment.update(settled_tcp_error_m=float(np.linalg.norm(settled_delta[:3])),
                                 settled_rotation_error_rad=float(np.linalg.norm(settled_delta[3:6])),
                                 settled_jaw_difference_m=float(settled_delta[6]))
            else:
                errors = pose_errors(poses, poses)
            if not (errors["max_position_m"] <= POSITION_TOL_M and errors["max_rotation_rad"] <= ROTATION_TOL_RAD):
                raise ValueError(f"Settling moved the paired task scene beyond the admission tolerance: {errors}")
            if env.check_success():
                raise ValueError("Task already successful at the first policy observation")
            plan = env.get_perturbation_plan()
            if plan["case_id"] != self.case_id or plan["seed"] != seed:
                raise ValueError(f"Unexpected released perturbation plan {plan}")
            self._step = 0
            self.model_identity = identity
            self._done = False
            obs = _public(raw, identity["joint_count"])
            self._previous_public = _measured_pose(obs)
            return dict(
                observation=obs, language_instruction=str(env.language_instruction),
                model_identity=identity, tool_observation=self._copy_tool(), max_steps=self.entry["max_steps"],
                worker_step_index=0, environment_reproducibility=dict(
                    source_pin=SOURCE_PIN, manifest_sha256=MANIFEST_SHA, bddl_sha256=self.entry["bddl_sha256"],
                    suite=self.suite, task_id=self.task_id, init_state_id=int(init_state_id), seed=seed,
                    released_row_fingerprint=numerical_state_fingerprint(row), scene_fingerprint=initial_scene_hash,
                    post_settle_scene_fingerprint=canonical_fingerprint(poses), post_settle_scene_errors=errors,
                    initial_state_fingerprint=numerical_state_fingerprint(env.get_sim_state()), alignment=alignment,
                    perturbation_plan=plan, perturbation_plan_fingerprint=canonical_fingerprint(plan),
                ),
            )
        except BaseException:
            self.close()
            raise
        finally:
            if donor is not None:
                donor.close()

    def _copy_tool(self):
        return {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in self._tool.items()}

    def step(self, action7):
        if self._done:
            raise RuntimeError("Call reset before stepping a new or completed episode")
        action = np.asarray(action7)
        if (action.shape != (7,) or action.dtype.kind not in "fiu" or not np.isfinite(action).all()
                or np.any(action < -1) or np.any(action > 1)):
            raise ValueError("Expected one finite bounded 7D OSC/gripper control")
        issued = action.astype(np.float32, copy=True)
        # A simulator error may occur after an integration substep. Require a fresh reset in that case.
        self._done = True
        raw, reward, done, _ = self._env.step(issued.copy())
        self._step += 1
        success = bool(self._env.check_success())
        terminated = bool(done or success)
        truncated = self._step >= self.entry["max_steps"] and not terminated
        current = tool_state(self._env, self._scratch)
        obs = _public(raw, self.model_identity["joint_count"])
        measured = _measured_pose(obs)
        motion = tool_delta(self._previous_public, measured)
        if not np.isfinite(motion).all() or not np.isfinite(reward):
            raise RuntimeError("Nonfinite sensor response or task reward")
        self._tool = current
        self._previous_public = measured
        self._done = terminated or truncated
        return dict(observation=obs, language_instruction=str(self._env.language_instruction),
                    issued_action=issued, sensor_motion=motion, tool_observation=self._copy_tool(), reward=float(reward),
                    success=success, terminated=terminated, truncated=truncated, worker_step_index=self._step)

    def probe(self, action7):
        """Issue a real, counted control on the current episode and measure its physical response."""
        return self.step(action7)
