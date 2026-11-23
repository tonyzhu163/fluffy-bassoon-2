"""Render-only siblings for the drawer's native Kinova observation."""
from contextlib import contextmanager
from unittest.mock import patch

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from robosuite import macros
from robosuite.utils.binding_utils import MjSim
from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING

from arm_env import ArmEnv, CAMERAS
from geometry import Geometry, transform
from sim_reset import scene_state


def matrix_error(a, b):
    return float(Rotation.from_matrix(a.reshape(3, 3) @ b.reshape(3, 3).T).magnitude())


def activate(sim):
    sim._render_context_offscreen.gl_ctx.make_current()


class Exposure:
    """Snapshot the exposure, including the pre-integration FK used by mjv_updateScene."""
    def __init__(self):
        self.photos = {}
        self.previous = {}

    @contextmanager
    def capture(self):
        original_step, original_render = MjSim.step, MjSim.render

        def step(sim, *args, **kw):
            self.previous[id(sim)] = (sim.data.qpos.copy(), sim.data.qvel.copy())
            return original_step(sim, *args, **kw)

        def render(sim, *args, **kw):
            activate(sim)
            pixels = original_render(sim, *args, **kw)
            cam = kw.get("camera_name")
            if cam in CAMERAS:
                d = sim.data._data
                self.photos[cam] = dict(sim=sim, rgb=pixels.copy(), time=d.time,
                    candidates=[(d.qpos.copy(), d.qvel.copy()), self.previous.get(id(sim))],
                    geom_xpos=d.geom_xpos.copy(), geom_xmat=d.geom_xmat.copy(),
                    site_xpos=d.site_xpos.copy(), site_xmat=d.site_xmat.copy(),
                    cam_xpos=d.cam_xpos.copy(), cam_xmat=d.cam_xmat.copy(),
                    mocap_pos=d.mocap_pos.copy(), mocap_quat=d.mocap_quat.copy())
            return pixels

        with patch.object(MjSim, "step", step), patch.object(MjSim, "render", render):
            yield

    def read(self, native, obs):
        if set(self.photos) != set(CAMERAS):
            raise ValueError("Missing camera exposure")
        sim = native.sim
        convention = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
        if convention != 1:
            raise ValueError("Expected the native OpenGL image convention")
        a, b = [self.photos[c] for c in CAMERAS]
        for cam in CAMERAS:
            p = self.photos[cam]
            if p["sim"] is not sim or not np.array_equal(p["rgb"], obs[cam+"_image"]):
                raise ValueError(f"Native observation does not match captured {cam} exposure")
        for k in ("time", "geom_xpos", "geom_xmat", "cam_xpos", "cam_xmat", "site_xpos", "site_xmat"):
            if not np.array_equal(a[k], b[k]):
                raise ValueError(f"Two camera exposures differ in {k}")
        m = sim.model._model
        scratch = mujoco.MjData(m)
        scratch.mocap_pos[:] = a["mocap_pos"]
        scratch.mocap_quat[:] = a["mocap_quat"]
        # Only accept configurations which reconstruct the transforms actually drawn.
        for index, candidate in enumerate(a["candidates"]):
            if candidate is None:
                continue
            scratch.qpos[:], scratch.qvel[:] = candidate
            mujoco.mj_fwdPosition(m, scratch)
            errors = {k: float(np.max(np.abs(getattr(scratch, k) - a[k])))
                      for k in ("geom_xpos", "geom_xmat", "site_xpos", "site_xmat", "cam_xpos", "cam_xmat")}
            if max(errors.values()) <= 1e-12:
                return scratch, dict(exposure_time=a["time"], fk_candidate=["current","pre_step"][index],
                                     reconstruction=errors)
        raise ValueError(f"Exposure kinematics could not be reconstructed: {errors}")


def joint_span(m, name, velocity=False):
    j = m.joint(name)
    adr = m.jnt_dofadr if velocity else m.jnt_qposadr
    end = (m.nv if velocity else m.nq) if j.id + 1 == m.njnt else adr[j.id + 1]
    return slice(adr[j.id], end)


def robot_arm_ids(native):
    m = native.sim.model._model
    bodies = set(native.robots[0].robot_model.bodies)
    return np.array([i for i in range(m.ngeom) if m.body(m.geom_bodyid[i]).name in bodies])


class Sibling:
    def __init__(self, arm, layout, seed):
        self.arm = arm
        self.env = ArmEnv(arm, "libero_goal", 3)
        self.initial = self.env.reset(layout, seed)
        self.native = self.env._env
        self.sim = self.native.sim
        self.g = Geometry(self.native)
        self.arm_ids = robot_arm_ids(self.native)
        self.previous_q = self.g.qreset.copy()

    def close(self):
        activate(self.sim)
        self.env.close()

    def sync(self, target, exposure):
        tm, g = target.sim.model._model, self.g
        m, data = g.m, self.sim.data._data
        source_scene, destination = scene_state(target), scene_state(self.native)
        if source_scene["fixtures"].keys() != destination["fixtures"].keys():
            raise ValueError("Fixture inventory mismatch")
        for name, values in source_scene["fixtures"].items():
            m.body(name).pos[:] = values["pos"]
            m.body(name).quat[:] = values["quat"]
        if source_scene["joints"].keys() != destination["joints"].keys():
            raise ValueError("Task joint inventory mismatch")
        names = list(source_scene["joints"]) + list(target.robots[0].gripper_joints)
        if self.arm == "Kinova3":
            names += list(target.robots[0].robot_joints)
        for name in names:
            tj, j = tm.joint(name), m.joint(name)
            if tj.type != j.type or tm.body(tj.bodyid[0]).name != m.body(j.bodyid[0]).name:
                raise ValueError(f"Joint schema differs for {name}")
            data.qpos[joint_span(m, name)] = exposure.qpos[joint_span(tm, name)]
            data.qvel[joint_span(m, name, True)] = exposure.qvel[joint_span(tm, name, True)]
        if m.nmocap or tm.nmocap:
            raise ValueError("Scene contains unqualified mocap bodies")
        g.d.qpos[:] = data.qpos
        g.d.qvel[:] = data.qvel
        g.forward()
        site = target.robots[0].eef_site_id
        goal = transform(exposure.site_xpos[site], exposure.site_xmat[site])
        if self.arm == "Panda":
            tries = []
            for seed in (self.previous_q, g.qreset, np.clip(g.qreset + [.2,-.3,.3,.2,-.1,.3,0], g.lo, g.hi)):
                fit = g.solve(goal, seed, None)
                # Only the common tool may touch the scene. Its world pose is checked below.
                bad_contact = any(c["kind"] in ("arm_self_or_base","arm_scene") for c in fit["contacts"])
                tries.append(dict(fit, arm_collision=bad_contact))
                if fit["pose_ok"] and not bad_contact:
                    break
            fit = tries[-1]
            if not fit["pose_ok"] or fit["arm_collision"]:
                return dict(admissible=False, attempts=tries)
            self.previous_q = np.array(fit["q"])
            data.qpos[g.adr] = fit["q"]
        else:
            tries = []
        mujoco.mj_fwdPosition(m, data)
        names_target = set(target.env.obj_body_id)
        if names_target != set(self.native.env.obj_body_id):
            raise ValueError("Task object identities differ")
        roots = set(target.env.obj_body_id.values())
        task_xyz, task_rot = [], []
        for i in range(1, tm.nbody):
            if int(tm.body_parentid[i]) in roots:
                roots.add(i)
            if i in roots and tm.body(i).name and not tm.body(i).name.startswith(("mount0_", "robot0_","gripper0_")):
                j = m.body(tm.body(i).name).id
                task_xyz.append(float(np.linalg.norm(data.xpos[j] - exposure.xpos[i])))
                task_rot.append(matrix_error(data.xmat[j], exposure.xmat[i]))
        if not task_xyz or max(task_xyz + task_rot) > 1e-12:
            raise ValueError(f"Scene pose mismatch {task_xyz} {task_rot}")
        # Lock camera coordinates and clipping planes to the actual target exposure.
        camera_native_delta = {}
        m.stat.extent = tm.stat.extent
        m.stat.center[:] = tm.stat.center
        for cam in CAMERAS:
            a, b = tm.camera(cam).id, m.camera(cam).id
            camera_native_delta[cam] = dict(m=float(np.linalg.norm(data.cam_xpos[b] - exposure.cam_xpos[a])),
                                             rad=matrix_error(data.cam_xmat[b], exposure.cam_xmat[a]))
            if tm.cam_fovy[a] != m.cam_fovy[b]:
                raise ValueError(f"Camera intrinsics differ {cam}")
            data.cam_xpos[b] = exposure.cam_xpos[a]
            data.cam_xmat[b] = exposure.cam_xmat[a]
        t_fingers = exposure.qpos[target.robots[0]._ref_gripper_joint_pos_indexes]
        jaw_error = float(np.max(np.abs(data.qpos[g.fingers] - t_fingers)))
        tool_body_errors = []
        for i in range(tm.nbody):
            name = tm.body(i).name
            if name.startswith("gripper0_"):
                j = m.body(name).id
                tool_body_errors.append([float(np.linalg.norm(data.xpos[j] - exposure.xpos[i])),
                                         matrix_error(data.xmat[j], exposure.xmat[i])])
        return dict(admissible=True, attempts=tries, jaw_error_m=jaw_error,
                    tcp_error_m=float(np.linalg.norm(data.site_xpos[g.site] - goal[:3,3])),
                    tcp_error_rad=matrix_error(data.site_xmat[g.site], goal[:3,:3]),
                    tool_body_errors_max_m_rad=np.max(tool_body_errors, axis=0).tolist(),
                    task_body_count=len(task_xyz), scene_max_m=max(task_xyz), scene_max_rad=max(task_rot),
                    joint_names=names, fixtures=sorted(source_scene["fixtures"]), camera_before_lock=camera_native_delta)

    def render(self, cam, *, clean=False, only_arm=False):
        """Native renderer Z-tests the entire scene; alternate passes measure occlusions."""
        activate(self.sim)
        def draw(segmentation=False):
            return self.sim.render(camera_name=cam, width=256, height=256, depth=True, segmentation=segmentation)
        m = self.g.m
        saved = m.geom_rgba[:,3].copy()
        groups = m.geom_group.copy()
        ctx = self.sim._render_context_offscreen
        visibility = ctx.vopt.geomgroup.copy()
        sites = ctx.vopt.sitegroup.copy()
        try:
            if clean or only_arm:
                hide = self.arm_ids if clean else np.setdiff1d(np.arange(m.ngeom), self.arm_ids)
                m.geom_group[hide] = 5
                ctx.vopt.geomgroup[5] = 0
            if only_arm:
                ctx.vopt.sitegroup[:] = 0
            rgb, depth = draw()
            seg, _ = draw(True)
        finally:
            m.geom_group[:] = groups
            ctx.vopt.geomgroup[:] = visibility
            ctx.vopt.sitegroup[:] = sites
        arm = (seg[:,:,0] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(seg[:,:,1], self.arm_ids)
        return rgb, depth, arm


def summarize_diff(a, b):
    diff = np.abs(a.astype(np.int16)-b.astype(np.int16))
    return dict(changed=int(np.any(diff,axis=-1).sum()), mae=float(diff.mean()), max=int(diff.max()))


def compose(raw, control, source, cam):
    # Visible IDs are the winners of the renderer's depth test in each complete sibling.
    # At vacated Kinova pixels the synchronized Panda scene exposes the surface behind it.
    native, depth, orig_arm = control.render(cam)
    changed, source_depth, src_arm = source.render(cam)
    clean, clean_depth, _ = control.render(cam, clean=True)
    src_clean, src_clean_depth, _ = source.render(cam, clean=True)
    _, _, silhouette = source.render(cam, only_arm=True)
    union = orig_arm | src_arm
    sham = raw.copy()
    sham[orig_arm] = native[orig_arm]
    swap = raw.copy()
    swap[union] = changed[union]
    return sham, swap, dict(raw_vs_full_control=summarize_diff(raw,native),
        raw_vs_sham=summarize_diff(raw,sham), sham_vs_swap=summarize_diff(sham,swap),
        clean_plate=summarize_diff(clean,src_clean),
        clean_depth_max=float(np.max(np.abs(src_clean_depth-clean_depth))),
        orig_arm_px=int(orig_arm.sum()), source_arm_px=int(src_arm.sum()), uncovered_px=int((orig_arm & ~src_arm).sum()),
        source_arm_hidden_px=int((silhouette & ~src_arm).sum()), source_only_px=int((src_arm & ~orig_arm).sum()),
        changed_outside_union=int(np.any(raw[~union]!=swap[~union],axis=-1).sum()),
        visible_depth_ahead_px=int((src_arm & (source_depth < src_clean_depth)).sum()),
        visible_depth_behind_px=int((src_arm & (source_depth > src_clean_depth)).sum()))
