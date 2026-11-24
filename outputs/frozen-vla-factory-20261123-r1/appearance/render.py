"""Generate matched, static LIBERO-Goal observations using released robot meshes."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import mujoco
import numpy as np
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite.models.robots.manipulators.ur5e_robot import UR5e
from robosuite.robots import ROBOT_CLASS_MAPPING, SingleArm
from scipy.ndimage import binary_dilation
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


class MountedUR5e(UR5e):
    """LIBERO table constructor prepends 'Mounted' to the model name."""


ROBOT_CLASS_MAPPING["MountedUR5e"] = SingleArm
CAM = "agentview"


def reset(path, arm, seed):
    random.seed(seed)
    np.random.seed(seed)
    wrapped = OffScreenRenderEnv(
        bddl_file_name=str(path), robots=[arm], gripper_types="PandaGripper",
        camera_heights=256, camera_widths=256, hard_reset=False,
    )
    # Call the native reset directly so an invalid reset raises to the caller.
    wrapped.env.reset()
    return wrapped


def match_scene(source, target):
    s, t = source.sim, target.sim
    for name in s.model.joint_names:
        if not name.startswith("robot0_"):
            t.data.set_joint_qpos(name, s.data.get_joint_qpos(name))
    names = [name for name in s.model.body_names
             if not name.startswith(("robot0_", "gripper0_"))]
    for name in names:
        a, b = s.model.body_name2id(name), t.model.body_name2id(name)
        t.model.body_pos[b] = s.model.body_pos[a]
        t.model.body_quat[b] = s.model.body_quat[a]
    t.forward()
    for name in names:
        np.testing.assert_allclose(s.data.get_body_xpos(name), t.data.get_body_xpos(name), atol=1e-10, rtol=0)
        np.testing.assert_allclose(s.data.get_body_xmat(name), t.data.get_body_xmat(name), atol=1e-10, rtol=0)
    return len(names)


def tool_pose(env):
    i = env.robots[0].eef_site_id
    return env.sim.data.site_xpos[i].copy(), Rotation.from_matrix(env.sim.data.site_xmat[i].reshape(3, 3))


def match_tool(source, target):
    ref, ref_rot = tool_pose(source)
    idx = target.robots[0]._ref_joint_pos_indexes

    def residual(q):
        target.sim.data.qpos[idx] = q
        target.sim.forward()
        pos, rot = tool_pose(target)
        return np.r_[pos - ref, (ref_rot.inv() * rot).as_rotvec()]

    bounds = target.sim.model.jnt_range[target.robots[0]._ref_joint_indexes]
    fit = least_squares(residual, target.sim.data.qpos[idx].copy(), bounds=bounds.T,
                        ftol=1e-13, xtol=1e-13, gtol=1e-13)
    err = residual(fit.x)
    np.testing.assert_allclose(err, 0, atol=1e-8)
    return float(np.linalg.norm(err[:3])), float(np.linalg.norm(err[3:]))


def render(env):
    s = env.sim
    # robosuite 1.4 sets the GL context only on construction. Switching models
    # without rebinding it draws meshes/textures from the previous context.
    s._render_context_offscreen.gl_ctx.make_current()
    kw = dict(width=256, height=256, camera_name=CAM)
    rgb = s.render(**kw)
    seg = s.render(**kw, segmentation=True)
    gids = [i for i, b in enumerate(s.model.geom_bodyid)
            if s.model.body_id2name(b).startswith(("robot0_", "gripper0_"))]
    mask = (seg[..., 0] == mujoco.mjtObj.mjOBJ_GEOM) & np.isin(seg[..., 1], gids)
    assert mask.any()
    return rgb, mask


def robot_scene_contacts(env):
    sim = env.sim
    pairs = []
    for c in sim.data.contact[:sim.data.ncon]:
        bodies = [sim.model.body_id2name(sim.model.geom_bodyid[g]) for g in (c.geom1, c.geom2)]
        moving = [name.startswith(("robot0_", "gripper0_")) for name in bodies]
        if sum(moving) == 1 and not any(name.startswith("mount0_") for name in bodies):
            pairs.append(bodies)
    return pairs


def case(root, path, i):
    seed = 3200 + i
    source, target = reset(path, "Panda", seed), reset(path, "UR5e", seed)
    try:
        # Diversify the public reset postures without stepping either system.
        idx = source.robots[0]._ref_joint_pos_indexes
        source.sim.data.qpos[idx] += np.random.default_rng(seed).uniform(-0.12, 0.12, len(idx))
        source.sim.forward()
        n_bodies = match_scene(source, target)
        pos_err, rot_err = match_tool(source, target)
        source_contacts, target_contacts = robot_scene_contacts(source), robot_scene_contacts(target)
        for cam in (CAM, "robot0_eye_in_hand"):
            for field in ("cam_xpos", "cam_xmat"):
                a = getattr(source.sim.data, field)[source.sim.model.camera_name2id(cam)]
                b = getattr(target.sim.data, field)[target.sim.model.camera_name2id(cam)]
                np.testing.assert_allclose(a, b, atol=1e-8, rtol=0)
            assert source.sim.model.cam_fovy[source.sim.model.camera_name2id(cam)] == target.sim.model.cam_fovy[target.sim.model.camera_name2id(cam)]

        rgb, mask = render(source)
        other, other_mask = render(target)
        source.sim._render_context_offscreen.gl_ctx.make_current()
        source._update_observables(force=True)
        obs = source.env._get_observations()
        np.testing.assert_array_equal(rgb, obs["agentview_image"])
        # Keep anti-aliasing within a one-pixel band, freeze all other pixels.
        union = binary_dilation(mask | other_mask, structure=np.ones((3, 3)))
        controlled = np.where(union[..., None], other, rgb)
        diff = np.any(rgb != other, axis=-1)
        assert not np.any((rgb != controlled)[~union])
        out = root / "cases"
        out.mkdir(exist_ok=True)
        with (out / f"{i:02d}.npz").open("wb") as f:
            np.savez_compressed(f, task=path.stem.replace("_", " "), source=rgb,
                                native=other, target=controlled, source_mask=mask, target_mask=other_mask,
                                wrist=obs["robot0_eye_in_hand_image"], pos=obs["robot0_eef_pos"],
                                quat=obs["robot0_eef_quat"], grip=obs["robot0_gripper_qpos"])
        return dict(task=path.stem, seed=seed, scene_bodies=n_bodies, tool_error_m=pos_err,
                    tool_error_rad=rot_err, source_fraction=mask.mean(), target_fraction=other_mask.mean(),
                    union_band_fraction=union.mean(), isolated_changed_fraction=(diff & union).mean(),
                    discarded_changed_fraction=(diff & ~union).mean(),
                    camera_and_scene_match=True,
                    source_scene_contacts=source_contacts, target_scene_contacts=target_contacts,
                    source_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                    target_sha256=hashlib.sha256(controlled.tobytes()).hexdigest())
    finally:
        target.close()
        source.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    files = sorted((Path(get_libero_path("bddl_files")) / "libero_goal").glob("*.bddl"))
    assert len(files) == 10
    rows = []
    for i, path in enumerate(files):
        rows.append(case(args.root, path, i))
        print(i, rows[-1], flush=True)
    (args.root / "render.json").write_text(json.dumps(rows, indent=2) + "\n")
