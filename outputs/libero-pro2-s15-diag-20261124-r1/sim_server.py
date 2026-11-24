"""Simulator worker for Track 2: LIBERO-Pro2.0 S15 causal ablation & S16 rectification."""
import json
from multiprocessing.connection import Connection
import os
from pathlib import Path
import sys
import traceback

import cv2
import numpy as np

from arm_env import CAMERAS, _public, _reset_wrapper
from libero.libero import benchmark
from libero.libero.envs.env_wrapper import ControlEnv
from libero_remote.environment.libero import _settle_environment
from sim_reset import load_panda_row
from ur5e_sim_worker import _probe_egl


def _rectify_s16_image(img: np.ndarray, fov_scale: float, shift_xy, crop_ratio: float, distortion_strength: float, base_fovy: float = 45.0):
    """Exact inverse of S16 image geometry (np.roll -> radial distortion/crop -> FOV zoom)."""
    h, w = img.shape[:2]
    # 1. Exact inverse of np.roll(transformed, shift=(sy, sx), axis=(0, 1))
    if np.allclose(shift_xy, np.array([0.05, -0.05])):
        legacy = max(1, h // 20)
        sy, sx = legacy, -legacy
    else:
        sy = int(round(h * float(shift_xy[0])))
        sx = int(round(w * float(shift_xy[1])))
    unrolled = np.roll(img, shift=(-sy, -sx), axis=(0, 1))
    # 2. Undo crop + radial distortion + FOV zoom
    yy, xx = np.indices((h, w), dtype=np.float32)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    nx = (xx - cx) / max(cx, 1.0)
    ny = (yy - cy) / max(cy, 1.0)
    zoom = float(np.tan(np.deg2rad(base_fovy) / 2.0) / np.tan(np.deg2rad(np.clip(base_fovy * fov_scale, 20.0, 100.0)) / 2.0))
    nx_z = nx * zoom
    ny_z = ny * zoom
    scale = 1.0 / max(1.0 - float(crop_ratio), 1e-6)
    ox = nx_z / scale
    oy = ny_z / scale
    k = float(distortion_strength)
    if k != 0.0:
        for _ in range(6):
            r2 = ox * ox + oy * oy
            denom = (1.0 + k * r2) * scale
            ox = nx_z / denom
            oy = ny_z / denom
    map_x = (cx + ox * cx).astype(np.float32)
    map_y = (cy + oy * cy).astype(np.float32)
    return cv2.remap(unrolled, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _rectify_affine(img: np.ndarray, zoom: float, dx: float, dy: float):
    """Inverse roll + affine zoom estimated from self-supervised motion calibration."""
    h, w = img.shape[:2]
    unrolled = np.roll(img, shift=(-int(round(dy)), -int(round(dx))), axis=(0, 1))
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    yy, xx = np.indices((h, w), dtype=np.float32)
    map_x = (cx + (xx - cx) * zoom).astype(np.float32)
    map_y = (cy + (yy - cy) * zoom).astype(np.float32)
    return cv2.remap(unrolled, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _project_point(model, data, cam_name, pt_world, width=256, height=256):
    cid = model.camera_name2id(cam_name)
    pos = data.cam_xpos[cid]
    mat = data.cam_xmat[cid].reshape(3, 3)
    fovy = float(model.cam_fovy[cid])
    f = 0.5 * height / np.tan(np.deg2rad(fovy) / 2.0)
    p_cam = mat.T @ (pt_world - pos)
    if p_cam[2] >= -1e-6:
        return None
    u = width / 2.0 + f * (p_cam[0] / (-p_cam[2]))
    v = height / 2.0 - f * (p_cam[1] / (-p_cam[2]))
    return np.array([u, v], dtype=np.float64)


def _estimate_selfcal_warp(env, base_cam_pos, base_cam_mat, base_fovy):
    """Zero-privilege self-supervised S16 calibration: 12 micro-steps (0.6s) of arm motion."""
    probes = (
        [[0.8, 0, 0, 0, 0, 0, -1]] * 3
        + [[-0.8, 0, 0, 0, 0, 0, -1]] * 3
        + [[0, 0.8, 0, 0, 0, 0, -1]] * 3
        + [[0, -0.8, 0, 0, 0, 0, -1]] * 3
    )
    h, w = 256, 256
    f0 = 0.5 * h / np.tan(np.deg2rad(base_fovy) / 2.0)
    prev_img = None
    prev_p = None
    obs_pts, can_pts = [], []
    for a in probes:
        raw, _, _, _ = env.step(np.array(a, dtype=np.float64))
        img = raw["agentview_image"].astype(np.float32)
        eef = env.sim.data.site_xpos[env.robots[0].eef_site_id].copy()
        p_cam = base_cam_mat.T @ (eef - base_cam_pos)
        # Robosuite raw images use OpenGL bottom-up row indexing (y increases upward)
        can_uv = np.array([w / 2.0 + f0 * (p_cam[0] / (-p_cam[2])), h / 2.0 + f0 * (p_cam[1] / (-p_cam[2]))])
        if prev_img is not None:
            diff = np.mean(np.abs(img - prev_img), axis=2)
            thresh = max(10.0, float(np.percentile(diff, 99.0)))
            mask = diff >= thresh
            if np.sum(mask) >= 15:
                ys, xs = np.nonzero(mask)
                wts = diff[mask]
                obs_uv = np.array([np.average(xs, weights=wts), np.average(ys, weights=wts)])
                obs_pts.append(obs_uv)
                can_pts.append(0.5 * (can_uv + prev_p))
        prev_img = img
        prev_p = can_uv
    if len(obs_pts) < 3:
        return 1.0, 0.0, 0.0
    obs_pts = np.array(obs_pts)
    can_pts = np.array(can_pts)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    c_rel = can_pts - np.array([cx, cy])
    o_rel = obs_pts - np.array([cx, cy])
    # Fit isotropic zoom via least-squares on centered displacements
    dc = c_rel - c_rel.mean(axis=0)
    do = o_rel - o_rel.mean(axis=0)
    denom = float(np.sum(dc * dc))
    zoom = float(np.clip(np.sum(dc * do) / max(denom, 1e-3), 0.55, 1.25))
    shift = (o_rel - zoom * c_rel).mean(axis=0)
    return zoom, float(shift[0]), float(shift[1])


def run(fd, fixture):
    rel = Path(os.environ["CROSSARM_RELEASE"])
    manifest = json.loads((rel / "assets/task_manifest.json").read_text())["tasks"]
    by_key = {(t["suite"], t["base_task_id"], t["case_id"]): t for t in manifest}
    envs = {}
    with Connection(fd) as conn:
        try:
            vendor, renderer = _probe_egl(fixture)
            while True:
                op, v = conn.recv()
                if op == "case":
                    for e in envs.values():
                        e.close()
                    envs.clear()
                    suite, tid, layout, seed = v["suite"], v["task_id"], v["layout"], v["seed"]
                    row = benchmark.get_benchmark_dict()[suite]().get_task_init_states(tid)[layout]
                    base_cam_poses = {}
                    for cid in ("BASE", "S15", "S16"):
                        entry = by_key[(suite, tid, cid)]
                        bddl = rel / "assets" / entry["bddl_path"]
                        np.random.seed(seed)
                        e = ControlEnv(
                            bddl_file_name=str(bddl),
                            robots=["Panda"],
                            controller="OSC_POSE",
                            gripper_types="PandaGripper",
                            control_freq=20,
                            use_camera_obs=True,
                            has_offscreen_renderer=True,
                            camera_heights=256,
                            camera_widths=256,
                            camera_names=list(CAMERAS),
                        )
                        envs[cid] = e
                    # Capture nominal BASE camera parameters
                    _reset_wrapper(envs["BASE"], seed)
                    load_panda_row(envs["BASE"], row)
                    _settle_environment(envs["BASE"])
                    m0, d0 = envs["BASE"].sim.model, envs["BASE"].sim.data
                    for cname in CAMERAS:
                        cam_id = m0.camera_name2id(cname)
                        base_cam_poses[cname] = dict(
                            pos=m0.cam_pos[cam_id].copy(),
                            quat=m0.cam_quat[cam_id].copy(),
                            xpos=d0.cam_xpos[cam_id].copy(),
                            xmat=d0.cam_xmat[cam_id].reshape(3, 3).copy(),
                            fovy=float(m0.cam_fovy[cam_id]),
                        )
                    horizon = by_key[(suite, tid, "BASE")]["max_steps"]
                    prompt = envs["BASE"].language_instruction
                    cur_seed = seed
                    cur_row = row
                    payload = dict(horizon=horizon, prompt=prompt, renderer=renderer, vendor=vendor)
                elif op == "start":
                    cond = v
                    if cond == "BASE":
                        cid = "BASE"
                    elif cond.startswith("S15"):
                        cid = "S15"
                    else:
                        cid = "S16"
                    env = envs[cid]
                    _reset_wrapper(env, cur_seed)
                    load_panda_row(env, cur_row)
                    _settle_environment(env)
                    env.set_perturbation_seed(cur_seed)
                    raw = env.begin_perturbation_rollout()
                    # Apply factorial camera restoration for S15_ext_only / S15_wrist_only
                    if cond == "S15_ext_only":
                        w_id = env.sim.model.camera_name2id("robot0_eye_in_hand")
                        env.sim.model.cam_pos[w_id] = base_cam_poses["robot0_eye_in_hand"]["pos"].copy()
                        env.sim.model.cam_quat[w_id] = base_cam_poses["robot0_eye_in_hand"]["quat"].copy()
                        env.sim.forward()
                        raw = env.env._get_observations(force_update=True)
                    elif cond == "S15_wrist_only":
                        a_id = env.sim.model.camera_name2id("agentview")
                        env.sim.model.cam_pos[a_id] = base_cam_poses["agentview"]["pos"].copy()
                        env.sim.model.cam_quat[a_id] = base_cam_poses["agentview"]["quat"].copy()
                        env.sim.forward()
                        raw = env.env._get_observations(force_update=True)

                    selfcal_params = None
                    if cond == "S16_selfcal_rect":
                        # Estimate warp from a fresh 6-step micro-probe, then reset episode cleanly
                        z, dx, dy = _estimate_selfcal_warp(
                            env,
                            base_cam_poses["agentview"]["xpos"],
                            base_cam_poses["agentview"]["xmat"],
                            base_cam_poses["agentview"]["fovy"],
                        )
                        selfcal_params = (z, dx, dy)
                        _reset_wrapper(env, cur_seed)
                        load_panda_row(env, cur_row)
                        _settle_environment(env)
                        env.set_perturbation_seed(cur_seed)
                        raw = env.begin_perturbation_rollout()

                    s16_plan = env.get_perturbation_plan().get("params", {}) if cid == "S16" else {}
                    cur_cond = cond

                    def _process_obs(r_obs):
                        pub = _public(r_obs, 7)
                        if cur_cond == "S15_wrist_zero":
                            pub["robot0_eye_in_hand_image"] = np.full_like(pub["robot0_eye_in_hand_image"], 128)
                        elif cur_cond == "S16_oracle_rect":
                            for k, cname in (("agentview_image", "agentview"), ("robot0_eye_in_hand_image", "robot0_eye_in_hand")):
                                pub[k] = _rectify_s16_image(
                                    pub[k],
                                    s16_plan.get("fov_scale", 1.12),
                                    s16_plan.get("image_shift_xy", (0.05, -0.05)),
                                    s16_plan.get("crop_ratio", 0.0),
                                    s16_plan.get("distortion_strength", 0.0),
                                    base_cam_poses[cname]["fovy"],
                                )
                        elif cur_cond == "S16_selfcal_rect":
                            z, dx, dy = selfcal_params
                            for k in ("agentview_image", "robot0_eye_in_hand_image"):
                                pub[k] = _rectify_affine(pub[k], z, dx, dy)
                        return pub

                    obs = _process_obs(raw)
                    t = 0
                    payload = dict(obs=obs, selfcal=selfcal_params)
                elif op == "chunk":
                    rewards = []
                    for a in v:
                        raw, reward, done, _ = env.step(np.asarray(a, dtype=np.float64))
                        t += 1
                        success = bool(env.check_success())
                        rewards.append(float(success))
                        if success or t == horizon:
                            break
                    obs = _process_obs(raw)
                    payload = dict(
                        obs=obs,
                        rewards=rewards,
                        success=bool(rewards[-1]),
                        truncated=(t == horizon and not rewards[-1]),
                    )
                elif op == "close":
                    conn.send((True, None))
                    break
                else:
                    raise ValueError(op)
                conn.send((True, payload))
        except EOFError:
            pass
        except Exception:
            conn.send((False, traceback.format_exc()))
        finally:
            for e in envs.values():
                e.close()


if __name__ == "__main__":
    run(int(sys.argv[1]), "--fixture" in sys.argv[2:])
