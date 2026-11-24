"""CPU diagnostic: inspect S15 & S16 camera geometry, pixel shifts, and wrist visibility across 8 tasks."""
import json
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
from libero.libero import benchmark
from libero.libero.envs.env_wrapper import ControlEnv
from libero_remote.environment.libero import _settle_environment
from sim_reset import load_panda_row

TASKS = [
    ("libero_spatial", 0),
    ("libero_spatial", 8),
    ("libero_object", 1),
    ("libero_object", 8),
    ("libero_goal", 3),
    ("libero_goal", 6),
    ("libero_10", 5),
    ("libero_10", 8),
]
CASES = ("BASE", "S15", "S16")
CAMERAS = ("agentview", "robot0_eye_in_hand")


def project_point(model, data, cam_name, pt_world, width=256, height=256):
    cid = model.camera_name2id(cam_name)
    pos = data.cam_xpos[cid]
    mat = data.cam_xmat[cid].reshape(3, 3)  # columns are camera x(right), y(up), z(back)
    fovy = float(model.cam_fovy[cid])
    f = 0.5 * height / np.tan(np.deg2rad(fovy) / 2.0)
    p_cam = mat.T @ (pt_world - pos)
    if p_cam[2] >= -1e-6:
        return None
    u = width / 2.0 + f * (p_cam[0] / (-p_cam[2]))
    v = height / 2.0 - f * (p_cam[1] / (-p_cam[2]))
    in_frame = (0.0 <= u < width) and (0.0 <= v < height)
    return dict(u=float(u), v=float(v), depth=float(-p_cam[2]), in_frame=bool(in_frame))


def main(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    rel = Path(os.environ["CROSSARM_RELEASE"])
    manifest = json.loads((rel / "assets/task_manifest.json").read_text())["tasks"]
    by_key = {(t["suite"], t["base_task_id"], t["case_id"]): t for t in manifest}

    records = []
    for suite, tid in TASKS:
        row = benchmark.get_benchmark_dict()[suite]().get_task_init_states(tid)[13]
        base_cam = {}
        for cid in CASES:
            entry = by_key[(suite, tid, cid)]
            bddl = rel / "assets" / entry["bddl_path"]
            np.random.seed(42)
            env = ControlEnv(
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
            try:
                env.seed(42)
                env.reset()
                load_panda_row(env, row)
                _settle_environment(env)
                env.set_perturbation_seed(42)
                obs = env.begin_perturbation_rollout()
                plan = env.get_perturbation_plan()
                m, d = env.sim.model, env.sim.data
                eef = d.site_xpos[env.robots[0].eef_site_id].copy()
                cam_info = {}
                for cname in CAMERAS:
                    cam_id = m.camera_name2id(cname)
                    proj = project_point(m, d, cname, eef)
                    cam_info[cname] = dict(
                        cam_pos_local=m.cam_pos[cam_id].tolist(),
                        cam_xpos_world=d.cam_xpos[cam_id].tolist(),
                        fovy=float(m.cam_fovy[cam_id]),
                        eef_proj=proj,
                    )
                if cid == "BASE":
                    base_cam = cam_info
                    base_imgs = {k: obs[f"{k}_image"].copy() for k in CAMERAS}
                    px_shift = {k: 0.0 for k in CAMERAS}
                    rgb_mae = {k: 0.0 for k in CAMERAS}
                else:
                    px_shift = {}
                    rgb_mae = {}
                    for cname in CAMERAS:
                        p0 = base_cam[cname]["eef_proj"]
                        p1 = cam_info[cname]["eef_proj"]
                        if p0 and p1:
                            px_shift[cname] = float(np.hypot(p1["u"] - p0["u"], p1["v"] - p0["v"]))
                        else:
                            px_shift[cname] = None
                        rgb_mae[cname] = float(
                            np.mean(np.abs(obs[f"{cname}_image"].astype(np.float32) - base_imgs[cname].astype(np.float32)))
                        )
                rec = dict(
                    suite=suite,
                    task_id=tid,
                    case_id=cid,
                    params=plan.get("params", {}),
                    targets=plan.get("targets", []),
                    cams=cam_info,
                    eef_px_shift=px_shift,
                    rgb_mae=rgb_mae,
                )
                records.append(rec)
                if cid != "BASE":
                    w_in = cam_info["robot0_eye_in_hand"]["eef_proj"]["in_frame"] if cam_info["robot0_eye_in_hand"]["eef_proj"] else False
                    a_in = cam_info["agentview"]["eef_proj"]["in_frame"] if cam_info["agentview"]["eef_proj"] else False
                    print(
                        f"{suite}/{tid} {cid}: agent_shift={px_shift['agentview']}px (in={a_in}) "
                        f"wrist_shift={px_shift['robot0_eye_in_hand']}px (in={w_in}) "
                        f"rgb_mae=({rgb_mae['agentview']:.1f}, {rgb_mae['robot0_eye_in_hand']:.1f})",
                        flush=True,
                    )
            finally:
                env.close()

    (out_dir / "s15_s16_diag.json").write_text(json.dumps(records, indent=2) + "\n")
    print("Wrote", out_dir / "s15_s16_diag.json", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
