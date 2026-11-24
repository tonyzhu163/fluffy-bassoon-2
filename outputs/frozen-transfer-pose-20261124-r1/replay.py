"""Read physical controller goals and endpoints while replaying untouched raw commands."""
import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from arm_env import ArmEnv
from feasibility import get_reference
from geometry import Geometry
from panel import cases, key
from ur5e_sim_worker import _probe_egl


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def angle(a, b):
    return Rotation.from_matrix(a @ b.T).magnitude()


def replay(spec, out, args):
    arm = spec["arm"]
    case = f"{key(spec)}-{arm}"
    source = args.tapes / f"{case}.json"
    archive = json.loads(source.read_bytes())
    assert archive["case"] == spec and digest(source) == args.hashes[source.name]
    with ArmEnv(arm, spec["suite"], spec["task_id"]) as env:
        first = env.reset(spec["init_state_id"], spec["seed"])
        reset = first["environment_reproducibility"]
        assert reset == get_reference(arm, spec["suite"], spec["task_id"], spec["init_state_id"])["reset"]
        for field in ("released_row_fingerprint", "scene_fingerprint", "bddl_sha256", "perturbation_plan_fingerprint"):
            assert reset[field] == archive["full_reset"][field]
        assert first["model_identity"]["xml_sha256"] == archive["model"]["xml_sha256"]
        g = Geometry(env._env)
        native = env._env
        robot = native.robots[0]
        c = robot.controller
        aid = robot._ref_joint_actuator_indexes
        pad_ids = [g.m.geom(f"gripper0_finger{i}_pad_collision").id for i in (1, 2)]

        def sample():
            # Restore the integration-boundary qpos into scratch. sim.data.site_xpos can lag this clock.
            g.d.qpos[:] = native.sim.data.qpos
            mujoco.mj_kinematics(g.m, g.d)
            return (g.d.qpos[g.adr].copy(), g.tcp(), g.d.geom_xpos[pad_ids].mean(axis=0))

        rows = []
        prev_obs = first["observation"]
        done = False
        blank = np.zeros((256, 256, 3), np.uint8)
        with patch.object(native.sim, "render", return_value=blank):
            for k, chunk in enumerate(archive["chunks"]):
                if done:
                    break
                for j, (a, xyz, jaw) in enumerate(zip(chunk["cmd"], chunk["xyz"], chunk["jaw"], strict=True)):
                    cmd = np.array(a, np.float32)
                    q0, t0, pad0 = sample()
                    last_goal = c.goal_ori.copy()
                    step = env.step(cmd)
                    assert np.array_equal(cmd, step["issued_action"])
                    q1, t1, pad1 = sample()
                    obs = step["observation"]
                    gp = np.array(c.goal_pos)
                    gr = np.array(c.goal_ori)
                    scaled = c.scale_action(cmd[:6])
                    expected_rot = (Rotation.from_rotvec(scaled[3:]).as_matrix() @ t0[:3,:3]
                                    if np.any(scaled[3:]) else last_goal)
                    pad_local = t0[:3,:3].T @ (pad0 - t0[:3,3])
                    target_pad = gp + gr @ pad_local
                    end_chunk = j == len(chunk["cmd"])-1
                    rows.append(dict(cmd=cmd, chunk=k, q=q0, q_end=q1,
                                     public_q=prev_obs["robot0_joint_pos"], public_p=prev_obs["robot0_eef_pos"],
                                     p=t0[:3,3], R=t0[:3,:3], goal_p=gp, goal_R=gr,
                                     end_p=t1[:3,3], end_R=t1[:3,:3], pad=pad0, goal_pad=target_pad, end_pad=pad1,
                                     pos_goal_reconstruction_m=np.linalg.norm(t0[:3,3]+scaled[:3]-gp),
                                     ori_goal_reconstruction_rad=angle(gr, expected_rot),
                                     native_reward=step["reward"], recorded_tcp_delta_m=np.linalg.norm(obs["robot0_eef_pos"]-xyz),
                                     recorded_finger_max_delta_m=np.max(np.abs(obs["robot0_gripper_qpos"]-jaw)),
                                     recorded_chunk_angle_rad=(Rotation.from_quat(obs["robot0_eef_quat_xyzw"]) *
                                        Rotation.from_quat(chunk["q1"]).inv()).magnitude() if end_chunk else np.nan))
                    prev_obs = obs
                    done = step["truncated"] or step["terminated"]
                    if done:
                        break
        data = {k: np.array([r[k] for r in rows]) for k in rows[0]}
        np.savez_compressed(out / f"{case}.npz", **data)
        rec = archive["record"]
        model = first["model_identity"]
        info = dict(case=spec, tape=str(source), tape_sha256=digest(source), recording_job="job-xaugtearcacf",
                    opening_proposal_sha256=rec["opening_proposal_sha256"], action_schema=rec["reset"]["action_schema"],
                    reset=reset, recorded_settled_hash=archive["full_reset"]["initial_state_fingerprint"],
                    model=model, controller_class=type(c).__name__, arm_actuators=[g.m.actuator(i).name for i in aid],
                    arm_motor_ctrlrange=g.m.actuator_ctrlrange[aid].tolist(), action_dimensions=robot.action_dim,
                    simulator_actuators=g.m.nu, grip_command_dof=robot.gripper.dof,
                    arm_limited=g.limited.tolist(), compiled_arm_ranges=g.m.jnt_range[g.joints].tolist(),
                    input_max=c.input_max.tolist(), output_max=c.output_max.tolist(), kp=c.kp.tolist(), kd=c.kd.tolist(),
                    physics_dt=g.m.opt.timestep, sample_dt=1 / model["control_freq"],
                    native_controls=rec["controls"], replay_controls=len(rows), recorded_success=rec["success"],
                    success=bool(step["success"]), terminated=bool(step["terminated"]), truncated=bool(step["truncated"]),
                    reward_tape_equal=data["native_reward"].tolist()==rec["native_rewards"],
                    terminal_flags_equal=all(step[k]==rec[k] for k in ("success","terminated","truncated")),
                    data_sha256=digest(out / f"{case}.npz"))
        save(out / f"{case}.json", info)
        print(case, len(rows), rec["controls"], info["success"], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=["Panda","Kinova3"])
    ap.add_argument("tapes", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    tree = Path(os.environ["POSE_CODE"])
    args.hashes = json.loads((tree / "outputs/frozen-vla-native-transfer-20261123-r1/gpu-evidence.json").read_bytes())["sha256"]
    renderer = _probe_egl(True)
    affinity = sorted(os.sched_getaffinity(0))
    assert set(affinity) <= {46,47} and os.environ["CUDA_VISIBLE_DEVICES"] == "" and "llvmpipe" in renderer[1].lower()
    receipt = dict(host=socket.gethostname(), platform=platform.platform(), affinity=affinity, renderer=renderer,
                   source_revision=subprocess.check_output(["git","-C",tree,"rev-parse","HEAD"],text=True).strip(),
                   script_sha256=digest(Path(__file__)), gpu_visible=os.environ["CUDA_VISIBLE_DEVICES"], gpu_hours=0,
                   versions={k:version(k) for k in ("mujoco","robosuite","scipy","numpy")})
    save(args.out / f"receipt-{args.arm}.json", receipt)
    for spec in cases(args.arm):
        replay(spec, args.out, args)
