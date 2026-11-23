"""Read the actual drawer policy trajectories by deterministic replay of existing raw action tapes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from arm_env import ArmEnv
from feasibility import ARMS, BOWL, DRAWER, TASKS
from geometry import Geometry, body_pose
from ur5e_sim_worker import _probe_egl


def run(tapes, output):
    renderer = _probe_egl(True)
    assert "llvmpipe" in renderer[1].lower() and not os.environ["CUDA_VISIBLE_DEVICES"]
    assert len(os.sched_getaffinity(0)) <= 2
    for layout in (14, 15):
        for arm in ARMS:
            f = tapes / f"libero_goal-3-{layout}-{arm}.json"
            raw = f.read_bytes()
            tape = json.loads(raw)
            spec = tape["case"]
            assert (spec["arm"], spec["init_state_id"], spec["suite"], spec["task_id"]) == (arm, layout, "libero_goal", 3)
            result = dict(source=f.name, sha256=hashlib.sha256(raw).hexdigest(), frames=[], renderer=renderer)
            with ArmEnv(arm, spec["suite"], spec["task_id"]) as env:
                initial = env.reset(layout, spec["seed"])
                assert initial["environment_reproducibility"] == tape["full_reset"]
                g = Geometry(env._env)
                drawer = g.m.joint("wooden_cabinet_1_top_level")
                result.update(reset=tape["record"]["reset"]["physics"], joint_range=drawer.range.tolist(),
                              drawer_initial=float(g.d.qpos[drawer.qposadr[0]]),
                              bowl_xyz_initial=body_pose(g.m, g.d, BOWL)[:3, 3].tolist())
                result["joint_limit_info"] = {g.m.joint(i).name: g.m.jnt_range[i].tolist()
                                              for i in g.joints[g.limited]}
                # Actions are fixed. Skip pixel generation after the native reset. Recheck all recorded motion.
                blank = np.zeros((256, 256, 3), np.uint8)
                pos_err, jaw_err, quat_err, rewards = [], [], [], []
                with patch.object(env._env.sim, "render", return_value=blank):
                    for chunk in tape["chunks"]:
                        for action, xyz, jaw in zip(chunk["cmd"], chunk["xyz"], chunk["jaw"], strict=True):
                            step = env.step(np.array(action, np.float32))
                            obs = step["observation"]
                            pos_err.append(float(np.linalg.norm(obs["robot0_eef_pos"] - xyz)))
                            jaw_err.append(float(np.max(np.abs(obs["robot0_gripper_qpos"] - jaw))))
                            rewards.append(step["reward"])
                            g.d.qpos[:] = env._env.sim.data.qpos
                            g.forward()
                            q = g.d.qpos[g.adr]
                            margins = np.minimum(g.hi-q, q-g.lo)
                            limiting_joint = int(g.joints[np.argmin(margins)])
                            result["frames"].append(dict(step=step["worker_step_index"], q=q.tolist(),
                                min_limited_margin_rad=float(min(margins)), nearest_limit=g.m.joint(limiting_joint).name,
                                drawer_q=float(g.d.qpos[drawer.qposadr[0]]), tcp=g.tcp()[:3, 3].tolist(),
                                fingers=g.d.qpos[g.fingers].tolist(), bowl_xyz=body_pose(g.m, g.d, BOWL)[:3, 3].tolist(),
                                contacts=g.contacts(None)))
                        actual = obs["robot0_eef_quat_xyzw"]
                        target = np.array(chunk["q1"])
                        quat_err.append(float(min(np.linalg.norm(actual-target), np.linalg.norm(actual+target))))
                record = tape["record"]
                assert rewards == record["native_rewards"] and len(rewards) == record["controls"]
                assert all(step[k] == record[k] for k in ("success", "terminated", "truncated"))
                result.update(success=step["success"], controls=len(rewards), reward_sum=sum(rewards),
                              max_tcp_delta_m=max(pos_err), max_jaw_coord_delta_m=max(jaw_err),
                              max_boundary_quat_l2=max(quat_err))
                # Stored public signals are float32 arrays. Require their exact wire values.
                if max(max(pos_err), max(jaw_err), max(quat_err)) != 0:
                    raise ValueError(f"Replay disagrees with action tape {result['max_tcp_delta_m']}")
            (output / f"raw-drawer-{layout}-{arm}.json").write_text(json.dumps(result, separators=(",", ":")) + "\n")
            print(f.name, len(rewards), result["success"], "exact motion and rewards", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    run(args.input, args.output)
