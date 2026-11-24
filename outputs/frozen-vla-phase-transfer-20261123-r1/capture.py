"""Re-enact recorded Kinova3 commands under native CPU physics. No policy inference or fitting."""
import argparse
import json
import os
import socket
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from arm_env import ArmEnv
from ur5e_sim_worker import _probe_egl
from common import save, sha, tiles

ROOT = Path("/mowenjie/fluffy-bassoon-2/outputs")
NATIVE = ROOT/"frozen-vla-native-transfer-20261123-r1/gpu01"
FEEDBACK = ROOT/"frozen-vla-tracking-feedback-20261123-r1/gpu01"


def record(path, layout, seed, same_start, out):
    tape = json.loads(path.read_bytes())
    spec = tape["case"]
    assert (spec["arm"], spec["suite"], spec["task_id"], spec["variant"]) == ("Kinova3","libero_goal",3,"BASE")
    donor = spec["init_state_id"]
    assert same_start == (layout == donor and seed == spec["seed"])
    commands = [(a, xyz, jaw, c["q1"] if i+1 == len(c["cmd"]) else None)
                for c in tape["chunks"] for i,(a,xyz,jaw) in enumerate(zip(c["cmd"], c["xyz"], c["jaw"], strict=True))]
    assert len(commands) == tape["record"]["controls"]
    name = f"row{layout}-tape{donor}"
    with ArmEnv("Kinova3","libero_goal",3) as arm:
        init = arm.reset(layout,seed)
        reset = init["environment_reproducibility"]
        if same_start:
            for k in ("bddl_sha256","released_row_fingerprint","scene_fingerprint","perturbation_plan_fingerprint"):
                assert reset[k] == tape["full_reset"][k], k
        native = arm._env
        mj = native.sim.model._model
        addr = mj.joint("wooden_cabinet_1_top_level").qposadr[0]
        # No labels or privileged state in tile extraction. Keep every native state for later scoring.
        q = [float(native.sim.data.qpos[addr])]
        samples = [tiles(init["observation"]["agentview_image"])]
        times = [0]
        rgb_shape = list(init["observation"]["agentview_image"].shape)
        blank = np.zeros(rgb_shape, np.uint8)
        original = native.sim.render
        def render(*args, **kwargs):
            if t % 5 == 0 and kwargs.get("camera_name") == "agentview":
                return original(*args, **kwargs)
            return blank
        tcp, fingers, angles, rewards = [], [], [], []
        with patch.object(native.sim, "render", side_effect=render):
            for t,(command,xyz,jaw,quat) in enumerate(commands, 1):
                step = arm.step(np.asarray(command,np.float32))
                assert t == step["worker_step_index"]
                obs = step["observation"]
                if t % 5 == 0:
                    pixels = obs["agentview_image"]
                    assert pixels.max() != pixels.min(), "Missing public RGB"
                    times.append(t)
                    samples.append(tiles(pixels))
                q.append(float(native.sim.data.qpos[addr]))
                rewards.append(step["reward"])
                if same_start:
                    tcp.append(float(np.linalg.norm(obs["robot0_eef_pos"] - xyz)))
                    fingers.append(float(np.max(np.abs(obs["robot0_gripper_qpos"] - jaw))))
                    if quat is not None:
                        angles.append(float((Rotation.from_quat(obs["robot0_eef_quat_xyzw"]) *
                                             Rotation.from_quat(quat).inv()).magnitude()))
                if step["terminated"] or step["truncated"]:
                    break
        np.savez_compressed(out/f"public-{name}.npz", t=times, tiles=samples)
        save(out/f"truth-{name}.json",dict(q=q, rewards=rewards, success=step["success"],
             truncated=step["truncated"], terminated=step["terminated"]))
        save(out/f"lineage-{name}.json",dict(input_path=str(path),sha256=sha(path),donor=spec,layout=layout,
             seed=seed, matched_original=same_start,reset=reset, model_identity=init["model_identity"],
             source_settled_reset=tape["full_reset"]["initial_state_fingerprint"],source_controls=len(commands),
             controls=t, source_native_success=tape["record"]["success"], source_native_return=tape["record"]["native_return"],
             reward_tape_equal=(rewards == tape["record"]["native_rewards"]) if same_start else None,
             tcp_error_m=tcp, finger_max_error_m=fingers, rotation_boundary_error_rad=angles))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("group",choices=["known","crossed","live"])
    p.add_argument("out",type=Path)
    a = p.parse_args()
    a.out.mkdir(exist_ok=True)
    renderer = _probe_egl(True)
    assert "llvmpipe" in renderer[1].lower() and not os.environ["CUDA_VISIBLE_DEVICES"]
    cpus = sorted(os.sched_getaffinity(0))
    assert len(cpus) <= 2
    save(a.out/"runtime.json",dict(renderer=renderer,cpus=cpus,host=socket.gethostname()))
    if a.group == "live":
        for row in (16,17):
            record(FEEDBACK/f"libero_goal-3-{row}-raw.json",row,41810300+row,True,a.out)
    else:
        for row in ((14,15) if a.group == "known" else (30,31,32,33)):
            for donor in ((row,) if a.group == "known" else (14,15)):
                record(NATIVE/f"libero_goal-3-{donor}-Kinova3.json", row,
                       (41810300 if a.group == "known" else 42710300)+row, a.group == "known", a.out)
