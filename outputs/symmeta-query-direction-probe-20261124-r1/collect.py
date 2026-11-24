"""Physical intervention and held fixed-command responses on the native OSC."""
import copy
from contextlib import contextmanager
import os
from pathlib import Path
import socket
import sys
from unittest.mock import patch

import mujoco
import numpy as np
from arm_env import ArmEnv
from ur5e_sim_worker import _probe_egl
from design import FAMILIES, HERE, SCHEDULES, TASKS, inputs, maps, save, schedule, source

sys.path.insert(0, str(HERE.parent / "frozen-vla-native-controller-20261123-r1"))
from native import restore, snapshot


def pack(env, obs):
    return (snapshot(env._env), env._step, copy.deepcopy(env._previous_public),
            env._copy_tool(), copy.deepcopy(obs))


def rewind(env, saved):
    sim, step, public, tool, obs = saved
    restore(env._env, sim)
    env._step, env._done = step, False
    env._previous_public, env._tool = copy.deepcopy(public), copy.deepcopy(tool)
    return obs


def pixels(obs, name):
    image = obs[name]
    return image.reshape(64, 4, 64, 4, 3).mean(axis=(1, 3)).round().astype(np.uint8)


class Recorder:
    def __init__(self, env):
        self.env = env
        r = env._env.robots[0]
        self.controller = r.controller
        self.sim = env._env.sim
        self.site = r.eef_site_id
        self.idx = r._ref_joint_pos_indexes
        self.scratch = mujoco.MjData(self.sim.model._model)
        self.scratch.qpos[:] = self.sim.data.qpos

    def observe(self, obs):
        # FK receives sampled public arm encoders. Paired public eef records its sensor discrepancy.
        self.scratch.qpos[self.idx] = obs["robot0_joint_pos"]
        mujoco.mj_kinematics(self.sim.model._model, self.scratch)
        return dict(time=self.sim.data.time, q=obs["robot0_joint_pos"].copy(),
                    dq=obs["robot0_joint_vel"].copy(), jaw=obs["robot0_gripper_qpos"].copy(),
                    quat=obs["robot0_eef_quat_xyzw"].copy(), public_xyz=obs["robot0_eef_pos"].copy(),
                    fk=self.scratch.site_xpos[self.site].copy())

    def run(self, path, initial_obs, actions, matrix):
        rows = [self.observe(initial_obs)]
        cameras = [(0, pixels(initial_obs, "agentview_image"), pixels(initial_obs, "robot0_eye_in_hand_image"))]
        issued, physical, goals, rewards, flags = [], [], [], [], []
        obs = initial_obs
        set_goal = self.controller.set_goal
        def record_goal(*args, **kwargs):
            result = set_goal(*args, **kwargs)
            goals.append((self.controller.goal_pos - self.controller.ee_pos).copy())
            return result
        try:
            with patch.object(self.controller, "set_goal", side_effect=record_goal):
                for n, action in enumerate(actions, 1):
                    local = np.asarray(action, dtype=np.float32)
                    ctrl = local.copy()
                    ctrl[:2] = matrix @ local[:2]
                    assert np.abs(ctrl).max() <= 1
                    step = self.env.step(ctrl)
                    obs = step["observation"]
                    issued.append(local.copy())
                    physical.append(step["issued_action"].copy())
                    rewards.append(step["reward"])
                    flags.append([step["terminated"], step["truncated"]])
                    rows.append(self.observe(obs))
                    if n % 5 == 0 or any(flags[-1]):
                        cameras.append((n, pixels(obs, "agentview_image"), pixels(obs, "robot0_eye_in_hand_image")))
                    assert len(goals) == n
                    np.testing.assert_array_equal(physical[-1], ctrl)
                    np.testing.assert_allclose(goals[-1], ctrl[:3] * self.controller.output_max[:3], rtol=0, atol=1e-8)
                    if any(flags[-1]):
                        break
        finally:
            data = {k:np.array([r[k] for r in rows]) for k in rows[0]}
            data.update(issued=np.asarray(issued), physical=np.asarray(physical), goal_m=np.asarray(goals),
                        camera_steps=np.array([s[0] for s in cameras]), external=np.stack([s[1] for s in cameras]),
                        wrist=np.stack([s[2] for s in cameras]), reward=np.asarray(rewards), terminal=np.asarray(flags))
            np.savez_compressed(path, **data)
        return obs, data


def run(out, arm, task_index, refs, amp):
    suite, task = TASKS[task_index]
    sup, x, frame0, obj, cmd, xyz, queryref = inputs(refs, arm, suite, task)
    folder = out / f"{arm}-{suite}-{task}"
    folder.mkdir()
    np.savez_compressed(folder / "archive-controls.npz", support_chunk_means=x, query=cmd, archived_query_xyz=xyz)
    matrices = list(maps(task_index))
    start = 5 * max(0, len(cmd) // 10 - 2)
    meta = dict(arm=arm, suite=suite, task=task, ordinal=task_index, query=queryref, support=sup,
                archived_query_reset=obj["full_reset"], archived_model=obj["model"], midpoint_start=start,
                maps={f:dict(A=a.tolist(), draw=draw) for f,a,draw in matrices}, source_amplitude=amp)
    save(folder/"case.json", meta)
    with ArmEnv(arm, suite, task) as env:
        first = env.reset(11, sup["seed"])
        assert first["model_identity"]["xml_sha256"] == obj["model"]["xml_sha256"]
        assert first["model_identity"]["controller"] == "OSC_POSE" and first["model_identity"]["control_freq"] == 20
        meta.update(support_reset=first["environment_reproducibility"], model=first["model_identity"],
                    support_start_xyz_delta_mm=(1000*(first["observation"]["robot0_eef_pos"]-frame0[:3])).tolist())
        save(folder/"case.json", meta)
        rec = Recorder(env)
        saved = pack(env, first["observation"])
        for fam, a, _ in matrices:
            baseline = None
            for name in SCHEDULES:
                obs = rewind(env, saved)
                actions = (v for chunk in schedule(iter(x), name, amp) for v in [chunk]*5)
                _, data = rec.run(folder/f"{fam}-{name}.npz", obs, actions, a)
                assert len(data["issued"]) == 15
                if baseline is None:
                    baseline = data
                else:
                    for key in ("fk","q","dq","public_xyz","jaw"):
                        np.testing.assert_allclose(data[key][:6], baseline[key][:6], rtol=0, atol=1e-7)
                    for key in ("external","wrist"):
                        np.testing.assert_array_equal(data[key][:2], baseline[key][:2])
            print(arm, suite, task, fam, "support complete", flush=True)
        first = env.reset(14, obj["case"]["seed"])
        meta["held_reset"] = first["environment_reproducibility"]
        for k in ("released_row_fingerprint","bddl_sha256","scene_fingerprint","perturbation_plan_fingerprint"):
            assert meta["held_reset"][k] == obj["full_reset"][k]
        assert meta["held_reset"]["released_row_fingerprint"] != meta["support_reset"]["released_row_fingerprint"]
        save(folder/"case.json", meta)
        original = pack(env, first["observation"])
        rec = Recorder(env)
        for fam, a, _ in matrices:
            for label, axis in [("x+", [.5,0]), ("x-", [-.5,0]), ("y+", [0,.5]), ("y-", [0,-.5]), ("zero",[0,0])]:
                u = cmd[:5].copy()
                u[:,:2] = axis
                _, data = rec.run(folder/f"{fam}-check-{label}.npz", rewind(env,original), u, a)
                assert len(data["issued"]) == 5
        obs, prefix = rec.run(folder/"native-prefix.npz", rewind(env, original), cmd[:start], np.eye(2))
        if len(prefix["issued"]) == start and not prefix["terminal"][-1].any():
            mid = pack(env, obs)
            for fam, a, _ in matrices:
                rec.run(folder/f"{fam}-held.npz", rewind(env, mid), cmd[start:start+20], a)
        meta["complete"] = True
        save(folder/"case.json", meta)
        print(arm, suite, task, "complete", flush=True)


if __name__ == "__main__":
    out, cpu, arm = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    assert os.sched_getaffinity(0) == {cpu} and os.environ["CUDA_VISIBLE_DEVICES"] == ""
    renderer = _probe_egl(True)
    assert "llvmpipe" in renderer[1].lower()
    refs, fit, amp = source()
    save(out/f"runtime-{arm}.json", dict(renderer=renderer, host=socket.gethostname(), cpu=cpu, arm=arm,
         source_hash=refs["source_fit_sha256"], source_amplitude=amp, prior=fit["prior"], residual=fit["residual_power"]))
    for index in range(len(TASKS)):
        run(out, arm, index, refs, amp)
