"""Render the two cameras independently at a fixed, settled LIBERO task state."""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import mujoco
import numpy as np
from OpenGL import GL
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

PANELS = Path(__file__).resolve().parents[2] / "libero-xemb-camera-20261119-r1/cpu01/protocol.json"
CAMERAS = {"agent": "agentview", "wrist": "robot0_eye_in_hand"}
SENSORS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")


def digest(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def physics(model, data):
    flag = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, flag))
    mujoco.mj_getState(model, data, state, flag)
    return digest(state)


def images_and_sensors(env):
    obs = env.env._get_observations(force_update=True)
    values = {k: obs[k].copy() for k in SENSORS}
    values.update({k: obs[name + "_image"].copy() for k, name in CAMERAS.items()})
    return values


def render(out, split, suite, task_id, seed):
    random.seed(seed)
    np.random.seed(seed)
    tasks = benchmark.get_benchmark(suite)()
    task = tasks.get_task(task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    try:
        env.seed(seed)
        env.reset()
        reset = 0 if split == "fit" else 13
        env.set_init_state(tasks.get_task_init_states(task_id)[reset])
        for _ in range(10):
            env.step([0, 0, 0, 0, 0, 0, -1])
        renderer = GL.glGetString(GL.GL_RENDERER).decode()
        assert "llvmpipe" in renderer.lower(), renderer
        model, data = env.sim.model._model, env.sim.data._data
        orig_state = physics(model, data)
        base = images_and_sensors(env)
        values = dict(base)
        poses = {}
        if split == "development":
            for key, name in CAMERAS.items():
                cam = env.sim.model.camera_name2id(name)
                quat = model.cam_quat[cam].copy()
                rotation = data.cam_xmat[cam].reshape(3, 3).copy()
                center = data.cam_xpos[cam].copy()
                assert model.cam_mode[cam] == int(mujoco.mjtCamLight.mjCAMLIGHT_FIXED)
                for deg in (-90, 90, 0):
                    a = np.deg2rad(deg)
                    delta = np.array([np.cos(a/2), 0., 0., -np.sin(a/2)])
                    mujoco.mju_mulQuat(model.cam_quat[cam], quat, delta)
                    mujoco.mj_camlight(model, data)
                    exposed = images_and_sensors(env)
                    rel = rotation.T @ data.cam_xmat[cam].reshape(3, 3)
                    np.testing.assert_allclose(rel, [[np.cos(a), np.sin(a), 0.],
                                                     [-np.sin(a), np.cos(a), 0.],
                                                     [0., 0., 1.]], atol=1e-10, rtol=0)
                    np.testing.assert_array_equal(data.cam_xpos[cam], center)
                    assert physics(model, data) == orig_state
                    for field, arr in exposed.items():
                        if field != key or not deg:
                            np.testing.assert_array_equal(arr, base[field])
                    if deg:
                        assert not np.array_equal(exposed[key], base[key])
                        values[f"{key}{deg:+d}"] = exposed[key]
                        poses[f"{key}{deg:+d}"] = float(np.rad2deg(np.arctan2(rel[0, 1], rel[0, 0])))
        task_key = f"{split}-{suite}-{task_id}"
        np.savez_compressed(out / f"{task_key}.npz", **values)
        return dict(key=task_key, task=task.language, seed=seed, reset=reset, physics=orig_state,
                    renderer=renderer, camera_roll_deg=poses, sha256={k:digest(v) for k,v in values.items()})
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(PANELS.read_text())
    rows = []
    assert os.environ["PYTHONHASHSEED"] == "0"
    for split in ("fit", "development"):
        for suite, task_id in protocol[split]:
            rows.append(render(args.output, split, suite, task_id, 19111900 + len(rows)))
            print(rows[-1]["key"], flush=True)
    (args.output / "frames.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
