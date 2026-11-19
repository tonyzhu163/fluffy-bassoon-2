"""Audit a native, camera-only optical-axis roll at LIBERO public frame zero."""

import hashlib
import json
import os
from pathlib import Path
import sys
from importlib.metadata import version

import mujoco
import numpy as np
from OpenGL import GL
from PIL import Image
from libero_remote.environment.libero import LiberoEnvironmentAdapter
from libero_remote.environment.observations import normalize_standard_observation
from libero_remote.environment.task_manifest import TaskManifest


ANGLES = (0, 90, -90)
CAMERAS = ("agentview", "robot0_eye_in_hand")
SIZE = 128
TOL = 1e-10
PROPRIO = {
    "robot0_joint_pos", "robot0_joint_vel", "robot0_eef_pos",
    "robot0_eef_quat_xyzw", "robot0_gripper_qpos",
}


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def camera_pose(data, cam, base):
    world_from_base = data.xmat[base].reshape(3, 3)
    base_from_world = world_from_base.T
    return (base_from_world @ data.cam_xmat[cam].reshape(3, 3),
            base_from_world @ (data.cam_xpos[cam] - data.xpos[base]))


def physical_hashes(model, data):
    flag = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, flag))
    mujoco.mj_getState(model, data, state, flag)
    return {k: digest(v) for k, v in {
        "integration": state, "body_xpos": data.xpos, "body_xmat": data.xmat,
    }.items()}


def refresh_public(world):
    raw = world._get_observations(force_update=True)
    raw = world.perturbation_runtime.transform_observation(raw, 0)
    return normalize_standard_observation(raw, CAMERAS)


def feature(image):
    assert image.shape == (SIZE, SIZE, 3) and image.dtype == np.uint8
    return image.reshape(16, 8, 16, 8, 3).mean(axis=(1, 3))


def exposures(env, initial, folder):
    world = env._env.env
    sim = world.sim
    model, data = sim.model._model, sim.data._data
    cam = sim.model.camera_name2id(CAMERAS[0])
    base_name = world.robots[0].robot_model.root_body
    base = sim.model.body_name2id(base_name)
    assert base > 0 and model.cam_mode[cam] == int(mujoco.mjtCamLight.mjCAMLIGHT_FIXED)
    nominal_quat = model.cam_quat[cam].copy()
    nominal_R, nominal_t = camera_pose(data, cam, base)
    reset = {key: arr.copy() for key, arr in initial["observation"].items()}
    reset_hashes = {key: digest(value) for key, value in reset.items()}
    view = "agentview_image"
    assert set(reset) == PROPRIO | {name + "_image" for name in CAMERAS}
    reference = physical_hashes(model, data)
    vendor = GL.glGetString(GL.GL_VENDOR).decode()
    renderer = GL.glGetString(GL.GL_RENDERER).decode()
    assert "llvmpipe" in renderer.lower() and os.environ["CUDA_VISIBLE_DEVICES"] == ""
    angles, features = [], []
    for index, angle in enumerate((*ANGLES, 0)):
        # MuJoCo's forward viewing axis is camera-local -Z.
        a = np.deg2rad(angle)
        if angle:
            delta = np.array([np.cos(a / 2), 0., 0., -np.sin(a / 2)])
            mujoco.mju_mulQuat(model.cam_quat[cam], nominal_quat, delta)
        else:
            model.cam_quat[cam] = nominal_quat
        mujoco.mj_camlight(model, data)
        obs = refresh_public(world)
        R, t = camera_pose(data, cam, base)
        rel = nominal_R.T @ R
        expected = np.array([[np.cos(a), np.sin(a), 0.],
                             [-np.sin(a), np.cos(a), 0.], [0., 0., 1.]])
        np.testing.assert_allclose(rel, expected, atol=TOL, rtol=0)
        np.testing.assert_allclose(t, nominal_t, atol=TOL, rtol=0)
        np.testing.assert_allclose(R[:, 2], nominal_R[:, 2], atol=TOL, rtol=0)
        physics = physical_hashes(model, data)
        hashes = {k: digest(v) for k, v in obs.items()}
        assert physics == reference and set(obs) == set(reset)
        for key in set(reset) - {view}:
            assert hashes[key] == reset_hashes[key], key
        assert (hashes[view] == reset_hashes[view]) == (angle == 0)
        assert env._step_index == world._perturbation_step == 0
        changed = np.mean(np.any(obs[view] != reset[view], axis=-1))
        rec = dict(command_deg=angle, realized_deg=np.rad2deg(np.arctan2(rel[0, 1], rel[0, 0])),
                   base_R_cam=R.tolist(), base_t_cam=t.tolist(), relative_rotation=rel.tolist(),
                   world_cam_xmat=data.cam_xmat[cam].reshape(3, 3).tolist(),
                   model_cam_quat=model.cam_quat[cam].tolist(), physics=physics, public_sha256=hashes,
                   changed_pixel_fraction=changed, rotation_max_abs_error=np.max(np.abs(rel - expected)),
                   center_displacement_m=np.linalg.norm(t - nominal_t),
                   optical_axis_displacement=np.linalg.norm(R[:, 2] - nominal_R[:, 2]))
        if index < len(ANGLES):
            rec["image"] = f"roll{angle:+d}.png"
            Image.fromarray(obs[view]).save(folder / rec["image"])
            features.append(feature(obs[view]))
        angles.append(rec)
    assert len({r["public_sha256"][view] for r in angles[:3]}) == 3
    return np.stack(features), dict(renderer=renderer, vendor=vendor, base_body=base_name,
                                   world_base_xmat=data.xmat[base].reshape(3, 3).tolist(),
                                   world_base_xpos=data.xpos[base].tolist(), time_s=float(data.time),
                                   reset_physics=reference, reset_public_sha256=reset_hashes, exposures=angles)


def task_frame(split, suite, task, index, manifest, out):
    folder = out / f"{split}-{suite}-{task}"
    folder.mkdir()
    reset = 0 if split == "fit" else 13
    seed = 19111900 + index
    key = (suite, task, "BASE")
    entry = manifest.require(*key)
    assert entry["case_id"] == "BASE"
    spec = dict(suite=suite, task_id=task, variant="BASE", init_state_id=reset, seed=seed,
                camera_names=list(CAMERAS), camera_height=SIZE, camera_width=SIZE,
                control_mode="relative", max_steps=entry["max_steps"],
                _bddl_path=str(manifest.resolve_bddl(*key)))
    env = LiberoEnvironmentAdapter()
    try:
        initial = env.create(spec)
        identity = initial["environment_reproducibility"]
        assert initial["worker_step_index"] == 0 and identity["perturbation_plan"]["case_id"] == "BASE"
        feat, record = exposures(env, initial, folder)
        dump(folder / "readback.json", dict(spec=spec, initial_state=identity["initial_state_fingerprint"],
                                            plan_fingerprint=identity["perturbation_plan_fingerprint"], **record))
    finally:
        env.close()
    print(split, suite, task, "qualified", flush=True)
    return feat


def predict(image_feature, templates):
    distances = np.mean((templates - image_feature) ** 2, axis=(1, 2, 3))
    best = int(np.argmin(distances))
    sorted_dist = np.sort(distances)
    return best, distances.tolist(), float(sorted_dist[1] - sorted_dist[0])


def pairs(by_suite):
    return [(suite, task) for suite, ids in sorted(by_suite.items()) for task in ids]


def main():
    out, release, split_path = map(Path, sys.argv[1:])
    splits = json.loads(split_path.read_text())
    fit = pairs(splits["development_scaling"]["task_subsets"]["8"])
    dev = pairs(splits["development"])
    assert len(fit) == len(dev) == len(set(fit)) == len(set(dev)) == 8
    assert set(fit) <= set(pairs(splits["train"])) and set(fit).isdisjoint(dev)
    assert (set(fit) | set(dev)).isdisjoint(pairs(splits["test"]))
    manifest = TaskManifest.load(str(release / "assets/task_manifest.json"))
    assert manifest.sha256 == splits["manifest_sha256"]
    dump(out / "protocol.json", dict(split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
                                    manifest_sha256=manifest.sha256, fit=fit, development=dev,
                                    angles=ANGLES, python=sys.version, packages={name: version(name)
                                    for name in ("mujoco", "robosuite", "numpy", "Pillow")}))
    templates = np.mean([task_frame("fit", *key, i, manifest, out) for i, key in enumerate(fit)], axis=0)
    np.save(out / "templates.npy", templates)
    templates.flags.writeable = False
    matrix = np.zeros((3, 3), dtype=int)
    cleared_hits, rows = 0, []
    for i, (suite, task) in enumerate(dev):
        frames = task_frame("dev", suite, task, len(fit) + i, manifest, out)
        cleared = predict(frames[0], templates)[0]
        for label, frame in enumerate(frames):
            pred, mse, margin = predict(frame, templates)
            matrix[label, pred] += 1
            cleared_hits += int(cleared == label)
            error = abs((ANGLES[pred] - ANGLES[label] + 180) % 360 - 180)
            rows.append(dict(suite=suite, task=task, truth=ANGLES[label], prediction=ANGLES[pred],
                             mse=mse, runner_up_margin=margin, circular_error_deg=error,
                             nominal_frame_prediction=ANGLES[cleared]))
    perfect = sum(all(row["circular_error_deg"] == 0 for row in rows if (row["suite"], row["task"]) == key)
                  for key in dev)
    dump(out / "result.json", dict(angles=ANGLES, matrix_true_by_pred=matrix.tolist(), rows=rows,
                                  correct=int(np.trace(matrix)), total=len(rows), perfect_tasks=perfect,
                                  nominal_only_correct=cleared_hits, mean_error_deg=np.mean([r["circular_error_deg"]
                                  for r in rows]), min_margin=min(r["runner_up_margin"] for r in rows),
                                  templates_sha256=digest(templates)))
    print("complete", matrix.tolist(), flush=True)


if __name__ == "__main__":
    main()
