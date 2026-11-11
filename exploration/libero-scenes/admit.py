"""Simulator admission for scenes from `scenes.py`. Never loads a policy.

Run inside an allocated Linux job with the qualified Behavior Prompting
runtime. For every scene it builds the environment from the generated task
file, lets LIBERO place the objects, holds the arm still, and records whether
the scene is physically usable:

    python admit.py --source <behavior_prompting checkout> --scenes <dir> --output <new dir>

The simulator calls follow `outputs/libero-gen-custom-layouts-20261111-r1/geometry.py`,
which ran on Baige. This file itself has only been syntax-checked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

SIZE = 128  # camera pixels; overridden by --image-size
HOLD_STEPS = 20


def sha(value) -> str:
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def position(env, name):
    return env.sim.data.body_xpos[env.env.obj_body_id[name]].copy()


def set_position(env, name, xy, z):
    import numpy as np
    joint = env.env.objects_dict[name].joints[0]
    qpos = np.asarray(env.sim.data.get_joint_qpos(joint)).copy()
    qpos[:2], qpos[2] = xy, z
    env.sim.data.set_joint_qpos(joint, qpos)
    env.sim.data.set_joint_qvel(joint, np.zeros(6))


def project(env, xyz):
    """Pixel (u, v) of a world point in the agent-view camera."""
    import numpy as np
    sim = env.sim
    cid = sim.model.camera_name2id("agentview")
    local = sim.data.cam_xmat[cid].reshape(3, 3).T @ (np.asarray(xyz) - sim.data.cam_xpos[cid])
    depth = -float(local[2])
    assert depth > 0
    f = SIZE / 2 / np.tan(np.deg2rad(sim.model.cam_fovy[cid]) / 2)
    return [float(SIZE / 2 + f * local[0] / depth), float(SIZE / 2 - f * local[1] / depth)]


def hold_action(env):
    """Absolute end-effector command that keeps the arm where it is, gripper open."""
    import numpy as np
    from robosuite.utils.transform_utils import mat2quat, quat2axisangle, quat2mat
    obs = env.env._get_observations()
    # Same controller-frame convention as the Behavior Prompting runner.
    rotation = quat2mat(obs["robot0_eef_quat"]) @ np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
    return np.concatenate((obs["robot0_eef_pos"], quat2axisangle(mat2quat(rotation)), [-1.0]))


def holds(env, predicate) -> bool:
    return bool(env.env._eval_predicate(list(predicate)))


def admit(env, scene, folder):
    import numpy as np
    from PIL import Image

    movable = [o["name"] for o in scene["objects"]]
    fixtures = [f"{f}_1" for f in scene["fixtures"]]
    placed = {n: position(env, n) for n in movable}
    action = hold_action(env)
    for _ in range(HOLD_STEPS):
        env.step(action)
    state = env.get_sim_state().copy()
    obs = env.prepare_observation(env.regenerate_obs_from_state(state))
    settled = {n: position(env, n) for n in movable + fixtures}
    pixels = {n: project(env, settled[n]) for n in movable + fixtures}

    errors = []
    drift = max(float(np.linalg.norm(settled[n][:2] - placed[n][:2])) for n in movable)
    # World minus table frame should be one constant shift if LIBERO honoured the requested positions.
    shift = np.array([settled[o["name"]][:2] - np.array(o["xy"]) for o in scene["objects"]])
    spread = float(np.abs(shift - shift.mean(0)).max())
    # Region sampling introduces real position variation. Record drift and
    # frame spread for review; neither an arbitrary drift cutoff nor equality
    # to the region centre establishes physical validity.
    if not np.isfinite(state).all():
        errors.append("nonfinite_simulator_state")
    names = movable + fixtures
    contacts = [[a, b] for i, a in enumerate(names) for b in names[i + 1:]
                if env.env.check_contact(env.env.get_object(a), env.env.get_object(b))]
    if contacts:
        errors.append("initial_object_or_fixture_contact")
    if any(not 0 <= v < SIZE for n in movable for v in pixels[n]):
        errors.append("movable_object_centre_outside_image")
    already = [b["predicate"] for b in scene["behaviours"] if holds(env, b["predicate"])]
    if already:
        errors.append("behaviour_true_before_any_action")

    # Positive control for object targets: a bowl dropped on a receptacle satisfies that predicate
    # and no other receptacle's. Fixture and table targets are left to the first demonstrations.
    positive = []
    receptacles = [t["name"] for t in scene["targets"] if t["kind"] == "receptacle"]
    bowls = [o["name"] for o in scene["objects"] if o["role"] == "bowl"]
    for bowl in bowls:
        for target in receptacles:
            env.env.reset()  # clears robosuite's episode counter before restoring
            env.regenerate_obs_from_state(state)
            at = position(env, target)
            set_position(env, bowl, at[:2], at[2] + 0.08)
            env.sim.forward()
            for _ in range(2 * HOLD_STEPS):
                env.step(action)
            got = [int(holds(env, ["on", bowl, t])) for t in receptacles]
            positive.append(dict(bowl=bowl, target=target, values=got))
            if got != [int(t == target) for t in receptacles]:
                errors.append(f"positive_control_failed:{bowl}:{target}")
    env.env.reset()
    env.regenerate_obs_from_state(state)

    np.save(folder / "state.npy", state)
    Image.fromarray(obs["agentview_image"]).save(folder / "agentview.png")
    Image.fromarray(obs["robot0_eye_in_hand_image"]).save(folder / "wrist.png")
    return dict(index=scene["index"], seed=scene["seed"], bddl_sha256=scene["bddl_sha256"],
                world_positions={n: settled[n].tolist() for n in movable + fixtures}, pixels=pixels,
                table_to_world_shift=shift.mean(0).tolist(), shift_spread=spread, settling_drift=drift,
                contacts=contacts, initially_true=already, positive_controls=positive,
                state_sha256=sha(state), errors=errors, admitted=not errors)


def main() -> None:
    global SIZE
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", type=Path, required=True, help="behavior_prompting checkout")
    ap.add_argument("--scenes", type=Path, required=True, help="output directory of scenes.py")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--image-size", type=int, choices=(128, 256), default=128)
    args = ap.parse_args()
    SIZE = args.image_size
    args.output.mkdir(parents=True, exist_ok=False)

    import yaml
    base = args.source / "behavior_prompting/train_network"
    libero = args.source / "deps/LIBERO/libero/libero"
    (args.output / "libero_config").mkdir()
    (args.output / "libero_config/config.yaml").write_text(yaml.safe_dump({
        "benchmark_root": str(libero), "assets": str(libero / "assets"),
        "bddl_files": str(base / "env/libero/bddl_files"),
        "init_states": str(base / "env/libero/init_files"), "datasets": str(args.output)}))
    os.environ["LIBERO_CONFIG_PATH"] = str(args.output / "libero_config")
    from behavior_prompting.train_network.env.libero.env_wrapper import ControlEnvForRunner
    from behavior_prompting.train_network.utils.libero_util import discover_and_register_benchmarks
    discover_and_register_benchmarks()

    manifest = json.loads((args.scenes / "manifest.json").read_text())
    records = []
    for scene in manifest["scenes"]:
        print("SCENE", scene["index"], flush=True)
        text = (args.scenes / scene["bddl"]).read_bytes()
        assert hashlib.sha256(text).hexdigest() == scene["bddl_sha256"], "task file changed since generation"
        folder = args.output / f"scene-{scene['index']:04d}"
        folder.mkdir()
        env = None
        try:
            # A scene LIBERO cannot build or place is a rejected sample, kept in the record.
            try:
                env = ControlEnvForRunner(bddl_file_name=str(args.scenes / scene["bddl"]),
                                          camera_heights=SIZE, camera_widths=SIZE, has_offscreen_renderer=True,
                                          has_renderer=False, use_camera_obs=True, control_delta=False,
                                          render_gpu_device_id=0)
                env.env.hard_reset = False
                env.seed(scene["seed"])
                env.env.reset()
            except Exception as error:  # noqa: BLE001
                record = dict(index=scene["index"], seed=scene["seed"], admitted=False,
                              errors=[f"build_or_reset_failed:{type(error).__name__}:{error}"])
            else:
                record = admit(env, scene, folder)
        finally:
            if env is not None:
                env.close()
        (folder / "admission.json").write_text(json.dumps(record, indent=2) + "\n")
        records.append(record)

    admitted = [r["index"] for r in records if r["admitted"]]
    reasons: dict[str, int] = {}
    for r in records:
        for e in r["errors"]:
            reasons[e.split(":")[0]] = reasons.get(e.split(":")[0], 0) + 1
    summary = dict(policy_loaded=False, scenes=len(records), admitted=admitted, rejection_reasons=reasons)
    (args.output / "result.json").write_text(json.dumps(dict(summary, records=records), indent=2) + "\n")
    print("ADMISSION_RESULT", json.dumps(dict(summary, admitted=len(admitted))), flush=True)


if __name__ == "__main__":
    main()
