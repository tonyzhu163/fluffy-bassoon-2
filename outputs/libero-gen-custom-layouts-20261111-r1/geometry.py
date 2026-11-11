"""Allocated simulator-only layout admission; never loads a policy."""
import argparse
import hashlib
import json
import os
from pathlib import Path

RUN = "libero-gen-custom-layouts-20261111-r1"
SPLIT = "libero_spatial_selected_combinations_inverse_view"
TASKS = ["pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
         "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_2"]
BOWLS = ["akita_black_bowl_1", "akita_black_bowl_2"]


def sha(value):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def templates():
    """Fixed locations in table/world metres; no outcome-based rejection search."""
    import numpy as np
    near = [(0.005, 0.255), (-0.185, 0.255), (0.005, -0.025), (-0.21, 0.04)]
    plate = [(0.17, 0.255), (-0.28, 0.375), (0.17, -0.025), (-0.35, 0.105)]
    records = []
    for quadrant in range(4):
        for repeat in range(4):
            seed = 3100 + quadrant * 4 + repeat
            rng = np.random.default_rng(seed)
            center_xy = np.array([-0.09, 0.115]) + rng.uniform(-0.004, 0.004, 2)
            near_xy = np.array(near[quadrant]) + rng.uniform(-0.004, 0.004, 2)
            plate_xy = np.array(plate[quadrant]) + rng.uniform(-0.004, 0.004, 2)
            ids = BOWLS if repeat % 2 == 0 else BOWLS[::-1]
            positions = {ids[0]: center_xy.tolist(), ids[1]: near_xy.tolist(),
                         "plate_1": plate_xy.tolist(), "cookies_1": [0.17, 0.115],
                         "glazed_rim_porcelain_ramekin_1": [-0.34, 0.24]}
            records.append({"index": len(records), "seed": seed, "quadrant": quadrant,
                            "repeat": repeat, "candidate_object_ids": list(ids), "positions_xy": positions})
    return records


def object_position(env, name):
    return env.sim.data.body_xpos[env.env.obj_body_id[name]].copy()


def set_object_position(env, name, xy, z=None):
    import numpy as np
    joint = env.env.objects_dict[name].joints[0]
    qpos = np.asarray(env.sim.data.get_joint_qpos(joint)).copy()
    qpos[:2] = xy
    if z is not None:
        qpos[2] = z
    env.sim.data.set_joint_qpos(joint, qpos)
    env.sim.data.set_joint_qvel(joint, np.zeros(6))


def project(env, xyz):
    import numpy as np
    sim = env.sim
    cid = sim.model.camera_name2id("agentview")
    rot = sim.data.cam_xmat[cid].reshape(3, 3)
    local = rot.T @ (np.asarray(xyz) - sim.data.cam_xpos[cid])
    depth = -float(local[2])
    assert depth > 0
    f = 64 / np.tan(np.deg2rad(sim.model.cam_fovy[cid]) / 2)
    return [float(64 + f * local[0] / depth), float(64 - f * local[1] / depth)]


def hold_action(env):
    import numpy as np
    from robosuite.utils.transform_utils import quat2axisangle, quat2mat, mat2quat
    obs = env.env._get_observations()
    # Exact convention used by BPP libero_image_runner.py:637-648.
    rotation = quat2mat(obs["robot0_eef_quat"]) @ np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
    return np.concatenate((obs["robot0_eef_pos"], quat2axisangle(mat2quat(rotation)), [-1.0]))


def reset_restore(env, state):
    """Reset Python episode/controller state before restoring MuJoCo state."""
    env.seed(7)
    env.env.reset()  # Fail fast; upstream wrapper reset swallows exceptions.
    return env.regenerate_obs_from_state(state)


def restored_native(sidecar, state):
    sidecar.set_state(state)
    sidecar.sim.forward()
    sidecar._post_process()
    return bool(sidecar.check_success())


def parity(primary, sidecars):
    """Compare independent restored simulators against official primary predicates."""
    import numpy as np
    state = primary.get_sim_state().copy()
    original_goals = primary.env.parsed_problem["goal_state"]
    values = []
    try:
        for sidecar in sidecars:
            assert primary.sim.model.nq == sidecar.sim.model.nq
            assert primary.sim.model.nv == sidecar.sim.model.nv
            # Fixed bodies (fixtures/cameras) are model data, not flattened state.
            assert np.array_equal(primary.sim.model.body_pos, sidecar.sim.model.body_pos)
            assert np.array_equal(primary.sim.model.body_quat, sidecar.sim.model.body_quat)
            goals = sidecar.env.parsed_problem["goal_state"]
            direct = all(bool(primary.env._eval_predicate(goal)) for goal in goals)
            restored = restored_native(sidecar, state)
            assert direct == restored, "Native sidecar predicate mismatch"
            assert np.array_equal(state, primary.get_sim_state()), "Sidecar changed primary physics"
            values.append(int(restored))
    finally:
        primary.env.parsed_problem["goal_state"] = original_goals
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import yaml
    base = args.source / "behavior_prompting/train_network"
    cfg = args.output / "libero_config"
    cfg.mkdir()
    (cfg / "config.yaml").write_text(yaml.safe_dump({
        "benchmark_root": str(args.source / "deps/LIBERO/libero/libero"),
        "assets": str(args.source / "deps/LIBERO/libero/libero/assets"),
        "bddl_files": str(base / "env/libero/bddl_files"),
        "init_states": str(base / "env/libero/init_files"), "datasets": str(args.output)}))
    os.environ["LIBERO_CONFIG_PATH"] = str(cfg)
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from behavior_prompting.train_network.env.libero.env_wrapper import ControlEnvForRunner
    from behavior_prompting.train_network.utils.libero_util import discover_and_register_benchmarks
    discover_and_register_benchmarks()
    init_file = base / "env/libero/init_files" / SPLIT / (TASKS[0] + ".pruned_init")
    bank = torch.load(init_file, weights_only=False)
    envs = []
    results = []
    images = []
    failed = []
    try:
        for index in range(3):
            task = TASKS[0 if index == 0 else index - 1]
            e = ControlEnvForRunner(bddl_file_name=str(base / "env/libero/bddl_files" / SPLIT / (task + ".bddl")),
                                    camera_heights=128, camera_widths=128,
                                    has_offscreen_renderer=index == 0, has_renderer=False,
                                    use_camera_obs=index == 0, control_delta=False, render_gpu_device_id=0)
            e.env.hard_reset = False
            e.seed(7)
            e.env.reset()
            envs.append(e)
        primary, *sidecars = envs
        for spec in templates():
            print("GEOMETRY", spec["index"], flush=True)
            reset_restore(primary, bank[spec["index"]])
            for name, xy in spec["positions_xy"].items():
                set_object_position(primary, name, xy)
            primary.sim.forward()
            # Exactly 20 fixed controls settle relocated objects; no policy or reward-conditioned actions.
            action = hold_action(primary)
            before_hold = primary.env._get_observations()
            before_eef_pos = before_hold["robot0_eef_pos"].copy()
            before_eef_quat = before_hold["robot0_eef_quat"].copy()
            for _ in range(20):
                primary.step(action)
            state = primary.get_sim_state().copy()
            obs = primary.prepare_observation(primary.regenerate_obs_from_state(state))
            folder = args.output / f"layout-{spec['index']:02d}"
            folder.mkdir()
            np.save(folder / "state.npy", state)
            np.savez_compressed(folder / "initial-observation.npz", **obs)
            image = Image.fromarray(obs["agentview_image"])
            image.save(folder / "agentview.png")
            Image.fromarray(obs["robot0_eye_in_hand_image"]).save(folder / "wrist.png")
            pos = {name: object_position(primary, name).tolist() for name in spec["positions_xy"]}
            pixels = [project(primary, pos[name]) for name in spec["candidate_object_ids"]]
            left = int(pixels[1][0] < pixels[0][0])
            upper = int(pixels[1][1] < pixels[0][1])
            stratum = "agreement" if left == upper else "disagreement"
            margins = [abs(pixels[0][i] - pixels[1][i]) for i in [0, 1]]
            errors = []
            eef_translation = float(np.linalg.norm(obs["robot0_eef_pos"] - before_eef_pos))
            eef_rotation_degrees = float(np.rad2deg(2 * np.arccos(np.clip(abs(np.dot(
                before_eef_quat, obs["robot0_eef_quat"])), 0, 1))))
            if eef_translation > 0.02 or eef_rotation_degrees > 10:
                errors.append("stationary_hold_eef_drift")
            if min(margins) < 6:
                errors.append("projected_axis_margin_below_6px")
            if any(not 8 <= value <= 120 for point in pixels for value in point):
                errors.append("projected_center_outside_8px_border")
            contacts = []
            names = list(spec["positions_xy"]) + ["wooden_cabinet_1", "flat_stove_1"]
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    if primary.env.check_contact(primary.env.get_object(a), primary.env.get_object(b)):
                        contacts.append([a, b])
            if contacts:
                errors.append("initial_object_or_fixture_contact")
            drift = max(float(np.linalg.norm(np.array(pos[name][:2]) - xy)) for name, xy in spec["positions_xy"].items())
            if drift > 0.005:
                errors.append("settling_xy_drift_above_5mm")
            native = parity(primary, sidecars)
            if native != [0, 0]:
                errors.append("initial_native_goal_already_true")
            # Positive predicate fixtures are diagnostics, never motor successes.
            positive = []
            for bowl in BOWLS:
                reset_restore(primary, state)
                plate_pos = object_position(primary, "plate_1")
                set_object_position(primary, bowl, plate_pos[:2], plate_pos[2] + 0.08)
                primary.sim.forward()
                for _ in range(40):
                    primary.step(action)
                values = parity(primary, sidecars)
                positive.append(values)
                if values != [int(bowl == b) for b in BOWLS]:
                    errors.append("positive_native_fixture_not_distinct")
            reset_restore(primary, state)
            record = {**spec, "stratum": stratum, "rule_choices": {"leftmost": left, "uppermost": upper},
                      "actual_positions": pos, "projected_candidate_centers": pixels,
                      "axis_margins_pixels": margins, "settling_max_xy_drift": drift,
                      "contacts": contacts, "initial_native": native, "positive_native_fixtures": positive,
                      "hold_action": action.tolist(), "hold_eef_translation_metres": eef_translation,
                      "hold_eef_rotation_degrees": eef_rotation_degrees,
                      "state_sha256": sha(state), "observation_hashes": {k: sha(v) for k, v in obs.items()},
                      "sidecar_parity": True, "errors": errors, "admitted_geometry": not errors}
            (folder / "geometry.json").write_text(json.dumps(record, indent=2) + "\n")
            results.append(record)
            if errors:
                failed.append(spec["index"])
            tile = image.resize((256, 256))
            draw = ImageDraw.Draw(tile)
            draw.text((4, 4), f"{spec['index']} {stratum} L{left} U{upper}", fill="red")
            for candidate, (u, v) in enumerate(pixels):
                color = "cyan" if candidate == 0 else "yellow"
                draw.ellipse((2*u-4, 2*v-4, 2*u+4, 2*v+4), outline=color, width=2)
                draw.text((2*u+5, 2*v-5), str(candidate), fill=color)
            tile.save(folder / "projection-diagnostic.png")
            images.append(tile)
        canvas = Image.new("RGB", (1024, 1024), "white")
        for i, image in enumerate(images):
            canvas.paste(image, ((i % 4) * 256, (i // 4) * 256))
        canvas.save(args.output / "geometry-contact-sheet.png")
        balances = {}
        for stratum in ["agreement", "disagreement"]:
            selected = [r for r in results if r["stratum"] == stratum]
            balances[stratum] = {"count": len(selected)}
            for rule in ["leftmost", "uppermost"]:
                choices = [r["rule_choices"][rule] for r in selected]
                physical = [r["candidate_object_ids"][r["rule_choices"][rule]] for r in selected]
                balances[stratum][rule] = {"candidate0": choices.count(0), "candidate1": choices.count(1),
                                           "bowl1": physical.count(BOWLS[0]), "bowl2": physical.count(BOWLS[1])}
        balanced = all(v["count"] == 8 and all(v[r] == {"candidate0": 4, "candidate1": 4, "bowl1": 4, "bowl2": 4}
                       for r in ["leftmost", "uppermost"]) for v in balances.values())
        report = {"complete": True, "policy_loaded": False, "geometry_passed": not failed and balanced,
                  "failed_indices": failed, "balance_passed": balanced, "balances": balances, "layouts": results}
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print("GEOMETRY_RESULT", json.dumps({k: v for k, v in report.items() if k != "layouts"}), flush=True)
    finally:
        for env in reversed(envs):
            env.close()


if __name__ == "__main__":
    main()
