"""Run one fixed development support and independent frame-zero query on the same arm/case."""

import hashlib
import json
from pathlib import Path
import sys
import traceback

from qualify import CASES, SEED, STATE, SUITE, TASK, save

import numpy as np
from arm_env import ArmEnv, CAMERAS
from embodiment import ACTIVE_PROBE_BURSTS
from libero.libero.envs.nonphysical_perturbations import _entity_body_ids, _transform_image_geometry
from mujoco.egl import GLContext
from OpenGL import GL

FIELDS = ("geom_rgba", "light_ambient", "light_diffuse", "cam_pos", "cam_quat", "cam_fovy")
VIEWS = tuple(camera + "_image" for camera in CAMERAS)


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def difference(a, b):
    return float(np.mean(np.abs(np.asarray(a, dtype=np.float32) - b)))


def snapshot(wrapper, first):
    scene = wrapper._env.env
    model = scene.sim.model
    arrays = {field: np.array(getattr(model, field), copy=True) for field in FIELDS}
    arrays["camera_ids"] = np.array([model.camera_name2id(camera) for camera in CAMERAS])
    targets = {i for name in CASES["S13"][2] for i in _entity_body_ids(scene, name)}
    arrays["target_mask"] = np.isin(model.geom_bodyid, list(targets))
    arrays.update({key: first["observation"][key].copy() for key in VIEWS})
    return arrays


def operator(case, current, base, raw, report):
    checks = {}
    metric = report["operator_measurements"] = {}
    l1 = metric["base_to_case_rgb_l1"] = {k: difference(current[k], base[k]) for k in VIEWS}
    checks["visible_change"] = any(x > 0 for x in l1.values())
    cams = base["camera_ids"]
    checks["camera_indices"] = np.array_equal(current["camera_ids"], cams)
    if case == "S13":
        mask = base["target_mask"]
        rgb = current["geom_rgba"][:, :3]
        metric["recolored_target_geoms"] = int(np.any(rgb[mask] != base["geom_rgba"][mask, :3], axis=1).sum())
        checks["target_recolor"] = (np.array_equal(mask, current["target_mask"])
                                    and metric["recolored_target_geoms"] > 0
                                    and np.allclose(rgb[mask], [0.18, 0.72, 0.38])
                                    and np.array_equal(rgb[~mask], base["geom_rgba"][~mask, :3]))
    if case == "S14":
        old = base["geom_rgba"]
        visible = old[:, 3] > 0
        rgb = old[visible, :3]
        tint = (rgb @ [0.2126, 0.7152, 0.0722])[:, None] * [0.72, 0.86, 1.18]
        tint[:, 2] = np.minimum(1, tint[:, 2])
        expected = np.clip(0.52 * rgb + 0.48 * tint, 0, 1)
        checks["palette"] = (np.allclose(current["geom_rgba"][visible, :3], expected, atol=1e-6)
                             and not np.array_equal(current["geom_rgba"], old))
        for field in ("light_diffuse", "light_ambient"):
            metric[field] = current[field].tolist()
            checks[field] = np.allclose(current[field], base[field] * 0.64)
        checks["lights_changed"] = not np.array_equal(current["light_diffuse"], base["light_diffuse"])
    if case == "S15":
        delta = current["cam_pos"][cams] - base["cam_pos"][cams]
        metric["camera_xyz_offset"] = delta.tolist()
        checks["extrinsics"] = (np.allclose(delta, [0.10, -0.10, 0.07])
                                and np.array_equal(current["cam_quat"], base["cam_quat"]))
        checks["both_views_changed"] = all(x > 0 for x in l1.values())
    if case in ("S16", "S19"):
        rng = np.random.default_rng(SEED)
        if case == "S16":
            fov = current["cam_fovy"][cams]
            metric["fovy_deg"] = fov.tolist()
            checks["fov"] = np.allclose(fov, np.clip(base["cam_fovy"][cams] * 1.35, 20, 100))
        # Native begin_rollout reseeds before producing frame zero. Read the same
        # cached, pre-transform RGB in native observable order without advancing it.
        for key in raw:
            if key not in VIEWS:
                continue
            image = raw[key]
            if case == "S19":
                noise = rng.normal(0, 12, image.shape)
                expected = np.clip(image.astype(np.float32) * 0.82 + noise, 0, 255).astype(np.uint8)
                checks["clean_base_" + key] = np.array_equal(image, base[key])
            else:
                expected = _transform_image_geometry(image, np.array([0.10, -0.10]), 0.04, 0.04)
            metric["image_space_l1_" + key] = difference(current[key], image)
            checks["image_operator_" + key] = (np.array_equal(expected, current[key])
                                               and metric["image_space_l1_" + key] > 0)
        checks["both_raw_views"] = all(k in raw for k in VIEWS)
    return checks


def reset_record(first):
    return dict(model=first["model_identity"], reset=first["environment_reproducibility"],
                instruction=first["language_instruction"],
                tool={k: np.asarray(first["tool_observation"][k]).tolist()
                      for k in ("tcp_xyz", "tcp_matrix", "jaw_opening")},
                public_sha256={k: digest(v) for k, v in first["observation"].items()})


def run(path, arm, case, rec):
    context = GLContext(16, 16)
    try:
        context.make_current()
        rec["renderer"] = GL.glGetString(GL.GL_RENDERER).decode()
        if not any(s in rec["renderer"].lower() for s in ("llvmpipe", "softpipe", "swrast")):
            raise RuntimeError(f"Expected software EGL renderer, got {rec['renderer']}")
    finally:
        context.free()
    checks = rec["checks"] = {}
    with ArmEnv(arm, SUITE, TASK, case_id=case) as env:
        first = env.reset(STATE, SEED)
        support = rec["support"] = reset_record(first)
        ident, reset = support["model"], support["reset"]
        plan = reset["perturbation_plan"]
        checks["released_row_13"] = reset["released_row_fingerprint"] == (
            "2c47daced5bfc02f1f7451ee2c2e49931834b053bafee7c3999683c8170e5c30")
        sha, op, targets = CASES[case]
        checks["bddl_and_plan"] = (reset["bddl_sha256"] == sha and plan["case_id"] == case
                                   and plan["operator"] == op and plan["targets"] == targets
                                   and plan["seed"] == SEED and plan["schedule"] == ("base" if case == "BASE" else "static"))
        joint_names = {"IIWA": [f"joint_{i}" for i in range(1, 8)],
                       "Kinova3": [f"Actuator{i}" for i in range(1, 8)],
                       "UR5e": ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                                "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]}[arm]
        checks["compiled_arm"] = (ident["arm"] == arm and ident["model_class"] == "Mounted" + arm
                                  and ident["joint_names"] == ["robot0_" + j for j in joint_names]
                                  and ident["joint_count"] == len(joint_names))
        snap = snapshot(env, first)
        np.savez_compressed(path / "frame0.npz", **snap)
        if case != "BASE":
            base_dir = path.parent / (arm + "-BASE")
            base_record = json.loads((base_dir / "cell.json").read_text())["support"]
            for k in ("released_row_fingerprint", "scene_fingerprint", "post_settle_scene_fingerprint",
                      "initial_state_fingerprint"):
                checks["paired_" + k] = reset[k] == base_record["reset"][k]
            checks["paired_tcp_jaw"] = support["tool"] == base_record["tool"]
            checks["paired_prompt"] = support["instruction"] == base_record["instruction"]
            with np.load(base_dir / "frame0.npz") as base:
                raw_cache = env._env.env._get_observations()
                checks.update(operator(case, snap, base, raw_cache, rec))
        elif arm != "UR5e":
            reference = path.parent / "UR5e-BASE/cell.json"
            if reference.exists():
                ref = json.loads(reference.read_text()).get("support", {}).get("reset")
                if ref:
                    for k in ("released_row_fingerprint", "scene_fingerprint"):
                        checks["cross_arm_" + k] = reset[k] == ref[k]

        rec["stage"] = "same_case_support_controls"
        rec["checks"] = {k: bool(v) for k, v in checks.items()}
        save(path / "cell.json", rec)
        probe = rec["controls"] = []
        frames = {key: [snap[key]] for key in VIEWS}
        boundaries = [support["tool"]]
        for burst in ACTIVE_PROBE_BURSTS:
            for action in burst:
                step = env.probe(action)
                tool = step["tool_observation"]
                probe.append(dict(index=step["worker_step_index"], issued=step["issued_action"].tolist(),
                                  tcp_xyz=tool["tcp_xyz"].tolist(), jaw_opening=tool["jaw_opening"],
                                  reward=step["reward"], success=step["success"], terminated=step["terminated"],
                                  truncated=step["truncated"]))
                if step["terminated"] or step["truncated"]:
                    break
            for key in frames:
                frames[key].append(step["observation"][key].copy())
            boundaries.append(probe[-1])
            if step["terminated"] or step["truncated"]:
                break
        np.savez_compressed(path / "support.npz", **{k: np.stack(v) for k, v in frames.items()})
        xyz = np.array([v["tcp_xyz"] for v in boundaries])
        jaw = np.array([v["jaw_opening"] for v in boundaries])
        distance = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
        jaw_move = float(np.max(np.abs(np.diff(jaw))))
        image_l1 = {k: float(np.mean([difference(a, b) for a, b in zip(v, v[1:])])) for k, v in frames.items()}
        rec["support_motion"] = dict(tcp_path_m=distance, max_jaw_delta_m=jaw_move, rgb_l1=image_l1)
        checks.update(complete_probe=len(probe) == 15, tool_motion=distance > 0.015, jaw_motion=jaw_move > 0.010,
                      image_motion=image_l1[VIEWS[0]] > 0.5 and image_l1[VIEWS[1]] > 0)
        rec["stage"] = "independent_query_reset"
        rec["checks"] = {k: bool(v) for k, v in checks.items()}
        save(path / "cell.json", rec)
        query = rec["query"] = reset_record(env.reset(STATE, SEED))
        checks["same_case_support_query"] = support == query
    rec["checks"] = {k: bool(v) for k, v in checks.items()}
    rec["status"] = "passed" if all(checks.values()) else "failed"
    rec["error"] = ", ".join(k for k, passed in checks.items() if not passed)


if __name__ == "__main__":
    path, arm, case = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    record = dict(arm=arm, case=case, status="failed", stage="renderer_or_reset")
    try:
        run(path, arm, case, record)
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        record["checks"] = {k: bool(v) for k, v in record.get("checks", {}).items()}
        save(path / "cell.json", record)
