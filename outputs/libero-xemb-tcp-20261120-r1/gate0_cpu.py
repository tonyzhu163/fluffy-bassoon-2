"""CPU Gate 0 audit for physical MuJoCo finger-TCP offsets on 8 LIBERO dev tasks."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from fluffy_bassoon.xemb.embodiment import downsample_agentview, jaw_width_from_qpos
from fluffy_bassoon.xemb.tcp import FINGER_COLLISION_SUFFIXES, apply_mujoco_tcp_offset


HERE = Path(__file__).resolve().parent
ROOT = Path("/mowenjie/fluffy-bassoon-2")
SPLITS_PATH = HERE.parent / "libero-meta-learning-20261117-r1" / "SPLITS.json"
SOURCE = ROOT / "outputs/libero-retry-headroom-20261119-r1/paired01/episodes"
RELEASE = ROOT / "sources/libero-pro2-e5500c69"
PROPRIO_FIELDS = (
    "robot0_joint_pos",
    "robot0_joint_vel",
    "robot0_eef_pos",
    "robot0_eef_quat_xyzw",
    "robot0_gripper_qpos",
)
RADII_M = (0.025, 0.035, 0.045)
DIRECTIONS = (
    ("px", 1.0, 0.0),
    ("mx", -1.0, 0.0),
    ("py", 0.0, 1.0),
    ("my", 0.0, -1.0),
)


def digest(arr) -> str:
    a = np.ascontiguousarray(arr)
    return hashlib.sha256(repr((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()


class ContactWatcher:
    """Record first physical finger contact during MuJoCo integration substeps."""

    def __init__(self, world):
        self.sim = world.sim
        model = self.sim.model
        names = [model.geom_id2name(i) or "" for i in range(model.ngeom)]
        ids = []
        for suffix in FINGER_COLLISION_SUFFIXES:
            matches = [i for i, n in enumerate(names) if n.endswith("_" + suffix)]
            if len(matches) != 1:
                raise ValueError(f"Expected 1 geom for _{suffix}, got {matches}")
            ids.append(matches[0])
        self.fingers = np.zeros(model.ngeom, dtype=bool)
        self.fingers[ids] = True
        self.names = names
        self.control = 0
        self.integration = 0
        self.first_contact = None
        self.native_step = self.sim.step
        self.sample()
        self.sim.step = self.step

    def sample(self):
        contact = self.sim.data._data.contact
        pairs = contact.geom
        active = np.flatnonzero(self.fingers[pairs].any(axis=1))
        for idx in active:
            if contact[int(idx)].efc_address >= 0:
                if self.first_contact is None:
                    self.first_contact = dict(
                        control=self.control,
                        substep=self.integration,
                        geoms=[self.names[int(g)] for g in pairs[idx]],
                    )
                break

    def step(self):
        self.native_step()
        self.integration += 1
        self.sample()


def run_task_family(job):
    """Run nominal + all (radius, direction) replays for one dev task cleanly inside one process."""
    suite, task_id = job
    from libero.libero import benchmark
    from libero.libero.envs.env_wrapper import ControlEnv
    from libero_remote.environment.libero import LiberoEnvironmentAdapter
    from libero_remote.environment.task_manifest import TaskManifest
    from OpenGL import GL

    splits = json.loads(SPLITS_PATH.read_text())
    manifest = TaskManifest.load(str(RELEASE / "assets/task_manifest.json"))
    task_label = f"{suite}-{task_id}"
    folder = SOURCE / f"{task_label}-r0-BASE" / "default"
    record = json.loads((folder / "result.json").read_text())
    case = record["case"]
    ident = record["identity"]
    horizon = int(ident["horizon"])
    with np.load(folder / "public.npz", allow_pickle=False) as src:
        issued = src["issued"][: int(record["controls"])].copy()

    bddl = manifest.resolve_bddl(suite, task_id, "BASE")
    bank = benchmark.get_benchmark_dict()[suite]().get_task_init_states(task_id)

    variants = [("nom", 0.0, 0.0, 0.0)]
    for rad in RADII_M:
        for dname, ux, uy in DIRECTIONS:
            variants.append((f"r{int(round(rad * 1000))}_{dname}", rad, rad * ux, rad * uy))

    # First run nominal to determine nominal first contact control.
    results = {}
    nom_precontact_last = None
    nom_first_contact = None
    nom_proprio_seq = None
    nom_camera_hashes = None

    for vname, rad, dx, dy in variants:
        env = LiberoEnvironmentAdapter()
        env._init_states = bank
        env._max_steps = horizon
        env._env = ControlEnv(str(bddl), use_camera_obs=False, camera_names=[], has_offscreen_renderer=True)
        try:
            first = env.reset(int(case["init_state_id"]), int(case["seed"]))
            world = env._env.env
            renderer = GL.glGetString(GL.GL_RENDERER).decode()
            if "llvmpipe" not in renderer.lower():
                raise RuntimeError(f"Expected Mesa llvmpipe on CPU, got {renderer!r}")
            shift_meta = apply_mujoco_tcp_offset(world, dx, dy, 0.0)
            watcher = ContactWatcher(world)
            if watcher.first_contact is not None:
                raise RuntimeError(f"Contact present right at reset on {task_label} {vname}: {watcher.first_contact}")

            sensors = {}
            for cam in ("agentview", "robot0_eye_in_hand"):
                cam_sens, cam_names = world._create_camera_sensors(cam, 256, 256, False, None)
                sensors[cam_names[0]] = cam_sens[0]

            raw0 = first["observation"]
            cam0 = {k: fn({}) for k, fn in sensors.items()}
            cam0_hashes = {k: digest(v) for k, v in cam0.items()}
            proprio_steps = [{k: raw0[k].copy() for k in PROPRIO_FIELDS}]
            jaws = [float(jaw_width_from_qpos(raw0["robot0_gripper_qpos"]))]
            rewards = []

            # Determine post-contact window frames to render once nominal first contact is known.
            if nom_first_contact is not None:
                check_controls = {
                    max(0, nom_precontact_last // 2),
                    nom_precontact_last,
                }
                window_controls = [
                    min(len(issued), nom_first_contact + off)
                    for off in (0, 5, 10, 15, 20, 30, 40, 50, 60)
                ] + [len(issued)]
            else:
                check_controls = set()
                window_controls = []

            precontact_cam_hashes = {"0": cam0_hashes}
            window_frames = []
            success = False
            stopped = len(issued)

            for idx, act in enumerate(issued, 1):
                watcher.control = idx
                step = env.step(act.copy())
                raw = step["observation"]
                proprio_steps.append({k: raw[k].copy() for k in PROPRIO_FIELDS})
                jaws.append(float(jaw_width_from_qpos(raw["robot0_gripper_qpos"])))
                rewards.append(float(step["reward"]))
                success = bool(step["success"])
                if idx in check_controls:
                    cams = {k: fn({}) for k, fn in sensors.items()}
                    precontact_cam_hashes[str(idx)] = {k: digest(v) for k, v in cams.items()}
                if idx in window_controls:
                    cams = {k: fn({}) for k, fn in sensors.items()}
                    window_frames.append(
                        np.stack(
                            [
                                downsample_agentview(cams["agentview_image"], 64),
                                downsample_agentview(cams["robot0_eye_in_hand_image"], 64),
                            ]
                        )
                    )
                if step["terminated"] or step["truncated"]:
                    stopped = idx
                    break

            if vname == "nom":
                if watcher.first_contact is None:
                    raise RuntimeError(f"Nominal replay never made finger contact on {task_label}")
                nom_first_contact = int(watcher.first_contact["control"])
                nom_precontact_last = max(0, nom_first_contact - 1)
                # Re-render mid and last precontact on a clean replay? Wait: we already checked precontact proprio;
                # let's record nom's mid and last precontact camera hashes cleanly in a fast 1-prefix pass if needed!
                nom_proprio_seq = proprio_steps[: nom_precontact_last + 1]
            else:
                # Re-verify precontact camera hashes when nom_precontact_last > 0.
                pass

            # Pad window_frames if episode terminated early.
            if window_controls and not window_frames:
                window_frames.append(
                    np.stack(
                        [
                            downsample_agentview(cam0["agentview_image"], 64),
                            downsample_agentview(cam0["robot0_eye_in_hand_image"], 64),
                        ]
                    )
                )
            while len(window_frames) < len(window_controls) and window_frames:
                window_frames.append(window_frames[-1])

            results[vname] = dict(
                variant=vname,
                radius_m=rad,
                dx_m=dx,
                dy_m=dy,
                physics=first["environment_reproducibility"]["initial_state_fingerprint"],
                shift_meta=shift_meta,
                first_contact=watcher.first_contact,
                controls=stopped,
                success=success,
                native_return=float(sum(rewards)),
                cam0_hashes=cam0_hashes,
                precontact_cam_hashes=precontact_cam_hashes,
                precontact_proprio_equal=(
                    all(
                        np.array_equal(proprio_steps[t][f], nom_proprio_seq[t][f])
                        for t in range(min(len(proprio_steps), len(nom_proprio_seq)))
                        for f in PROPRIO_FIELDS
                    )
                    if nom_proprio_seq is not None
                    else True
                ),
                jaws=np.asarray(jaws, dtype=np.float32),
                joints=np.stack([p["robot0_joint_pos"] for p in proprio_steps]),
                window_frames=np.stack(window_frames) if window_frames else None,
            )
        finally:
            env.close()

    # Check mid & last precontact camera equality cleanly against nom using a quick prefix replay of nom!
    nom_row = results["nom"]
    nom_mid = max(0, nom_precontact_last // 2)
    env = LiberoEnvironmentAdapter()
    env._init_states = bank
    env._max_steps = horizon
    env._env = ControlEnv(str(bddl), use_camera_obs=False, camera_names=[], has_offscreen_renderer=True)
    try:
        env.reset(int(case["init_state_id"]), int(case["seed"]))
        world = env._env.env
        sensors = {}
        for cam in ("agentview", "robot0_eye_in_hand"):
            cam_sens, cam_names = world._create_camera_sensors(cam, 256, 256, False, None)
            sensors[cam_names[0]] = cam_sens[0]
        for idx in range(1, nom_precontact_last + 1):
            env.step(issued[idx - 1].copy())
            if idx in (nom_mid, nom_precontact_last):
                cams = {k: fn({}) for k, fn in sensors.items()}
                nom_row["precontact_cam_hashes"][str(idx)] = {k: digest(v) for k, v in cams.items()}
    finally:
        env.close()

    # Summarize per-radius metrics and post-contact directional separability.
    summary_rows = []
    separability = {}
    for vname, row in results.items():
        frame0_eq = row["cam0_hashes"] == nom_row["cam0_hashes"]
        pre_cam_eq = all(
            row["precontact_cam_hashes"].get(k) == nom_row["precontact_cam_hashes"].get(k)
            for k in nom_row["precontact_cam_hashes"]
        )
        summary_rows.append(
            dict(
                task=task_label,
                variant=vname,
                radius_m=row["radius_m"],
                dx_m=row["dx_m"],
                dy_m=row["dy_m"],
                success=row["success"],
                controls=row["controls"],
                first_contact=row["first_contact"],
                frame0_rgb_equal=frame0_eq,
                precontact_rgb_equal=pre_cam_eq,
                precontact_proprio_equal=row["precontact_proprio_equal"],
                max_shift_err_m=row["shift_meta"]["max_shift_err_m"],
                pad_gap_err_m=row["shift_meta"]["pad_gap_err_m"],
                reset_ncon_unchanged=row["shift_meta"]["before_ncon"] == row["shift_meta"]["after_ncon"],
            )
        )

    for rad in RADII_M:
        rtag = f"r{int(round(rad * 1000))}"
        opp_metrics = {}
        for a, b in (("px", "mx"), ("py", "my")):
            ra, rb = results[f"{rtag}_{a}"], results[f"{rtag}_{b}"]
            fa, fb = ra["window_frames"], rb["window_frames"]
            rgb_l1 = float(np.mean(np.abs(fa.astype(np.float32) - fb.astype(np.float32))))
            n_min = min(len(ra["jaws"]), len(rb["jaws"]))
            jaw_l1_mm = float(np.mean(np.abs(ra["jaws"][nom_first_contact:n_min] - rb["jaws"][nom_first_contact:n_min])) * 1000.0) if n_min > nom_first_contact else 0.0
            joint_l1 = float(np.mean(np.abs(ra["joints"][nom_first_contact:n_min] - rb["joints"][nom_first_contact:n_min]))) if n_min > nom_first_contact else 0.0
            opp_metrics[f"{a}_vs_{b}"] = dict(rgb_l1=rgb_l1, jaw_l1_mm=jaw_l1_mm, joint_l1=joint_l1)
        separability[rtag] = opp_metrics

    return dict(task=task_label, nom_first_contact=nom_first_contact, rows=summary_rows, separability=separability)


def probe_robosuite_arms():
    """Informational probe of available robosuite arms and grippers for Phase 3 planning."""
    import robosuite.robots as rs_robots
    import robosuite.models.grippers as rs_grippers
    from libero.libero.envs.env_wrapper import ControlEnv
    from libero_remote.environment.task_manifest import TaskManifest

    manifest = TaskManifest.load(str(RELEASE / "assets/task_manifest.json"))
    bddl = str(manifest.resolve_bddl("libero_goal", 1, "BASE"))
    info = dict(
        robot_classes=sorted(getattr(rs_robots, "ROBOT_CLASS_MAPPING", {}).keys()),
        grippers=sorted(n for n in getattr(rs_grippers, "ALL_GRIPPERS", []) if n is not None),
        instantiation={},
    )
    for robot_name in ("Panda", "IIWA", "UR5e", "Sawyer"):
        try:
            env = ControlEnv(bddl, robots=robot_name, use_camera_obs=False, has_offscreen_renderer=False)
            obs = env.reset()
            info["instantiation"][robot_name] = dict(
                ok=True,
                action_dim=int(env.env.action_dim),
                obs_keys=sorted(obs.keys()),
            )
            env.close()
        except Exception as exc:
            info["instantiation"][robot_name] = dict(ok=False, error=f"{type(exc).__name__}: {exc}")
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    started = time.monotonic()
    splits = json.loads(SPLITS_PATH.read_text())
    jobs = [(suite, int(t)) for suite, tids in sorted(splits["development"].items()) for t in tids]
    args.output.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        task_results = list(pool.map(run_task_family, jobs))

    arms_probe = probe_robosuite_arms()
    (args.output / "robosuite-arms-probe.json").write_text(json.dumps(arms_probe, indent=2) + "\n")

    all_rows = [r for tr in task_results for r in tr["rows"]]
    nom_rows = [r for r in all_rows if r["variant"] == "nom"]
    nom_succ = sum(r["success"] for r in nom_rows)

    all_frame0_ok = all(r["frame0_rgb_equal"] for r in all_rows)
    all_pre_rgb_ok = all(r["precontact_rgb_equal"] for r in all_rows)
    all_pre_prop_ok = all(r["precontact_proprio_equal"] for r in all_rows)
    all_phys_ok = all(
        r["max_shift_err_m"] < 1e-6 and r["pad_gap_err_m"] < 1e-6 and r["reset_ncon_unchanged"]
        for r in all_rows
    )

    by_radius = {}
    candidate_radii = []
    for rad in RADII_M:
        rtag = f"r{int(round(rad * 1000))}"
        r_rows = [r for r in all_rows if r["radius_m"] == rad]
        succ = sum(r["success"] for r in r_rows)
        opp_rgb = [
            tr["separability"][rtag][pair]["rgb_l1"]
            for tr in task_results
            for pair in ("px_vs_mx", "py_vs_my")
        ]
        opp_jaw = [
            tr["separability"][rtag][pair]["jaw_l1_mm"]
            for tr in task_results
            for pair in ("px_vs_mx", "py_vs_my")
        ]
        distinguishable_tasks = sum(
            min(
                tr["separability"][rtag]["px_vs_mx"]["rgb_l1"],
                tr["separability"][rtag]["py_vs_my"]["rgb_l1"],
            )
            > 0.05
            for tr in task_results
        )
        ok = (
            nom_succ == len(nom_rows)
            and succ <= len(r_rows) // 4
            and all_frame0_ok
            and all_pre_rgb_ok
            and all_pre_prop_ok
            and all_phys_ok
            and distinguishable_tasks >= 6
        )
        by_radius[rtag] = dict(
            radius_m=rad,
            trials=len(r_rows),
            nominal_shifted_successes=succ,
            nominal_shifted_rate=succ / len(r_rows),
            mean_opp_rgb_l1=float(np.mean(opp_rgb)),
            min_opp_rgb_l1=float(np.min(opp_rgb)),
            mean_opp_jaw_l1_mm=float(np.mean(opp_jaw)),
            distinguishable_tasks=distinguishable_tasks,
            passes_gate0=ok,
        )
        if ok:
            candidate_radii.append(rad)

    selected_radius = 0.035 if 0.035 in candidate_radii else (candidate_radii[0] if candidate_radii else None)
    gate0_passed = selected_radius is not None

    summary = dict(
        gate0_passed=gate0_passed,
        selected_radius_m=selected_radius,
        seconds=round(time.monotonic() - started, 2),
        nominal_base_successes=f"{nom_succ}/{len(nom_rows)}",
        zero_leak_checks=dict(
            frame0_rgb_bitwise_equal=all_frame0_ok,
            precontact_rgb_bitwise_equal=all_pre_rgb_ok,
            precontact_proprio_bitwise_equal=all_pre_prop_ok,
            mujoco_shift_exact_and_collision_free=all_phys_ok,
        ),
        by_radius=by_radius,
        task_separability={tr["task"]: tr["separability"] for tr in task_results},
        rows=all_rows,
    )
    (args.output / "gate0-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                gate0_passed=gate0_passed,
                selected_radius_m=selected_radius,
                nominal_base=summary["nominal_base_successes"],
                zero_leak=summary["zero_leak_checks"],
                by_radius=by_radius,
            ),
            indent=2,
        ),
        flush=True,
    )
    if not gate0_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
