"""Fresh, explicitly approximate physical replays of the archived chunk means with orbital agentviews."""

from pathlib import Path

import numpy as np

from world_control import (GripSite, arguments, file_sha, motor_state, panel, qualify, reset, scored,
                           setup, tape, write_json)


def azimuth_name(angle):
    return f"azimuth{angle:+d}"


class OrbitalViews:
    """Rerender within robosuite's agentview sensor call, before the next physics substep."""

    def __init__(self, arm, first, angles):
        from robosuite import macros
        from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING

        self.sim = arm._env.sim
        self.robot = arm._env.robots[0]
        self.cam = self.sim.model.camera_name2id("agentview")
        self.render = self.sim.render
        self.direction = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
        self.angles = angles
        self.pivot = first["observation"]["robot0_eef_pos"].astype(np.float64)
        self.candidates = []

    def record(self, nominal, settings):
        """Move only derived camera extrinsics used by mjv_updateScene; restore before physics resumes."""
        import mujoco
        from embodiment import downsample_agentview

        model, data = self.sim.model._model, self.sim.data._data
        state_flag = mujoco.mjtState.mjSTATE_INTEGRATION
        before = np.empty(mujoco.mj_stateSize(model, state_flag))
        after = np.empty_like(before)
        mujoco.mj_getState(model, data, before, state_flag)
        pos = data.cam_xpos[self.cam].copy()
        mat = data.cam_xmat[self.cam].reshape(3, 3).copy()
        images, extrinsics = {}, {}
        try:
            for angle in self.angles:
                a = np.radians(angle)
                rot = np.array([[np.cos(a), -np.sin(a), 0.], [np.sin(a), np.cos(a), 0.], [0., 0., 1.]])
                if angle:
                    data.cam_xpos[self.cam] = self.pivot + rot @ (pos - self.pivot)
                    data.cam_xmat[self.cam] = (rot @ mat).reshape(9)
                    img = self.render(**settings)
                else:
                    data.cam_xpos[self.cam] = pos
                    data.cam_xmat[self.cam] = mat.reshape(9)
                    img = nominal
                extrinsics[angle] = dict(center_m=data.cam_xpos[self.cam].tolist(),
                                         camera_to_world=data.cam_xmat[self.cam].reshape(3, 3).tolist())
                images[angle] = downsample_agentview(img[::self.direction], 64)
        finally:
            data.cam_xpos[self.cam] = pos
            data.cam_xmat[self.cam] = mat.reshape(9)
        mujoco.mj_getState(model, data, after, state_flag)
        np.testing.assert_array_equal(before, after)
        if any(np.array_equal(images[0], images[a]) for a in self.angles if a):
            raise ValueError("Moving the camera did not change the rendered agentview")
        return dict(images=images, camera=extrinsics, time_s=float(data.time),
                    world_xyz_m=data.site_xpos[self.robot.eef_site_id].copy())

    def hook(self, *args, **kwargs):
        image = self.render(*args, **kwargs)
        if kwargs.get("camera_name") == "agentview":
            if args or kwargs.get("depth") or kwargs.get("segmentation"):
                raise ValueError(f"Unexpected sensor render call: {args}, {kwargs}")
            self.candidates.append(self.record(image, kwargs))
        return image

    def frame0(self, raw):
        settings = dict(camera_name="agentview", width=256, height=256)
        self.candidates = [self.record(self.render(**settings), settings)]
        return self.match(raw)

    def boundary_step(self, arm, command):
        # Capturing on the observable clock preserves even a cached native frame.
        self.candidates = []
        self.sim.render = self.hook
        try:
            step = arm.step(command)
        finally:
            self.sim.render = self.render
        return step, self.match(step["observation"])

    def match(self, raw):
        from embodiment import downsample_agentview

        expected = downsample_agentview(raw["agentview_image"], 64)
        expected_pos = raw["robot0_eef_pos"]
        for sample in reversed(self.candidates):
            if (np.array_equal(sample["images"][0], expected)
                    and np.array_equal(sample["world_xyz_m"].astype(np.float32), expected_pos)):
                return sample
        raise ValueError("No orbital render matches this native nominal frame and raw world grip-site sensor")


def replay(arm, case, entry, first, archive, angles):
    from embodiment import apply_embodiment_action, get_embodiment_spec, jaw_width_from_qpos, to_local_proprio

    obs = first["observation"]
    anchor = obs["robot0_eef_pos"].copy()
    q0 = obs["robot0_joint_pos"].copy()
    reader_state = motor_state(obs)
    np.testing.assert_allclose(reader_state, archive["frame0_state"], rtol=0, atol=2e-6)
    camera = OrbitalViews(arm, first, angles)
    frames = [camera.frame0(obs)]
    joints, jaws, local = [q0], [jaw_width_from_qpos(obs["robot0_gripper_qpos"])], [anchor]
    spec = get_embodiment_spec(entry["embodiment"])
    issued = np.repeat(archive["issued_mean"], 5, axis=0)
    physical = apply_embodiment_action(issued, spec, -np.ones(7), np.ones(7))
    rewards = []
    for i, command in enumerate(physical, start=1):
        if i % 5 == 0:
            step, sample = camera.boundary_step(arm, command)
            frames.append(sample)
            raw = step["observation"]
            joints.append(raw["robot0_joint_pos"].copy())
            jaws.append(jaw_width_from_qpos(raw["robot0_gripper_qpos"]))
            # Reproduce the reader's public LOCAL sensor from the fresh physical measurement.
            local.append(to_local_proprio(raw, anchor, spec)["robot0_eef_pos"] if spec.yaw_deg else raw["robot0_eef_pos"])
        else:
            step = arm.step(command)
        rewards.append(float(step["reward"]))
        if step["worker_step_index"] != i or (i < 15 and (step["terminated"] or step["truncated"])):
            raise ValueError(f"Fresh matched replay did not reach all 15 controls for {case}/{entry['embodiment']}")
    world = np.stack([f["world_xyz_m"] for f in frames])
    joint_deltas = np.diff(joints, axis=0)
    shared = dict(delta_joint=joint_deltas, delta_eef=np.diff(local, axis=0).astype(np.float32),
                  delta_jaw=np.stack((np.diff(jaws), jaws[1:]), axis=1).astype(np.float32),
                  issued_mean=archive["issued_mean"], chunk_steps=np.full(3, 5, np.int32),
                  joint_count=np.int32(6), valid=np.ones(3, bool), frame0_state=reader_state)
    fit = scored(entry, archive["issued_mean"], world)
    source_world = GripSite(arm).archived_positions(q0, archive["delta_joint"])
    nominal = np.stack([f["images"][0] for f in frames])
    archive_images = np.concatenate([archive["frame0_img"][None], archive["img_after"]])
    rec = dict(**case, **fit, index=entry["index"], embodiment=spec.name, archive_npz=str(entry["passive"]),
               archive_sha256=file_sha(entry["passive"]), replay_type="archived_chunk_mean_held_for_five",
               physics=first["environment_reproducibility"]["initial_state_fingerprint"],
               frame_times_s=[f["time_s"] for f in frames], orbital_pivot_m=camera.pivot.tolist(),
               camera_extrinsics=[f["camera"] for f in frames], primitive_issued=issued.tolist(),
               primitive_physical=physical.tolist(), native_rewards=rewards,
               archive_fk_to_fresh_public_endpoint_distance_mm=(1000 * np.linalg.norm(world - source_world, axis=1)).tolist(),
               archived_to_fresh_dq_max_error_rad=float(np.max(np.abs(joint_deltas - archive["delta_joint"]))),
               archive_to_fresh_nominal_rgb_mae=np.mean(np.abs(nominal.astype(float) - archive_images), axis=(1, 2, 3)).tolist())
    return frames, shared, rec


def main():
    p = arguments(__doc__)
    p.set_defaults(shards=[0])
    p.add_argument("--slots", type=int, choices=range(12), nargs="+", default=list(range(12)))
    p.add_argument("--angles", type=int, nargs="+", default=[0, -30, 30], help="World-Z azimuth degrees, include zero")
    p.add_argument("--output", type=Path, required=True, help="Fresh output DIRECTORY")
    args = p.parse_args()
    if (0 not in args.angles or len(set(a % 360 for a in args.angles)) != len(args.angles)
            or len(set(args.shards)) != len(args.shards) or len(set(args.slots)) != len(args.slots)):
        p.error("Supply distinct shards, slots and camera azimuths including zero")
    setup(args.source)
    from arm_env import ArmEnv

    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status="running", method="fresh_held_means_physical_replay", qualifications=[], rows=[])
    descriptors = {a: dict(name=f"fresh_held_means_{azimuth_name(a)}", rows=[]) for a in args.angles}
    try:
        for case, entries in panel(args.source, args.archive, args.shards):
            with ArmEnv("UR5e", case["suite"], case["task_id"]) as arm:
                first = reset(arm, case)
                report["qualifications"].append(qualify(case, entries, arm, first, GripSite(arm)))
                for slot in args.slots:
                    entry = entries[slot]
                    first = reset(arm, case)
                    if first["environment_reproducibility"]["initial_state_fingerprint"] != report["qualifications"][-1]["actual_physics"]:
                        raise ValueError("Different starting physics across replays of the same task")
                    archive = tape(entry["passive"])
                    frames, shared, rec = replay(arm, case, entry, first, archive, args.angles)
                    for angle in args.angles:
                        name = azimuth_name(angle)
                        images = np.stack([f["images"][angle] for f in frames])
                        rel = Path(name) / f"{entry['index']:02d}-{entry['embodiment']}.npz"
                        path = args.output / rel
                        path.parent.mkdir(exist_ok=True)
                        np.savez_compressed(path, **shared, img_before=images[:-1], img_after=images[1:], frame0_img=images[0])
                        descriptors[angle]["rows"].append(dict(
                            id=f"{case['suite']}/{case['task_id']}/{entry['embodiment']}",
                            **case, index=entry["index"], embodiment=entry["embodiment"], physics=rec["physics"],
                            camera_condition=name, chunk_steps=[5, 5, 5], passive_npz=str(rel), sha256=file_sha(path)))
                    report["rows"].append(rec)
                    write_json(args.output / "replay.json", report)
                    print(f"Rendered fresh replay {entry['index']} at {args.angles}", flush=True)
        for a, desc in descriptors.items():
            write_json(args.output / f"{azimuth_name(a)}.json", desc)
        report.update(status="complete")
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(args.output / "replay.json", report)


if __name__ == "__main__":
    main()
