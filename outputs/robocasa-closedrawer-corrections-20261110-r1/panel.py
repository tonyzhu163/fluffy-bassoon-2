"""Matched base corrections on the eight registered native CloseDrawer scenes."""
import json
from pathlib import Path
import sys
import time

import imageio.v2 as imageio
import jax
import numpy as np

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "icrl-idea-checks-20261110-r1"
sys.path.insert(0, str(PRIOR))
from checks import (HORIZON, config, convert_action, geometry, handle, inputs, matched_start,
                    normalize, physics, policy_config, public_hash, stable_counter_regions,
                    with_checkpoint_stats)

ARMS = {"native": None, "wait": (0., 0.), "base_y_minus": (0., -.45),
        "base_y_plus": (0., .45), "base_x_plus": (.45, 0.)}
SLOTS = 40
DISTANCE_M = .12


def run_episode(policy, start, scene, repeat, arm, video):
    env, obs = start
    writer = None
    try:
        raw = env.unwrapped.env
        robot = raw.robots[0]
        _, centers = handle(raw)
        model = raw.sim.model
        drawer_geoms = [model.geom_id2name(i) for i in range(model.ngeom)
                        if (model.geom_id2name(i) or "").startswith(raw.drawer.naming_prefix)
                        and (model.geom_contype[i] or model.geom_conaffinity[i])]
        assert drawer_geoms
        noise_seed = scene + 700000 + 1000 * repeat
        noise = np.random.default_rng(noise_seed).standard_normal((HORIZON // 5, 50, 32), dtype=np.float32)
        if video:
            writer = imageio.get_writer(str(video), fps=10, codec="libx264", macro_block_size=1)

        def opening():
            return float(max(raw.drawer.get_door_state(env=raw).values()))

        initial_state, initial_obs = physics(raw), public_hash(obs)
        base_start = np.asarray(obs["state.base_position"], dtype=float).copy().reshape(-1)[:2]
        initial_open = min_open = opening()
        nearest = float(np.linalg.norm(geometry(raw, robot, centers)))
        first_contact = first_motion = None
        base_delta = None
        peak_base_m = 0.
        stopped = False
        active_slots = 0
        chunk = None
        samples = []
        for t in range(HORIZON):
            if writer is not None and t % 5 == 0:
                public = inputs(obs)
                writer.append_data(np.concatenate([public["observation/image"], public["observation/wrist_image"]], axis=1))
            if arm != "native" and t < SLOTS:
                action = np.zeros(12, dtype=np.float32)
                action[11] = 1.  # Native base mode, open gripper and hold arm relative to the base.
                if not stopped:
                    action[7:9] = ARMS[arm]
                    active_slots += int(bool(np.any(action[7:9])))
            else:
                if t % 5 == 0:
                    chunk = np.asarray(policy.infer(inputs(obs), noise=noise[t // 5])["actions"])
                    assert chunk.shape == (50, 12) and np.isfinite(chunk).all()
                action = chunk[t % 5]
            obs, _, done, truncated, info = env.step(convert_action(action))
            distance = float(np.linalg.norm(geometry(raw, robot, centers)))
            nearest = min(nearest, distance)
            current_open = opening()
            min_open = min(min_open, current_open)
            if first_contact is None and raw.check_contact(robot.gripper["right"], drawer_geoms):
                first_contact = t + 1
            if first_motion is None and initial_open - current_open >= .05:
                first_motion = t + 1
            delta = np.asarray(obs["state.base_position"], dtype=float).reshape(-1)[:2] - base_start
            if t < SLOTS:
                base_delta = delta.tolist()
                peak_base_m = max(peak_base_m, float(np.linalg.norm(delta)))
                stopped = stopped or peak_base_m >= DISTANCE_M
            success = bool(info["success"])
            if t % 10 == 0 or success or done or truncated or t == HORIZON - 1:
                samples.append([t + 1, round(current_open, 4), round(distance, 4)])
            if success or done or truncated:
                break
        return dict(scene=scene, repeat=repeat, noise_seed=noise_seed, arm=arm,
                    task="CloseDrawer", fixture=raw.drawer.name, side=raw.drawer_side,
                    prompt=obs["annotation.human.task_description"], initial_sha256=initial_state,
                    initial_observation_sha256=initial_obs, terminal_sha256=physics(raw),
                    success=success, controls=t+1, first_drawer_contact=first_contact,
                    first_five_percent_motion=first_motion, min_handle_distance_m=nearest,
                    initial_open=initial_open, min_open=min_open, final_open=current_open,
                    base_delta_after_window_m=base_delta, peak_base_displacement_m=peak_base_m,
                    commanded_base_steps=active_slots, trace=samples)
    finally:
        if writer is not None:
            writer.close()
        env.close()


def main():
    out, ckpt, repeat = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    assert repeat in (0, 1)
    prior = [json.loads(line) for line in (PRIOR / "closedrawer02/events.jsonl").read_text().splitlines()]
    signatures = {r["scene"]: r["signature"] for r in prior
                  if r["kind"] == "reset_check" and r["selected"] and r["repeat"] == repeat}
    archived = {r["scene"]: r for r in prior if r["kind"] == "episode" and r["repeat"] == repeat}
    assert set(archived) == set(signatures) == set(range(983000, 983008))
    assert len(jax.devices()) == 1 and jax.devices()[0].platform == "gpu"
    stable_counter_regions()
    stats = normalize.load(ckpt / "assets")
    cfg = with_checkpoint_stats(config.get_config("pi05_pretrain_human300"), stats)
    policy = policy_config.create_trained_policy(cfg, ckpt, sample_kwargs={"num_steps": 10}, norm_stats=stats)

    def log(kind, **values):
        with (out / "events.jsonl").open("a") as stream:
            print(json.dumps(dict(kind=kind, unix=time.time(), **values)), file=stream, flush=True)

    for arm in ARMS:
        for scene in sorted(archived):
            start, _ = matched_start("CloseDrawer", scene, repeat, signatures[scene], log)
            video = out / f"{scene}-{arm}.mp4" if repeat == 0 else None
            row = run_episode(policy, start, scene, repeat, arm, video)
            old = archived[scene]
            for key in ("initial_sha256", "initial_observation_sha256"):
                assert row[key] == old[key], (scene, repeat, arm, key)
            log("episode", **row)
            if arm == "native":
                assert row["success"] == old["success"], (scene, repeat, "archived outcome")
            print(scene, repeat, arm, row["success"], row["controls"], flush=True)
    log("complete", repeat=repeat)


if __name__ == "__main__":
    main()
