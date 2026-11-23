"""Linux CPU qualification using native rendering, released noise, and the archive reader."""
from __future__ import annotations

import csv
from importlib.metadata import version
import os
import socket
import sys

import numpy as np

import reference
from reference import require, write
from symmeta_adapter import digest, policy_observation, public, reader_process, run_job, units


def processed(obs):
    from lerobot.envs.configs import LiberoEnv as Config
    from lerobot.envs.utils import preprocess_observation
    from reground.policies.vla import _stack

    process, _ = Config().get_env_processors()
    return process(preprocess_observation(_stack([obs])))


class NoiseFixture:
    revision = "cpu-wiring-fixture"
    device, chunk_size, noise_dim = "cpu", 50, 32

    def predict_chunk(self, obs, tasks, seeds, query):
        from reground.policies.vla import VLA
        noise = VLA._noise(self, seeds, query).numpy()
        # Include current public state to exercise the requery path. Only the noise
        # generator and processor come from pi05 in this CPU wiring fixture.
        commands = noise[:, :, :7].copy() * np.float32(.15)
        for i, item in enumerate(obs):
            commands[i, :, :3] += processed(item)["observation.state"][0, :3].numpy() * np.float32(.3)
        return commands


def qualify(out):
    import torch
    from reground.envs.libero_env import TaskEnv
    from reground.eval import jobs, rollout
    from reground.method.controller import Base
    from reground.policies.vla import VLA
    from reground.envs.shifts import RotZ
    from fluffy_bassoon.xemb.embodiment import downsample_agentview, retarget_action_from_motor
    from unittest.mock import patch

    require(os.environ["CUDA_VISIBLE_DEVICES"] == "" and not torch.cuda.is_available(), "Expected CPU")
    out.mkdir(exist_ok=False)
    grid = units(reference.SEEDS)
    require(len(grid) == 90, "Invalid shard count")
    for s in reference.SEEDS:
        assigned = [tuple(e) for j in grid if j["seed"] == s for e in j["episodes"]]
        require(len(assigned) == len(set(assigned)) == 280, "Invalid seed grid")
        require(assigned == [tuple(e) for j in grid if j["seed"] == 0 for e in j["episodes"]], "Seed grids differ")
    job = dict(grid[0], episodes=[["spatial_0", 15, 0]])
    eps = [rollout.Episode(*job["episodes"][0])]
    ctx = jobs.WorkerCtx("pi05", "cpu", 1, print)
    ctx._vla = NoiseFixture()
    try:
        with reader_process() as (pipe, info):
            run_job(job, ctx, pipe, info, out/"matched", fixture=True)
        trace = jobs.load_traces(out/"matched")["spatial_0", 15, 0]
        cost, = reference.read(out/"matched/calibration.json")
        row, = csv.DictReader((out/"matched/results.csv").read_text().splitlines())

        # Read the same command tape through the unmodified stock run_batch and Base,
        # with the stock render schedule. This is qualification only, never a method input.
        class Playback:
            def predict_chunk(self, obs, tasks, seeds, query):
                chunk = np.zeros((1, 50, 7), np.float32)
                a = trace["A"][10*query:10*query+10]
                chunk[0, :len(a)] = a
                return chunk
        replay = rollout.run_batch(Playback(), ctx.pool, eps, Base(), job["cell"], jobs.speed_limit("pi05"))
        for field in ("A", "X", "speed"):
            np.testing.assert_array_equal(replay.traces[0][field], trace[field], err_msg="Native replay " + field)
        safety = ("steps","success","unsafe","contact_steps","fallen","displaced","table_steps","peak_speed","speed_viol")
        for field in safety:
            require(float(row[field]) == replay.rows[0][field], field)
    finally:
        ctx.close()

    # Independent capture at the actual rs -> LeRobot formatting boundary on this case.
    # The raw recorder exists only in this standalone qualification program.
    env = TaskEnv("spatial_0", job["cell"])
    snapshots = []
    original = env.le._format_raw_obs
    def compare_raw(raw):
        obs = original(raw)
        cut = public(obs)
        for key in ("agentview_image", "robot0_gripper_qpos", "robot0_eef_pos"):
            np.testing.assert_array_equal(cut[key], raw[key])
        np.testing.assert_array_equal(obs["pixels"]["image2"], raw["robot0_eye_in_hand_image"])
        proc = processed(obs)
        for name,key in (("image","agentview_image"), ("image2","robot0_eye_in_hand_image")):
            chw = proc["observation.images." + name][0].numpy()
            pix = np.rint(chw.transpose(1,2,0) * 255).astype(np.uint8)
            np.testing.assert_array_equal(pix, raw[key][::-1,::-1])
        raw_image = cut["agentview_image"]
        upright = np.rot90(raw_image, 2)
        reduced = np.rint(upright.reshape(64,4,64,4,3).mean(axis=(1,3))).astype(np.uint8)
        np.testing.assert_array_equal(downsample_agentview(raw_image), reduced)
        require(digest(raw_image) != digest(upright), "Symmetric camera frame does not distinguish orientation")
        snapshots.append(dict(control=env.t, raw=digest(raw_image), reader_64=digest(reduced),
                              upright=digest(upright), eef=cut["robot0_eef_pos"].tolist(),
                              gripper_qpos=cut["robot0_gripper_qpos"].tolist()))
        return obs
    try:
        with patch.object(env.le, "_format_raw_obs", compare_raw):
            obs = env.reset(15, 0, render=True)["obs"]
            anchor = public(obs)["robot0_eef_pos"].copy()
            require(digest(ctx._vla.predict_chunk([obs], [env.description], [eps[0].noise_seed], 0)[0])
                    == cost["queries"][0]["proposal_sha256"], "Native opening proposal differs")
            for i, action in enumerate(trace["A"][:15], 1):
                result = env.step(action, render=i%5 == 0)
                if i == 10:
                    q = cost["queries"][1]
                    adapted = policy_observation(result["obs"], anchor, q["estimate"][0])
                    new = processed(adapted)["observation.state"][0].numpy()
                    native = processed(result["obs"])["observation.state"][0].numpy()
                    np.testing.assert_array_equal(new[3:], native[3:])
                    np.testing.assert_array_equal(new[:3], np.asarray(q["policy_eef"], np.float32))
                    require(np.any(new[:2] != native[:2]), "State correction not exercised")
                    require(digest(ctx._vla.predict_chunk([adapted], [env.description], [eps[0].noise_seed], 1)[0])
                            == q["proposal_sha256"], "Second policy query differs")
        for j, snap in enumerate(snapshots):
            if j:
                require(snap["raw"] == cost["updates"][j-1]["after_sha256"] and
                        snap["reader_64"] == cost["updates"][j-1]["after_64_sha256"], "After camera differs")
            if j < len(snapshots)-1:
                require(snap["raw"] == cost["updates"][j]["before_sha256"] and
                        snap["reader_64"] == cost["updates"][j]["before_64_sha256"], "Before camera differs")
        for j, u in enumerate(cost["updates"][:3]):
            for tag, snap in (("before", snapshots[j]), ("after", snapshots[j+1])):
                np.testing.assert_array_equal(u[tag+"_eef"], snap["eef"])
                np.testing.assert_array_equal(u[tag+"_jaw_qpos"], snap["gripper_qpos"])
        # Replay each complete or partial five-command slice with the archived
        # inverse, including intermediate updates at policy indices ending in five.
        for t in range(10, len(trace["A"]), 5):
            estimate = cost["updates"][min(t//5, len(cost["updates"]))-1]["next_adapter"]
            expected = retarget_action_from_motor(np.clip(trace["A_vla"][t:t+5], -1, 1), *estimate,
                                                  -np.ones(7, np.float32), np.ones(7, np.float32))
            np.testing.assert_array_equal(trace["A"][t:t+5], expected)
        from OpenGL.GL import glGetString, GL_VENDOR, GL_RENDERER
        gl = [glGetString(k).decode() for k in (GL_VENDOR, GL_RENDERER)]
    finally:
        env.close()
    # The environment rotates both XYZ triplets. Only its public input/output tape
    # is used above. This direct check documents the target in the CPU harness.
    probe = np.array([.25,-.5,.75,-.125,.375,.625,-1.])
    shifted = RotZ(90).transform(probe)
    np.testing.assert_allclose(shifted, [.5,.25,.75,-.375,-.125,.625,-1.], rtol=0, atol=1e-15)
    report = dict(host=socket.gethostname(), gpu_hours=0, python=sys.version, renderer=gl,
                  scope="native simulator, noise-driven fixture policy, frozen reader; wiring only",
                  assignment_count=sum(len(j["episodes"]) for j in grid), unique_cases=280,
                  revision=reference.PIN, code_revision=os.environ["SYMMETA_CODE_REV"],
                  assets=reference.read(reference.ROOT/"assets-verified.json"),
                  runtime={k:version(k) for k in ("lerobot","hf-libero","robosuite","mujoco","numpy","torch")},
                  reader=info, task=eps[0].task, init_id=15, rep=0, scene_seed=0, noise_seed=eps[0].noise_seed,
                  noise_first_two=[digest(VLA._noise(ctx._vla,[eps[0].noise_seed],j).numpy()[0]) for j in range(2)],
                  controls=cost["scored_controls"], queries=len(cost["queries"]), updates=len(cost["updates"]),
                  first_changed=cost["first_changed_control"], extra_renders=cost["extra_render_requests"],
                  exact_replayed_fields=["X","A","speed",*safety], native_replay={k:replay.rows[0][k] for k in safety},
                  snapshots=snapshots, first_two_queries=cost["queries"][:2], first_two_updates=cost["updates"][:2],
                  angular_commands_match_clipped_proposal=True, archived_inverse_commands_checked=len(trace["A"])-10,
                  public_eef_and_jaw_match_raw=True, probe_controls=0, extra_resets=0,
                  native_rotation_example=dict(input=probe.tolist(), transformed=shifted.tolist()))
    write(out/"qualified.json", report)
    print(f"Qualified {out/'qualified.json'}", flush=True)
