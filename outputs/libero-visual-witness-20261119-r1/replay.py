"""Replay frozen bowl commands through both branches of the released placement drop."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np
from libero.libero import benchmark
from libero.libero.envs.bddl_utils import robosuite_parse_problem
from libero.libero.envs.env_wrapper import ControlEnv
from libero_remote.environment.libero import LiberoEnvironmentAdapter
from libero_remote.environment.task_manifest import TaskManifest
from OpenGL import GL


ROOT = Path("/mowenjie/fluffy-bassoon-2")
ARCHIVE = ROOT / "outputs/libero-bowl-decisions-20261118-r1"
RELEASE = ROOT / "sources/libero-pro2-e5500c69"
GOALS = (4, 8)
CAMERAS = ("agentview", "robot0_eye_in_hand")


def digest(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def robot_state(raw):
    return np.concatenate([raw[k] for k in ("robot0_eef_pos", "robot0_eef_quat_xyzw", "robot0_gripper_qpos")])


def images(sim):
    # Flip to the policy-facing orientation. Raw MuJoCo pixels are bottom-up;
    # robosuite's default convention exposes them unchanged to the collector.
    return np.stack([sim.render(96, 96, camera_name=camera)[::-1, ::-1].copy() for camera in CAMERAS])


def run_pair(job):
    pose, goal, out = job
    source = ARCHIVE / ("paired01" if pose < 16 else "pacing01")
    path = source / "episodes" / f"s{pose}-BASE-g{goal}-gain1"
    archive_row = json.loads((path / "result.json").read_text())
    with np.load(path / "public.npz") as tape:
        issued, reference_states = tape["issued"], tape["state_after"]
    reference_hits = np.load(path / "evaluator-predicates.npy")
    assert issued.shape == (300, 7) and reference_hits.shape == (300, 2)
    manifest = TaskManifest.load(str(RELEASE / "assets/task_manifest.json"))
    predicates = [
        robosuite_parse_problem(str(manifest.resolve_bddl("libero_goal", g, "BASE")))["goal_state"] for g in GOALS
    ]
    bank = benchmark.get_benchmark_dict()["libero_goal"]().get_task_init_states(4)
    rows = []
    for variant in ("BASE", "D18"):
        env = LiberoEnvironmentAdapter()
        env._env = ControlEnv(str(manifest.resolve_bddl("libero_goal", 4, variant)),
                              use_camera_obs=False, has_offscreen_renderer=True, camera_names=[])
        env._init_states = bank
        env._max_steps = 300
        try:
            initial = env.reset(pose, 18180100 + pose)
            initial_state = robot_state(initial["observation"])
            physics = initial["environment_reproducibility"]["initial_state_fingerprint"]
            world = env._env.env
            assert not any(all(world._eval_predicate(t) for t in terms) for terms in predicates)
            frames = [images(world.sim)]
            renderer = GL.glGetString(GL.GL_RENDERER).decode()
            assert "llvmpipe" in renderer, renderer
            states, hits, events = [], [], []
            for step, action in enumerate(issued, 1):
                result = env.step(action)
                states.append(robot_state(result["observation"]))
                hits.append([all(world._eval_predicate(t) for t in terms) for terms in predicates])
                assert result["success"] == hits[-1][0]
                if step % 25 == 0:
                    frames.append(images(world.sim))
                if variant == "D18" and not events:
                    diagnostics = world.perturbation_runtime.diagnostics()
                    assert diagnostics["case_id"] == variant
                    events = [v for v in diagnostics["events"] if v["event"] == "dynamic_state_change"]
                    if events:
                        assert len(events) == 1 and events[0]["target"] == "akita_black_bowl_1"
                        events = [dict(control=step, details=events[0])]
            states, hits, frames = np.stack(states), np.array(hits), np.stack(frames)
            if variant == "BASE":
                np.testing.assert_array_equal(hits, reference_hits)
                np.testing.assert_allclose(states, reference_states, atol=1e-6, rtol=0)
                base_states, base_hits = states, hits
            else:
                prefix = events[0]["control"] - 1 if events else len(issued)
                np.testing.assert_array_equal(states[:prefix], base_states[:prefix])
                np.testing.assert_array_equal(hits[:prefix], base_hits[:prefix])
            np.savez_compressed(out / f"p{pose}-g{goal}-{variant}.npz", issued=issued, states=states,
                                hits=hits, frames=frames, initial_state=initial_state, physics=physics)
            rows.append(dict(pose=pose, goal=goal, variant=variant, events=events, physics=physics,
                             archive_physics_equal=physics == archive_row["initial_physics"],
                             archive_max_state_error=float(np.max(np.abs(states-reference_states))),
                             start_rgb=digest(frames[0]), start_state=digest(initial_state), actions=digest(issued),
                             archive_source=str(path), renderer=renderer, any_goal=hits.any(axis=0).tolist(),
                             final=hits[-1].tolist()))
        finally:
            env.close()
    for key in ("physics", "start_rgb", "start_state", "actions"):
        assert rows[0][key] == rows[1][key], (pose, goal, key)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    jobs = list(product(range(12, 20), range(2), [args.out]))
    rows = []
    with (args.out / "branches.jsonl").open("x", buffering=1) as file, ProcessPoolExecutor(max_workers=4) as pool:
        for pair in pool.map(run_pair, jobs):
            for row in pair:
                print(json.dumps(row), file=file)
            rows.extend(pair)
            print("verified pair", pair[0]["pose"], pair[0]["goal"], flush=True)
    for pose in range(12, 20):
        for key in ("start_rgb", "start_state", "physics"):
            assert len({row[key] for row in rows if row["pose"] == pose}) == 1, (pose, key)


if __name__ == "__main__":
    main()
