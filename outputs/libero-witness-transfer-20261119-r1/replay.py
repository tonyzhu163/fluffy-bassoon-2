"""Replay the six bottle development poses with the frozen practice and query tapes."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path

# Select the archived Mesa runtime before NumPy, MuJoCo or OpenGL is imported.
os.environ.update(
    CUDA_VISIBLE_DEVICES="", LIBGL_ALWAYS_SOFTWARE="1", MUJOCO_EGL_DEVICE_ID="1",
    __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/50_mesa.json",
    MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", LP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
)

import numpy as np


OUTPUTS = Path(__file__).resolve().parents[1]


def load_replay(run, name):
    spec = spec_from_file_location(name, OUTPUTS / run / "replay.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


visual = load_replay("libero-visual-witness-20261119-r1", "visual_witness")
archive = load_replay("libero-interrupted-practice-20261119-r1", "interrupted_practice")
EPISODES = archive.ROOT / "outputs/libero-interrupted-practice-20261119-r1/closed01/episodes"
CONTROLS = 300
STRIDE = 25
BOTTLE = "wine_bottle_1"


def read_tape(pose, slot, mode):
    path = EPISODES / f"p{pose}-g{slot}-{mode}"
    record = json.loads((path / "result.json").read_text())
    assert (record["pose"], record["goal"], record["mode"]) == (pose, slot, mode), path
    assert record["seed"] == (19195100 if mode == "query" else 19195000) + pose, path
    with np.load(path / "tape.npz", allow_pickle=False) as tape:
        arrays = {key: tape[key] for key in ("states", "issued", "hits")}
    assert arrays["issued"].shape == (CONTROLS, 7) and np.isfinite(arrays["issued"]).all(), path
    assert arrays["hits"].shape == (CONTROLS, 2) and len(arrays["states"]) == CONTROLS, path
    return arrays, record, path


def replay(pose, slot, variant, bddl, predicates, bank, source):
    tape, record, path = source
    env = visual.LiberoEnvironmentAdapter()
    env._env = visual.ControlEnv(str(bddl), use_camera_obs=False, has_offscreen_renderer=True,
                                 camera_names=[])
    env._init_states = bank
    env._max_steps = CONTROLS
    try:
        reset = env.reset(pose, 18180100 + pose)
        physics = reset["environment_reproducibility"]["initial_state_fingerprint"]
        assert physics == record["physics"], (path, variant)
        world = env._env.env
        sim = world.sim
        initial_state = visual.robot_state(reset["observation"])
        assert not any(all(world._eval_predicate(t) for t in terms) for terms in predicates)
        frames = [visual.images(sim)]
        renderer = visual.GL.glGetString(visual.GL.GL_RENDERER).decode()
        assert "llvmpipe" in renderer.lower(), renderer
        states, hits, issued = [], [], []
        sim_states, bottle_xy = [], []
        joint = world.objects_dict[BOTTLE].joints[0]
        event = None
        for control, command in enumerate(tape["issued"], 1):
            issued.append(command.copy())
            step = env.step(command.copy())
            states.append(visual.robot_state(step["observation"]))
            hits.append([all(world._eval_predicate(t) for t in terms) for terms in predicates])
            assert step["success"] == hits[-1][0], (path, variant, control)
            if control % STRIDE == 0:
                frames.append(visual.images(sim))
            if variant != "QUERY":
                sim_states.append(sim.get_state().flatten().copy())
                bottle_xy.append(sim.data.get_joint_qpos(joint)[:2].copy())
            if variant == "D18":
                diagnostics = world.perturbation_runtime.diagnostics()
                assert diagnostics["case_id"] == "D18"
                changes = [e for e in diagnostics["events"] if e["event"] == "dynamic_state_change"]
                if changes:
                    assert len(changes) == 1 and changes[0]["target"] == BOTTLE, changes
                    if event is None:
                        event = dict(control=control, details=dict(changes[0]))
                    else:
                        assert event["details"] == changes[0], changes
    finally:
        env.close()

    data = dict(frames=np.stack(frames), states=np.stack(states), issued=np.stack(issued),
                hits=np.asarray(hits), initial_state=initial_state, physics=physics)
    assert data["frames"].shape == (13, 2, 96, 96, 3) and data["frames"].dtype == np.uint8
    np.testing.assert_array_equal(data["issued"], tape["issued"])
    if variant == "D18":
        assert event is not None, (path, "missing wine bottle event")
    verified = CONTROLS if event is None else event["control"] - 1
    for field in ("states", "hits"):
        np.testing.assert_array_equal(data[field][:verified], tape[field][:verified],
                                      err_msg=f"{path} {variant} {field}")
    row = dict(pose=pose, goal=slot, variant=variant, physics=physics,
               start_state=visual.digest(initial_state), start_rgb=visual.digest(frames[0]),
               actions=visual.digest(data["issued"]), archive_source=str(path),
               archive_seed=record["seed"], archive_verified_controls=verified, renderer=renderer,
               events=[] if event is None else [event],
               any_goal=data["hits"].any(axis=0).tolist(), final=data["hits"][-1].tolist())
    return data, row, np.asarray(sim_states), np.asarray(bottle_xy)


def run_pose(job):
    pose, out = job
    base_path, predicates, bank = archive.setup("live")
    manifest = visual.TaskManifest.load(str(visual.RELEASE / "assets/task_manifest.json"))
    drop_path = manifest.resolve_bddl("libero_goal", 2, "D18")
    rows = []
    reset_hashes = None
    for slot in (0, 1):
        sham = read_tape(pose, slot, "sham")
        query = read_tape(pose, slot, "query")
        for variant, bddl, source in (("BASE", base_path, sham), ("D18", drop_path, sham),
                                      ("QUERY", base_path, query)):
            data, row, sim_states, bottle_xy = replay(pose, slot, variant, bddl, predicates, bank, source)
            hashes = tuple(row[k] for k in ("physics", "start_state", "start_rgb"))
            if reset_hashes is None:
                reset_hashes = hashes
            assert hashes == reset_hashes, (pose, slot, variant, "reset mismatch")
            if variant == "BASE":
                base, base_sim, base_xy = data, sim_states, bottle_xy
            elif variant == "D18":
                event = row["events"][0]
                n = event["control"] - 1
                np.testing.assert_array_equal(data["issued"], base["issued"])
                np.testing.assert_array_equal(sim_states[:n], base_sim[:n])
                frame_count = 1 + n // STRIDE
                np.testing.assert_array_equal(data["frames"][:frame_count], base["frames"][:frame_count])
                shift = bottle_xy[n] - base_xy[n]
                assert np.dot(shift, event["details"]["delta_xy"]) > 0, (pose, slot, event, shift)
                row["bottle_xy_difference_at_event"] = shift.tolist()
            with (out / f"p{pose}-g{slot}-{variant}.npz").open("xb") as output:
                np.savez_compressed(output, **data)
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    with (out / "branches.jsonl").open("x", buffering=1) as log, ProcessPoolExecutor(max_workers=4) as pool:
        for rows in pool.map(run_pose, ((pose, out) for pose in range(24, 30))):
            for row in rows:
                print(json.dumps(row), file=log)
            print("verified pose", rows[0]["pose"], flush=True)


if __name__ == "__main__":
    main()
