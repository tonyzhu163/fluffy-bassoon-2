"""Collect matched contact-continuation outcomes from pinned frozen π0.5 tapes."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parents[1] / "libero-xemb-friction-dev-20261119-r1" / "cpu01"
BASE = "757a44cd4374833cad790852df8b87ddf412e638"
RELEASE = "e5500c690c1498eca1b7050350654964190cdf70"
PROGRAMS = ("continue", "slow", "regrasp")
SUFFIXES = ("finger1_collision", "finger2_collision", "finger1_pad_collision", "finger2_pad_collision")
FIELDS = ("robot0_eef_pos", "robot0_eef_quat_xyzw", "robot0_gripper_qpos")
CLOSE = 12


def pose(obs):
    return np.concatenate([obs[name] for name in FIELDS])


def public_features(commands, states):
    # Called at the fork, immediately before the first alternative command.
    commands = np.asarray(commands[-CLOSE:])
    states = np.asarray(states[-(CLOSE + 1):])
    jaw = states[:, 7:]
    gap = np.abs(jaw[:, 0] - jaw[:, 1])
    xyz = states[:, :3]
    return dict(command=np.r_[commands[:, :3].mean(0), commands[-1, :3]].tolist(),
                contact=np.r_[gap[-1], gap[-1] - gap[0], gap[-1] - gap[-7],
                              abs(jaw[-1].sum()), max(0., np.diff(gap).max()),
                              xyz[-1] - xyz[0]].tolist())


def continuation(program, actions, cut, prefix, state):
    """Start at the next unissued command, tracking real wrist during regrasp."""
    cursor = cut
    if program == "regrasp":
        start = max(0, cut - CLOSE - 8)
        raised = prefix[-1].copy()
        raised[2] += .10
        for target in (raised, prefix[start]):
            for _ in range(30):
                current = state()
                delta_rot = (Rotation.from_quat(target[3:7]) *
                             Rotation.from_quat(current[3:7]).inv()).as_rotvec()
                yield np.r_[.65 * np.clip((target[:3] - current[:3]) / .05, -1, 1),
                            .65 * np.clip(delta_rot / .5, -1, 1), -1.].astype(np.float32)
        cursor = start
    if program == "slow":
        for _ in range(10):
            if cursor == len(actions) or actions[cursor, 6] <= 0:
                break
            half = actions[cursor].copy()
            half[:6] *= .5
            yield half
            yield half
            cursor += 1
    yield from actions[cursor:]
    hold = np.r_[np.zeros(6), actions[-1, 6]].astype(actions.dtype)
    while True:
        yield hold


def run_branch(bddl, bank, record, actions, archive_trace, material, program):
    from libero.libero.envs.env_wrapper import ControlEnv
    from libero_remote.environment.libero import LiberoEnvironmentAdapter

    case = record["case"]
    np.random.seed(case["seed"])
    random.seed(case["seed"])
    env = LiberoEnvironmentAdapter()
    env._init_states = bank
    env._max_steps = record["identity"]["horizon"]
    env._env = ControlEnv(str(bddl), use_camera_obs=False, has_offscreen_renderer=False, camera_names=[])
    try:
        reset = env.reset(case["init_state_id"], case["seed"])
        reset_hash = reset["environment_reproducibility"]["initial_state_fingerprint"]
        world = env._env.env
        assert world.perturbation_runtime.diagnostics()["case_id"] == "BASE"
        model = world.sim.model
        finger_ids = [i for i in range(model.ngeom)
                      if (model.geom_id2name(i) or "").endswith(tuple("_" + s for s in SUFFIXES))]
        assert len(finger_ids) == 4
        if material == "reduced":
            model.geom_friction[finger_ids, 0] = .05

        obs = reset["observation"]
        current = pose(obs)
        prefix, commands = [current], []
        prefix_hash = sha256(current.tobytes())
        run_length, cut, features, fork_hash = 0, None, None, None
        archive_max_abs = dict.fromkeys(FIELDS, 0.)
        stream = continuation("continue", actions, 0, prefix, lambda: current)
        for n in range(1, env._max_steps + 1):
            command = next(stream)
            response = env.step(command.copy())
            obs = response["observation"]
            current = pose(obs)
            if program == "continue" and n < len(archive_trace[FIELDS[0]]):
                for key in FIELDS:
                    error = np.max(np.abs(obs[key] - archive_trace[key][n]))
                    archive_max_abs[key] = max(archive_max_abs[key], float(error))
            assert response["worker_step_index"] == n and response["reward"] == float(response["success"])
            if cut is None:
                prefix_hash.update(command.tobytes())
                prefix_hash.update(current.tobytes())
                commands.append(command)
                prefix.append(current)
                run_length = run_length + 1 if command[6] > 0 else 0
                if run_length == CLOSE and n < len(actions) and not response["terminated"] and not response["truncated"]:
                    cut, fork_hash = n, prefix_hash.hexdigest()
                    features = public_features(commands, prefix)
                    stream = continuation(program, actions, cut, prefix, lambda: current)
            if response["terminated"] or response["truncated"]:
                break
        row = dict(controls=n, success=int(response["success"]), cut=cut,
                   prefix_hash=fork_hash or prefix_hash.hexdigest(), reset_hash=reset_hash,
                   features=features)
        if program == "continue":
            row["archive_max_abs"] = archive_max_abs
        return row
    finally:
        env.close()


def collect_task(job):
    from libero.libero import benchmark

    task, release, sources, archived = job
    cells = []
    for mode in ("default", "paced"):
        source = sources[f"{task}-{mode}"]
        rec = source["record"]
        case = rec["case"]
        bank = benchmark.get_benchmark_dict()[case["suite"]]().get_task_init_states(case["task_id"])
        bddl = release / "assets" / source["bddl"].split("/assets/", 1)[1]
        assert sha256(bddl.read_bytes()).hexdigest() == source["bddl_sha256"]
        with np.load(ARCHIVE / f"{task}-{mode}-ordinary.npz") as ordinary:
            actions = ordinary["issued"]
        digest = sha256(str((actions.shape, actions.dtype.str)).encode() + actions.tobytes()).hexdigest()
        assert digest == source["actions_sha256"]
        for material in ("ordinary", "reduced"):
            # Reduced replays may terminate early; the original complete source tape is the ordinary one.
            with np.load(ARCHIVE / f"{task}-{mode}-{material}.npz") as data:
                reference = {k: data[k] for k in FIELDS}
                assert np.array_equal(actions[:len(data["issued"])], data["issued"])
            branches = [run_branch(bddl, bank, rec, actions, reference, material, p) for p in PROGRAMS]
            first = branches[0]
            for key in ("cut", "prefix_hash", "reset_hash", "features"):
                assert all(b[key] == first[key] for b in branches), (task, mode, material, key)
            old = archived[task, mode, material]
            cells.append(dict(task=task, cadence=mode, material=material, seed=case["seed"],
                              horizon=rec["identity"]["horizon"], tape_length=len(actions),
                              cut=first["cut"], features=first["features"], prefix_hash=first["prefix_hash"],
                              reset_hash=first["reset_hash"], archive_reset_match=first["reset_hash"] == old["physics"],
                              archive_max_abs=first["archive_max_abs"],
                              archive_success=int(old["success"]), archive_controls=old["controls"],
                              archive_first_finger=(old["first_finger"] or {}).get("control"),
                              native_hits=[b["success"] for b in branches], stops=[b["controls"] for b in branches]))
        print(task, mode, "complete", flush=True)
    return cells


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=HERE / "panel.json")
    args = ap.parse_args()
    release = args.release.resolve()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert subprocess.check_output(["git", "-C", str(release), "rev-parse", "HEAD"], text=True).strip() == RELEASE
    from libero.libero.envs import env_wrapper
    from libero_remote.environment import libero
    assert Path(libero.__file__).resolve().is_relative_to(release)
    assert Path(env_wrapper.__file__).resolve().is_relative_to(release / "simulator-source")
    assert version("mujoco") == "3.2.3" and version("robosuite") == "1.4.0"
    src = json.loads((ARCHIVE / "sources.json").read_text())
    assert Path(src["design"]["checkpoint"]).name == "pi05-libero-openpi-20261027-r1"
    archived = {(r["task"], r["mode"], r["material"]): r for r in
                map(json.loads, (ARCHIVE / "branches.jsonl").read_text().splitlines())}
    tasks = sorted({key[0] for key in archived})
    assert len(tasks) == 8 and len(archived) == 32
    started = time.time()
    jobs = [(task, release, src["sources"], archived) for task in tasks]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        cells = [cell for group in pool.map(collect_task, jobs) for cell in group]
    receipt = dict(base=BASE, release=RELEASE, tape_run="c45aa965db73ad0644382da578ff03a30b2083d7",
                   friction_run="a6d6cb8d208bf8df744986e03c802a89343fe8d3", motor=src["design"]["checkpoint"],
                   packages={k: version(k) for k in ("numpy", "mujoco", "robosuite", "scipy", "numba", "torch")},
                   python=sys.version.split()[0], workers=args.workers, started_unix=started,
                   seconds=round(time.time() - started, 2), gpu_hours=0,
                   branch_order=PROGRAMS, paired_cells=len(cells), episodes=len(cells) * len(PROGRAMS))
    args.out.write_text(json.dumps(dict(receipt=receipt, cells=cells), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
