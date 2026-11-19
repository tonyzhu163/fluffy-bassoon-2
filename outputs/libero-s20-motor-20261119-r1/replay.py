"""Replay released fitting demonstrations and inspect issued command activity."""

import argparse
import json
from pathlib import Path
from statistics import median

import h5py
import numpy as np

from libero_remote.environment.libero import LiberoEnvironmentAdapter
from libero_remote.environment.task_manifest import TaskManifest


CAMERAS = ["agentview", "robot0_eye_in_hand"]
TASKS = (4, 8)
CONDITIONS = ("BASE", "S20")


def replay(demo, archive_name, demo_name, manifest):
    source = json.loads(demo.attrs["source_metadata"])
    spec = dict(source["env_args"], camera_names=CAMERAS)
    key = (spec["suite"], spec["task_id"], spec["variant"])
    case = manifest.require(*key)
    if spec["max_steps"] != case["max_steps"]:
        raise ValueError(f"Horizon mismatch for {key}")
    spec["_bddl_path"] = str(manifest.resolve_bddl(*key))
    actions = demo["actions"][:]
    recorded_position = demo["obs/robot0_eef_pos"][:]
    recorded_reward = demo["rewards"][:]
    env = LiberoEnvironmentAdapter()
    try:
        reset = env.create(spec)
        raw = reset["observation"]
        first_matches = {field: bool(np.array_equal(value, demo["obs"][field][0]))
                         for field, value in raw.items() if field in demo["obs"]}
        position_error = []
        reward_disagreements = 0
        for step, action in enumerate(actions):
            position_error.append(float(np.linalg.norm(raw["robot0_eef_pos"] - recorded_position[step])))
            result = env.step(action)
            reward_disagreements += int(result["reward"] != recorded_reward[step])
            raw = result["observation"]
            if result["terminated"] or result["truncated"]:
                break
    finally:
        env.close()
    return dict(archive=archive_name, demo=demo_name, seed=spec["seed"], reset_id=spec["init_state_id"],
                recorded_success=bool(demo.attrs["success"]), recorded_steps=len(actions),
                replay_success=result["success"], replay_steps=step + 1,
                instruction_match=reset["language_instruction"] == source["instruction"],
                first_matches=first_matches, peak_position_error_m=max(position_error),
                reward_disagreements=reward_disagreements)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("release", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = TaskManifest.load(str(args.release / "assets/task_manifest.json"))
    summaries, replays = [], []
    for task in TASKS:
        for case in CONDITIONS:
            name = f"{task}_{case}"
            with h5py.File(args.dataset / f"{name}.hdf5") as h5:
                demos = h5["data"]
                actions = [demo["actions"][:] for demo in demos.values()]
                counts = [int(np.sum(np.all(demo[:, :6] == 0, axis=1))) for demo in actions]
                summaries.append(dict(archive=name, n=len(actions), steps=sum(map(len, actions)),
                                      median_steps=median(map(len, actions)), zero_arm_steps=sum(counts),
                                      episodes_with_zero_arm=sum(count > 0 for count in counts)))
                for demo_name in ("demo_0", "demo_1"):
                    result = replay(demos[demo_name], name, demo_name, manifest)
                    print(json.dumps(result, sort_keys=True), flush=True)
                    replays.append(result)
    args.output.write_text(json.dumps(dict(command_summary=summaries, replays=replays), indent=2) + "\n")


if __name__ == "__main__":
    main()
