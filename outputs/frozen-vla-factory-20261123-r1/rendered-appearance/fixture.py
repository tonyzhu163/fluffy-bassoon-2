"""Acquire target observations and segregate oracle-only validation data."""

import argparse
import sys
from pathlib import Path

import numpy as np
from libero.libero import get_libero_path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "appearance"))
from render import match_scene, match_tool, render, reset


def export(work, i, task):
    seed = 3200 + i
    source, target = reset(task, "Panda", seed), reset(task, "UR5e", seed)
    try:
        idx = source.robots[0]._ref_joint_pos_indexes
        source.sim.data.qpos[idx] += np.random.default_rng(seed).uniform(-0.12, 0.12, len(idx))
        source.sim.forward()
        match_scene(source, target)
        match_tool(source, target)
        rgb, mask = render(source)
        native, tm = render(target)
        target._update_observables(force=True)
        obs = target.env._get_observations()
        np.testing.assert_array_equal(native, obs["agentview_image"])
        name = f"{i:02d}.npz"
        with np.load(work / "cases" / name) as original:
            for k, arr in [("native", native), ("source", rgb), ("source_mask", mask), ("target_mask", tm)]:
                np.testing.assert_array_equal(arr, original[k])
        # All deployment-side arrays below come directly from the UR5e.
        np.savez_compressed(work / "followup/public" / name, native=native,
                            wrist=obs["robot0_eye_in_hand_image"], pos=obs["robot0_eef_pos"],
                            quat=obs["robot0_eef_quat"], grip=obs["robot0_gripper_qpos"],
                            task=task.stem.replace("_", " "))
        np.savez_compressed(work / "followup/oracle" / name,
                            q=source.sim.data.qpos[idx], source=rgb, sm=mask, tm=tm)
    finally:
        target.close()
        source.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("work", type=Path)
    args = p.parse_args()
    for folder in ["public", "oracle"]:
        (args.work / "followup" / folder).mkdir(exist_ok=True, parents=True)
    tasks = sorted((Path(get_libero_path("bddl_files")) / "libero_goal").glob("*.bddl"))
    assert len(tasks) == 10
    for i, task in enumerate(tasks):
        export(args.work, i, task)
        print(i, task.stem, flush=True)
