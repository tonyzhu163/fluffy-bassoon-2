"""Live pre-action replay of released Panda demonstrations and smooth approach variants."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path

import h5py
import numpy as np
from demos import json_write, save_tape
from oracle_fallback import ARCHIVE_SHA, BDDL_SHA
from sim import Session


def execute(job):
    archive, output, number, variant = job
    with h5py.File(archive, "r") as f:
        group = f[f"data/demo_{number}"]
        source = json.loads(group.attrs["source_metadata"])
        if not (group.attrs["success"] and group["rewards"][-1] > 0 and group["dones"][-1]):
            raise ValueError("Released reference lacks native terminal success")
        commands = group["actions"][:].astype(np.float32)
        xyz_after = group["obs/robot0_eef_pos"][:]
    pose, seed = int(source["init_state_id"]), int(source["seed"])
    if pose > 31:
        raise ValueError("Reserved pose requested")
    rng = np.random.default_rng([20261120, number, variant])
    offset = rng.uniform([-0.024, -0.022, -0.012], [0.024, 0.022, 0.025])
    length = min(50, len(commands) // 2)
    frames, actions, motions, rewards = [], [], [], []
    with Session("Panda") as env:
        obs = env.reset(pose, seed)
        reset = env.reset_info
        if reset["environment_reproducibility"]["bddl_sha256"] != BDDL_SHA or source["instruction"] != obs["prompt"]:
            raise ValueError("Different task or prompt")
        initial = obs["state"][:3].copy()
        for t, command in enumerate(commands):
            action = command.copy()
            if variant and t < length:
                # Feedback to a smooth alternate approach. Offset vanishes before the push.
                ref = xyz_after[t-1] if t else initial
                amplitude = np.sin(np.pi * t / (length - 1)) ** 2
                action[:3] += 0.8 * (ref + amplitude * offset - obs["state"][:3]) / 0.05
                np.clip(action, -1, 1, out=action)
            frames.append(obs)
            obs, motion, reward, done, success = env.step(action)
            actions.append(action)
            motions.append(motion)
            rewards.append(reward)
            if done:
                break
        accepted = bool(done and success and reward > 0)
        row = dict(body="Panda", pose=pose, seed=seed, steps=len(actions), accepted=accepted,
                   reference_demo=number, variant=variant, native_success=success, terminal_reward=reward,
                   done=done, reset=reset, route="released_actions" if not variant else "feedback_approach")
        if accepted:
            file = Path("Panda") / f"{pose}-{seed}-d{number}-v{variant}.npz"
            save_tape(output / file, frames, actions, motions, rewards, obs["state"], obs["prompt"], "Panda", pose, seed)
            row["file"] = str(file)
        return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("bank"))
    args = parser.parse_args()
    archive = Path(os.environ["DUAL_ARCHIVE"])
    with archive.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != ARCHIVE_SHA:
            raise ValueError("Unverified demonstration archive")
    output = args.out.resolve()
    output.mkdir(exist_ok=False, parents=True)
    (output / "Panda").mkdir()
    jobs = []
    poses = set()
    with h5py.File(archive, "r") as f:
        for key, group in f["data"].items():
            pose = int(json.loads(group.attrs["source_metadata"])["init_state_id"])
            if pose < 32:
                poses.add(pose)
                for variant in range(args.variants if pose < 28 else 1):
                    jobs.append((archive, output, int(key.split("_")[1]), variant))
    index = dict(suite="libero_goal", task_id=5, case="BASE", poses=sorted(poses),
                 bodies=["Panda"], archive_sha256=ARCHIVE_SHA, seed_base=20261120, attempts=[])
    json_write(output / "index.json", index)
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for row in pool.map(execute, jobs):
            index["attempts"].append(row)
            json_write(output / "index.json", index)
            print("Panda", row["pose"], row["reference_demo"], row["variant"], row["steps"], row["terminal_reward"], flush=True)


if __name__ == "__main__":
    main()
