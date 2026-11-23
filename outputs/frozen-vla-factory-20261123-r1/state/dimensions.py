"""Probe the effect of each consumed state channel on one held-out real input."""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from run import build_policy, digest, observations, rmse, source_pose


@torch.inference_mode()
def probe(root):
    policy, pre, post, _ = build_policy(root)
    folder = root / "ur5e"
    frames = pd.read_parquet(folder / "data/chunk-000/file-000.parquet")
    traj = frames[frames.episode_index == frames.loc[frames.task_index == 2, "episode_index"].min()]
    row = traj.iloc[len(traj) // 2]
    episode = pd.read_parquet(folder / "meta/episodes/chunk-000/file-000.parquet").set_index("episode_index").loc[row.episode_index]
    images = observations(folder, episode, row)
    batch = pre({**images, "observation.state": torch.from_numpy(source_pose(row["observation.state"])),
                 "task": str(episode.tasks[0])})
    noise = torch.randn(1, 50, 32, generator=torch.Generator().manual_seed(817119))
    def forward(state):
        policy.reset()
        return post(policy.predict_action_chunk({**batch, "observation.state": state}, noise=noise.clone()))[0]
    reference = forward(batch["observation.state"])
    changes = {}
    for idx, channel in enumerate(["x", "y", "z", "rot_x", "rot_y", "rot_z", "finger_0", "finger_1"]):
        state = batch["observation.state"].clone()
        state[0, idx] += 1.0  # after checkpoint normalization
        action = forward(state)
        changes[channel] = rmse(action.numpy(), reference.numpy())
    return dict(arm="ur5e", task=2, episode=int(row.episode_index), frame=int(row.frame_index),
                noise_sha=digest(noise), known_chunk_sha=digest(reference), full50_rmse_plus_one_std=changes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(117)
    torch.set_num_threads(6)
    args.out.write_text(json.dumps(probe(args.assets.resolve()), indent=2, allow_nan=False) + "\n")
