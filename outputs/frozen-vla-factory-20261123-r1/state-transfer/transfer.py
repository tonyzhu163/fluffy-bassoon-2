"""Leave-one-body-out and 15-target-frame controls for the frozen state experiment."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

STATE = Path(__file__).resolve().parent.parent / "state"
sys.path.insert(0, str(STATE))
import run as original


def calibrations(frames):
    """Return features, canonical TCP and a frame manifest for both support sizes."""
    supports, manifests = {}, {}
    for name in ("full", "few"):
        joints, poses, manifest = [], [], []
        for task in (0, 1):
            episodes = sorted(frames.loc[frames.task_index == task, "episode_index"].unique())
            selected = episodes[:3] if name == "full" else episodes[3:5 if task == 0 else 4]
            for ep in selected:
                traj = frames[frames.episode_index == ep].sort_values("frame_index")
                sample = traj.iloc[np.linspace(0, len(traj) - 1, 24 if name == "full" else 5, dtype=int)]
                assert sample.frame_index.is_unique
                indices = sample.frame_index.astype(int).tolist()
                manifest.append(dict(task=task, episode=int(ep), frames=indices))
                joints.extend(original.encode_joints(q) for q in sample["observation.states.joint_state"])
                poses.extend(original.source_pose(p) for p in sample["observation.states.ee_state"])
        supports[name] = np.stack(joints), np.stack(poses)
        manifests[name] = manifest
    assert len(supports["full"][0]) == 144 and len(supports["few"][0]) == 15
    assert {r["episode"] for r in manifests["full"]}.isdisjoint(r["episode"] for r in manifests["few"])
    return supports, manifests


def summarize(rows):
    metrics = pd.DataFrame([r["first5_bounded_rmse"] for r in rows])
    metrics.index = pd.MultiIndex.from_tuples([(r["arm"], r["task"]) for r in rows], names=["arm", "task"])
    for condition in ("other_arm_only", "target_15"):
        for baseline in ("raw_joint_slots", "drop", "joint_to_pose"):
            metrics[f"{baseline}_minus_{condition}"] = metrics[baseline] - metrics[condition]
    metrics["other_arm_only_minus_target_15"] = metrics.other_arm_only - metrics.target_15
    panels = {arm: metrics.xs(arm).sort_index() for arm in original.ARMS}
    panels["both"] = metrics.groupby("task").mean()
    draws = np.random.default_rng(915).integers(8, size=(10000, 8))
    result = {}
    for panel, values in panels.items():
        assert values.index.tolist() == list(original.TASKS)
        result[panel] = {}
        for name, column in values.items():
            sampled = column.to_numpy()[draws].mean(axis=1)
            result[panel][name] = [float(column.mean()), *np.quantile(sampled, [0.025, 0.975]).tolist()]
    return result


@torch.inference_mode()
def evaluate(root):
    archived = json.loads((STATE / "scores.json").read_text())
    policy, pre, post, mean = original.build_policy(root)
    supports, manifests, data = {}, {}, {}
    for arm in original.ARMS:
        data[arm] = pd.read_parquet(root / arm / "data/chunk-000/file-000.parquet")
        supports[arm], manifests[arm] = calibrations(data[arm])
    pooled = tuple(np.concatenate([supports[arm]["full"][i] for arm in original.ARMS]) for i in (0, 1))
    for new, previous in zip(pooled, original.joint_fit(data)[:2], strict=True):
        np.testing.assert_array_equal(new, previous)

    results = []
    for arm in original.ARMS:
        other, = [name for name in original.ARMS if name != arm]
        queries = [r for r in archived["rows"] if r["arm"] == arm]
        assert [r["task"] for r in queries] == list(original.TASKS)
        calibration_episodes = {item["episode"] for manifest in manifests[arm].values() for item in manifest}
        assert calibration_episodes.isdisjoint(r["episode"] for r in queries)
        episodes = pd.read_parquet(root / arm / "meta/episodes/chunk-000/file-000.parquet").set_index("episode_index")
        for query in queries:
            ep, frame, task = query["episode"], query["frame"], query["task"]
            frames = data[arm]
            selected = frames[(frames.episode_index == ep) & (frames.frame_index == frame)]
            assert len(selected) == 1 and selected.iloc[0].task_index == task
            row = selected.iloc[0]
            episode = episodes.loc[ep]
            prompt, = episode.tasks
            assert prompt == query["prompt"]
            images = original.observations(root / arm, episode, row)
            assert {k: original.digest(v) for k, v in images.items()} == query["images"]
            noise = torch.randn(1, 50, 32, generator=torch.Generator().manual_seed(query["noise_seed"]))
            assert original.digest(noise) == query["noise_sha"]

            # An empty yaw-support list skips coordinate-replay controls in this follow-up.
            states = original.state_inputs(row, None, pooled, [])
            del states["reported_tcp"]
            states["drop"] = mean
            fingers = row["observation.states.gripper_state"]
            angles = row["observation.states.joint_state"]
            pools = {"other_arm_only": supports[other]["full"], "target_15": supports[arm]["few"]}
            for method, pool in pools.items():
                states[method] = np.r_[original.local_pose(angles, pool), fingers].astype(np.float32)

            chunks = {}
            for method, state in states.items():
                policy.reset()
                batch = pre({**images, "observation.state": torch.from_numpy(state.copy()), "task": str(prompt)})
                action = post(policy.predict_action_chunk(batch, noise=noise.clone()))
                assert action.shape == (1, 50, 7) and torch.isfinite(action).all()
                chunks[method] = action[0]
            assert original.digest(chunks["known"]) == query["known_chunk_sha"]
            bounded = {k: v[:5].numpy().clip(-1, 1) for k, v in chunks.items()}
            errors = {k: original.rmse(v, bounded["known"]) for k, v in bounded.items()}
            full = {k: original.rmse(v.numpy(), chunks["known"].numpy()) for k, v in chunks.items()}
            for control in ("known", "raw_joint_slots", "drop", "joint_to_pose"):
                assert errors[control] == query["first5_bounded_rmse"][control]
                assert full[control] == query["full50_unbounded_rmse"][control]
            position_error = {k: float(np.linalg.norm(states[k][:3] - states["known"][:3])) for k in pools}
            results.append(dict(arm=arm, source=other, task=task, episode=ep, frame=frame,
                                noise_seed=query["noise_seed"], oracle_sha256=query["known_chunk_sha"],
                                first5_bounded_rmse=errors, full50_unbounded_rmse=full,
                                position_error_m=position_error))
            print(arm, task, errors, flush=True)
    return dict(model_sha256=original.WEIGHT_SHA, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                original_evidence_commit="89bfe0ced881a001687c69a3bcde8af147edc545",
                total_parameters=sum(p.numel() for p in policy.parameters()),
                trainable_parameters=sum(p.numel() for p in policy.parameters() if p.requires_grad),
                device="cpu", calibration=manifests, rows=results, task_bootstrap95=summarize(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(117)
    torch.set_num_threads(6)
    result = evaluate(args.assets.resolve())
    args.out.write_text(json.dumps(result, allow_nan=False, indent=2) + "\n")
