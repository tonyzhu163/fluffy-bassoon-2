"""Read source demos, hold out entire simulator poses, and construct masked action chunks."""

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from motor_weights import sha256

BODIES = ("Panda", "Sawyer")
HORIZON = 5


def pose_key(stem):
    match = re.fullmatch(r"(?:pose[-_]?)?(\d+)", stem, re.IGNORECASE)
    return str(int(match[1])) if match else stem


def instruction(value):
    arr = np.asarray(value)
    if arr.dtype.kind != "U" or not arr.size:
        raise ValueError("Expected a numpy unicode instruction")
    flat = arr.reshape(-1)
    if not flat[0].strip() or not (flat == flat[0]).all():
        raise ValueError("Expected one nonempty instruction per episode")
    return str(flat[0])


@dataclass
class Episode:
    body: str
    pose: str
    rgb: np.ndarray
    wrist: np.ndarray
    state: np.ndarray
    actions: np.ndarray
    prompt: str

    def observation(self, t):
        return {"rgb": self.rgb[t], "wrist": self.wrist[t], "state": self.state[t], "prompt": self.prompt}

    def sample(self, t):
        end = min(t + HORIZON, len(self.actions))
        count = end - t
        target = np.zeros((HORIZON, 7), dtype=np.float32)
        target[:count] = self.actions[t:end]
        return self.observation(t), target, np.arange(HORIZON) < count


def read_episodes(root, scratch):
    root = Path(root).resolve(strict=True)
    episodes, records, seen = [], [], set()
    paths = sorted(root.rglob("*.npz"))
    for path in paths:
        relative = path.relative_to(root)
        if len(relative.parts) != 2 or relative.parts[0].capitalize() not in BODIES:
            raise ValueError(f"Unexpected source archive {relative}; expected Panda/Sawyer/<pose>.npz")
        body, pose = relative.parts[0].capitalize(), pose_key(path.stem)
        if (body, pose) in seen:
            raise ValueError(f"Duplicate source body/pose {relative}")
        seen.add((body, pose))
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Archive outside source directory: {relative}")
        with np.load(path, allow_pickle=False) as data:
            state, actions, motion = (data[k] for k in ("state", "actions", "motion"))
            n = len(actions)
            rewards = data["rewards"]
            if (
                not n
                or state.shape != (n, 8)
                or actions.shape != (n, 7)
                or state.dtype != np.float32
                or actions.dtype != np.float32
                or motion.shape != (n, 7)
                or not all(np.isfinite(v).all() for v in (state, actions, motion))
                or np.abs(actions).max() > 1
                or not np.allclose(np.linalg.norm(state[:, 3:7], axis=1), 1, atol=0.005)
            ):
                raise ValueError(f"{relative}: invalid state, issued action, or measured motion")
            if (
                rewards.shape != (n,)
                or not np.isfinite(rewards).all()
                or np.any(rewards[:-1] != 0)
                or rewards[-1] <= 0
            ):
                raise ValueError(f"{relative}: expected native sparse reward at the final control")
            for key in ("body", "arm"):
                if key in data and str(data[key].item()).capitalize() != body:
                    raise ValueError(f"{relative}: {key} disagrees with source directory")
            if "pose" in data and pose_key(str(data["pose"].item())) != pose:
                raise ValueError(f"{relative}: pose disagrees with filename")
            for key in ("success", "native_success"):
                if key in data and data[key].item() != 1:
                    raise ValueError(f"{relative}: {key} disagrees with native reward")
            prompt = instruction(data["prompt"])
            episode_seed = int(data["seed"].item()) if "seed" in data else None
        # Random frame access into a compressed npz would inflate an entire video
        # for every sample. Stage each .npy once, then let Linux page it on demand.
        views = []
        with ZipFile(path) as archive:
            for key in ("rgb", "wrist"):
                dest = scratch / f"{len(episodes)}-{key}.npy"
                with archive.open(key + ".npy") as source, dest.open("wb") as out:
                    shutil.copyfileobj(source, out, length=1024 * 1024)
                images = np.load(dest, mmap_mode="r", allow_pickle=False)
                # Linux keeps the mapping alive and reclaims it even on job termination.
                dest.unlink()
                if (
                    images.ndim != 4
                    or images.shape[0] != n
                    or images.shape[-1] != 3
                    or min(images.shape[1:3]) < 1
                    or images.dtype != np.uint8
                ):
                    raise ValueError(f"{relative}: invalid {key} camera")
                views.append(images)
        episodes.append(Episode(body, pose, *views, state, actions, prompt))
        records.append(
            {
                "file": str(relative),
                "sha256": sha256(path),
                "body": body,
                "pose": pose,
                "frames": n,
                "rgb_shape": list(views[0].shape),
                "wrist_shape": list(views[1].shape),
                "prompt": prompt,
                "seed": episode_seed,
                "native_terminal_reward": float(rewards[-1]),
            }
        )
    if {e.body for e in episodes} != set(BODIES):
        raise ValueError("Need successful episodes from both Panda and Sawyer")
    if len({e.prompt for e in episodes}) != 1:
        raise ValueError("This run studies one instruction shared by both bodies")
    index_path = root / "index.json"
    if index_path.is_file():
        verify_collector_index(index_path, records)
    return episodes, records


def verify_collector_index(path, records):
    """Corroborate the tape against demos.py's simulator receipts when supplied."""
    index = json.loads(path.read_text())
    if (index["suite"], index["task_id"], index["case"]) != ("libero_goal", 5, "BASE"):
        raise ValueError("Collector index is not the registered native BASE task")
    accepted = [a for a in index["attempts"] if a["accepted"]]
    by_file = {a["file"]: a for a in accepted}
    if len(by_file) != len(accepted) or set(by_file) != {r["file"] for r in records}:
        raise ValueError("Source tapes disagree with the collector's accepted episodes")
    for record in records:
        attempt = by_file[record["file"]]
        reset = attempt["reset"]
        repro = reset["environment_reproducibility"]
        if (
            not attempt["native_success"]
            or not attempt["done"]
            or reset["model_identity"]["arm"] != record["body"]
            or str(repro["init_state_id"]) != record["pose"]
            or attempt["steps"] != record["frames"]
            or attempt["terminal_reward"] != record["native_terminal_reward"]
            or (record["seed"] is not None and repro["seed"] != record["seed"])
        ):
            raise ValueError(f"{record['file']}: tape conflicts with native reset or success receipt")


def split_poses(episodes, seed):
    rng = np.random.default_rng(seed)
    poses = {b: {e.pose for e in episodes if e.body == b} for b in BODIES}
    if any(len(p) < 2 for p in poses.values()):
        raise ValueError("Need at least two successful distinct poses on each arm for holdout")
    shared = set.intersection(*poses.values())
    if shared:
        held = {str(rng.choice(sorted(shared)))}
    else:
        held = {str(rng.choice(sorted(p))) for p in poses.values()}
    all_poses = set.union(*poses.values())
    goal = max(1, round(0.2 * len(all_poses)))
    for p in rng.permutation(sorted(all_poses - held)):
        if len(held) >= goal:
            break
        candidate = held | {str(p)}
        if all(group - candidate for group in poses.values()):
            held = candidate
    train = [e for e in episodes if e.pose not in held]
    valid = [e for e in episodes if e.pose in held]
    return train, valid, sorted(held)


def select_batch(groups, rng, size):
    batch = []
    for body in BODIES:
        for _ in range(size // len(BODIES)):
            ep = groups[body][rng.integers(len(groups[body]))]
            batch.append(ep.sample(rng.integers(len(ep.actions))))
    order = rng.permutation(size)
    return [batch[i] for i in order]


def valid_anchors(episodes):
    return [
        (e, int(t))
        for e in episodes
        for t in np.linspace(0, len(e.actions) - 1, min(16, len(e.actions)), dtype=int)
    ]
