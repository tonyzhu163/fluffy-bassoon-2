"""Released Goal 5 Paddle controls. CLI: python phase_data.py --banks BANK [BANK ...].

iter_episodes(banks, *, split, route_scope="primary", scratch=None) yields unique
PhaseEpisodes. split="fit" uses 0..23; "holdout" uses 24..27. Camera pairs are
backed by temporary files under scratch (or TMPDIR) for the episode's lifetime.
At 20 Hz state[t] and both cameras precede the issued actions[t]. No terminal
frame, next action or teacher quantity is exposed through observation(t).
"""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryFile

import numpy as np

from generated_split import BANK_SHA
from recover import BODIES, PROMPT, reset_identity, sha256


@dataclass
class PhaseEpisode:
    body: str
    pose: int
    state: np.ndarray
    actions: np.ndarray
    plate: np.ndarray
    error: np.ndarray
    phase: np.ndarray
    switch: np.ndarray
    rgb: np.ndarray
    wrist: np.ndarray
    seed: int
    route: str
    trajectory_sha256: str
    hand_target: np.ndarray

    def observation(self, t):
        if not 0 <= t < len(self.actions):
            raise IndexError(t)
        return dict(rgb=self.rgb[t], wrist=self.wrist[t], state=self.state[t], prompt=PROMPT)


def _route(row):
    p = row.get("params")
    if not isinstance(p, dict):
        return "other"
    primary = dict(mode="rim", tilt=28, yaw=180, flank=-35, bottom=.006)
    if p == primary:
        return "primary"
    if p == dict(primary, flank=0):
        return "straight"
    return "other"


def _generated(obj):
    return ("source_kind" in obj or "layout_origin" in obj or "generated_row_fingerprint" in obj
            or obj.get("source_constructor") == "native_generated_layout")


def _items(banks, splits, scope, counts):
    if scope not in ("primary", "straight") or not set(splits) <= {"fit", "holdout"}:
        raise ValueError("Expected primary|straight and fit|holdout")
    if isinstance(banks, (str, Path)):
        banks = [banks]
    items = []
    fingerprints, poses = {}, {}
    for bank in banks:
        root = Path(bank).resolve(strict=True)
        idx = json.loads((root / "index.json").read_bytes())
        for row in idx["attempts"]:
            body, pose = row.get("body", "unknown"), row.get("pose")
            released = type(pose) is int and 0 <= pose < 28
            split = ("fit" if pose < 24 else "holdout") if released else "excluded"
            route = _route(row)
            stats = counts[split, body, route]
            stats["attempts"] += 1
            if row.get("accepted") is not True:
                continue
            stats["accepted"] += 1
            if (not released or body not in BODIES or route == "other"
                    or route != scope
                    or _generated(idx) or _generated(row)
                    or "capture" in idx or "label_schema" in idx or "waypoint_provenance" in row
                    or (idx.get("suite"), idx.get("task_id"), idx.get("case")) != ("libero_goal", 5, "BASE")
                    or any(row.get(k) is not None for k in ("reference_pose", "reference_demo", "official_demo"))
                    or row.get("route", "whole_gripper_push") not in ("rendered_commands", "whole_gripper_push")):
                continue
            if split not in splits:
                continue
            reset = row["reset"]
            physics = reset["environment_reproducibility"]
            seed = row["seed"]
            if _generated(physics) or _generated(reset) or seed in range(740000, 750000) or seed in range(840000, 850000):
                continue
            reset_identity(reset, body, pose, seed)
            renderer = reset.get("renderer", [])
            if (body not in idx["bodies"]
                    or row["done"] is not True or row["native_success"] is not True
                    or type(row["steps"]) is not int or not 0 < row["steps"] <= reset["max_steps"]
                    or not row["terminal_reward"] > 0
                    or len(renderer) != 2 or "NVIDIA" not in renderer[0].upper() or not renderer[1]):
                raise ValueError(f"Invalid positive released camera receipt: {root}/{row.get('file')}")
            fp = physics["released_row_fingerprint"]
            if (not fp or fingerprints.setdefault(pose, fp) != fp or poses.setdefault(fp, pose) != pose):
                raise ValueError("Released layout ID and fingerprint do not agree across receipts")
            bank_receipt = idx.get("input_receipts", {}).get("released_bank")
            if bank_receipt is not None and (bank_receipt["sha256"] != BANK_SHA
                                              or bank_receipt["rows"][str(pose)] != fp):
                raise ValueError("Released reset differs from capture input bank")
            dense = "phase_schema" in idx
            if dense != ("phase_file" in row) or (dense and idx["phase_schema"] != 1):
                raise ValueError("Expected phase_schema=1 and a phase_file on every dense positive")
            stats["selected"] += 1
            items.append((root, row, split, route, dense))
    # A sparse copy of a dense tape must never hide the complete labels.
    return sorted(items, key=lambda i: (not i[4], i[1]["body"], i[1]["pose"], i[1]["seed"],
                                        str(i[0]), i[1]["file"]))


def _path(root, name):
    path = (root / name).resolve(strict=True)
    if Path(name).is_absolute() or not path.is_relative_to(root) or path.suffix != ".npz":
        raise ValueError(f"Expected relative bank NPZ: {name}")
    return path


def _array(data, key, shape):
    a = data[key]
    if a.shape != shape or a.dtype != np.float32 or not np.isfinite(a).all():
        raise ValueError(f"Invalid float32 {key}: {a.shape} {a.dtype}")
    return a


def _dense(path, expected_digest, tape_digest, state, actions):
    if expected_digest is not None and sha256(path) != expected_digest:
        raise ValueError(f"Phase file digest differs: {path}")
    n = len(actions)
    with np.load(path, allow_pickle=False) as d:
        if (d["tape_sha256"].item() != tape_digest or not np.array_equal(d["pre_state"], state)
                or not np.array_equal(d["action"], actions)):
            raise ValueError(f"Dense labels differ from their camera tape: {path}")
        plate, error = d["plate_xy"], d["error"]
        for a, width in ((plate, 2), (error, 3)):
            if a.shape != (n, width) or a.dtype not in (np.float32, np.float64) or not np.isfinite(a).all():
                raise ValueError(f"Invalid dense geometry: {path}")
        phase, switch = d["phase"], d["switch"]
        if (phase.shape != (n,) or switch.shape != (n,) or phase.dtype.kind not in "iu"
                or switch.dtype.kind not in "biu" or not np.isin(phase, [0, 1, 2]).all()
                or not np.isin(switch, [0, 1]).all() or phase[0] != 0
                or np.any(phase + switch > 2)
                or not np.array_equal(phase[1:], phase[:-1] + switch[:-1])):
            raise ValueError(f"Invalid pre-action phase/switch sequence: {path}")
    return plate.astype(np.float32), error.astype(np.float32), phase.astype(np.int8), switch.astype(np.int8)


def _sparse(row):
    n = row["steps"]
    plate = np.full((n, 2), np.nan, np.float32)
    error = np.full((n, 3), np.nan, np.float32)
    phase = np.full(n, -1, np.int8)
    switch = phase.copy()
    # Paddle.last stores old-phase geometry but POST-transition phase. At a
    # sampled change across a gap we cannot place the switch on a specific tick.
    samples = dict()
    for sample in row.get("trace", []):
        t = sample["t"]
        if type(t) is not int or t not in range(n) or t in samples:
            raise ValueError("Invalid sparse trace tick")
        samples[t] = {k: v for k, v in sample.items() if k != "t"}
    # The collector also saves its diagnostic from the final issued action;
    # last has no explicit t, but precedes the terminal observation.
    if row.get("last"):
        if n - 1 in samples and samples[n-1] != row["last"]:
            raise ValueError("Conflicting terminal-action trace")
        samples[n-1] = row["last"]
    prev_t, prev_post = -1, 0
    for t, sample in sorted(samples.items()):
        for key, dest in (("plate", plate), ("error", error)):
            if key in sample:
                a = np.asarray(sample[key], dtype=np.float32)
                if a.shape != dest.shape[1:] or not np.isfinite(a).all():
                    raise ValueError(f"Invalid sparse {key} at {t}")
                dest[t] = a
        post = sample.get("phase")
        if post is None:
            continue
        if type(post) is not int or not prev_post <= post <= min(2, prev_post + t - prev_t):
            raise ValueError(f"Impossible Paddle transition at {t}")
        # The pre-phase immediately after a sampled post-phase is certain.
        phase[prev_t + 1] = prev_post
        if prev_post == post:
            phase[prev_t + 1:t + 1] = post
            switch[prev_t + 1:t + 1] = 0
        elif t == prev_t + 1:
            phase[t], switch[t] = prev_post, post - prev_post
        prev_t, prev_post = t, post
    if prev_t + 1 < n:
        phase[prev_t + 1] = prev_post
    if prev_post == 2:
        phase[prev_t + 1:] = 2
        switch[prev_t + 1:] = 0
    return plate, error, phase, switch


def _episode(root, row, route, dense, scratch):
    path = _path(root, row["file"])
    tape_digest = sha256(path)
    if row.get("sha256", tape_digest) != tape_digest:
        raise ValueError(f"Tape digest differs: {path}")
    n = row["steps"]
    with np.load(path, allow_pickle=False) as d:
        if (any(d[k].item() != row[k] for k in ("body", "pose", "seed"))
                or d["native_success"].item() is not True or d["prompt"].item() != PROMPT):
            raise ValueError(f"Tape identity differs: {path}")
        state = _array(d, "state", (n, 8))
        actions = _array(d, "actions", (n, 7))
        terminal = _array(d, "terminal_state", (8,))
        _array(d, "motion", (n, 7))
        reward = _array(d, "rewards", (n,))
        if (np.any(abs(actions) > 1) or np.any(reward[:-1]) or reward[-1] != row["terminal_reward"]
                or not np.allclose(np.linalg.norm(state[:, 3:7], axis=1), 1, atol=1e-5)
                or not np.isclose(np.linalg.norm(terminal[3:7]), 1, atol=1e-5)):
            raise ValueError(f"Invalid native control tape: {path}")
        labels = (_dense(_path(root, row["phase_file"]), row.get("phase_sha256"), tape_digest, state, actions)
                  if dense else _sparse(row))
        digest = hashlib.sha256(PROMPT.encode())
        for a in (state, actions):
            digest.update(np.ascontiguousarray(a))
        shape = (n, 256, 256, 3)
        with TemporaryFile(dir=scratch) as f:
            for key in ("rgb", "wrist"):
                image = d[key]
                if image.shape != shape or image.dtype != np.uint8:
                    raise ValueError(f"Invalid {key}: {path}")
                digest.update(np.ascontiguousarray(image))
                image.tofile(f)
                del image
            f.flush()
            cameras = np.memmap(f, mode="r", dtype=np.uint8, shape=(2, *shape))
    # Direct capture knows the rigid hand/site transform. For legacy command
    # replays use the observed hand at closest approach in *orientation*,
    # measured by the shadow teacher on these actual camera states.
    if dense:
        hand_target = np.asarray(row["desired_hand_xyzw"], dtype=np.float32)
    else:
        points = [(o["t"], o["angle"]) for o in row.get("trace", []) if "angle" in o]
        if "angle" in row.get("last", {}):
            points.append((n - 1, row["last"]["angle"]))
        hand_target = (state[min(points, key=lambda p: p[1])[0], 3:7].copy()
                       if points else np.full(4, np.nan, np.float32))
    if dense and (hand_target.shape != (4,) or not np.isclose(np.linalg.norm(hand_target), 1., atol=1e-5)):
        raise ValueError("Invalid dense hand orientation target")
    return PhaseEpisode(row["body"], row["pose"], state, actions, *labels, *cameras,
                        row["seed"], route, digest.hexdigest(), hand_target)


def _stream(items, scratch, counts):
    seen = {}
    for root, row, split, route, dense in items:
        ep = _episode(root, row, route, dense, scratch)
        key = ep.body, ep.trajectory_sha256
        labels = hashlib.sha256()
        for a in (ep.plate, ep.error, ep.phase, ep.switch, ep.hand_target):
            labels.update(np.ascontiguousarray(a))
        identity = (ep.pose, route)
        if key in seen:
            previous, was_dense, digest = seen[key]
            if previous != identity or (was_dense == dense and digest != labels.digest()):
                raise ValueError("Conflicting duplicate trajectory identity or teacher labels")
            continue
        seen[key] = (identity, dense, labels.digest())
        c = counts[split, ep.body, route]
        c["unique_trajectories"] += 1
        c["actions"] += len(ep.actions)
        c["dense_episodes"] += int(dense)
        for name in ("plate", "error"):
            c[name + "_labels"] += int(np.isfinite(getattr(ep, name)).all(axis=1).sum())
        for name in ("phase", "switch"):
            c[name + "_labels"] += int((getattr(ep, name) >= 0).sum())
        yield ep


def iter_episodes(banks, *, split, route_scope="primary", scratch=None):
    """Select by receipt before opening any NPZ. Returns all T issued actions."""
    counts = defaultdict(Counter)
    yield from _stream(_items(banks, [split], route_scope, counts), scratch, counts)


def inventory(banks, *, route_scope="primary", scratch=None):
    counts = defaultdict(Counter)
    layouts = defaultdict(set)
    resets = defaultdict(set)
    commands = defaultdict(set)
    initial_states = defaultdict(set)
    items = _items(banks, ["fit", "holdout"], route_scope, counts)
    for _, row, split, route, _ in items:
        initial_states[split, row["body"], route].add(row["reset"]["environment_reproducibility"]["initial_state_fingerprint"])
    for ep in _stream(items, scratch, counts):
        key = ("fit" if ep.pose < 24 else "holdout", ep.body, ep.route)
        layouts[key].add(ep.pose)
        resets[key].add((ep.pose, ep.seed))
        commands[key].add(hashlib.sha256(np.ascontiguousarray(ep.actions)).digest())
    fields = ("attempts", "accepted", "selected", "unique_trajectories", "actions", "dense_episodes",
              "plate_labels", "error_labels", "phase_labels", "switch_labels")
    return [dict(split=k[0], body=k[1], route=k[2], **{f: c[f] for f in fields}, poses=sorted(layouts[k]),
                 unique_action_sequences=len(commands[k]), unique_initial_states=len(initial_states[k]),
                 unique_pose_seeds=len(resets[k]), duplicates=c["selected"]-c["unique_trajectories"])
            for k, c in sorted(counts.items())]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--banks", nargs="+", required=True, type=Path)
    parser.add_argument("--route-scope", choices=("primary", "straight"), default="primary")
    parser.add_argument("--scratch", type=Path)
    print(json.dumps(inventory(**vars(parser.parse_args())), indent=2))
