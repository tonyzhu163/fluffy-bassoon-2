"""Approved, same-layout native teachers and feedback-based handoff initialization."""

from pathlib import Path

import h5py
import numpy as np
from recover import (
    BODIES,
    PROMPT,
    array_hash,
    check_source,
    digest,
    read,
    reset_identity,
    sha256,
)
from scipy.spatial.transform import Rotation

# isort: split
from oracle_fallback import ARCHIVE_SHA, TrajectoryExpert, panda_reference
from paddle import Paddle, Route


def bank_choices(config):
    available = {}
    for folder, index_sha in zip(config["banks"], config["bank_hashes"], strict=True):
        root = Path(folder)
        index_file = root / "index.json"
        if sha256(index_file) != index_sha:
            raise ValueError(f"Approved bank index changed: {root}")
        index = read(index_file)
        bodies = set(index["bodies"])
        if (
            (index["suite"], index["task_id"], index["case"]) != ("libero_goal", 5, "BASE")
            or not bodies
            or bodies - set(BODIES)
        ):
            raise ValueError(f"Expected source-only Goal 5 bank: {root}")
        for i, row in enumerate(index["attempts"]):
            body, pose = row["body"], row["pose"]
            if type(pose) is not int or pose not in range(28) or row["accepted"] is not True:
                continue
            check_source(body, pose)
            if (
                body not in bodies
                or row["native_success"] is not True
                or row["done"] is not True
                or not np.isfinite(row["terminal_reward"])
                or row["terminal_reward"] <= 0
            ):
                raise ValueError(f"Invalid accepted receipt {i} in {index_file}")
            reset_identity(row["reset"], body, pose, row["seed"])
            if row.get("reference_pose", pose) != pose:
                continue
            if row.get("route") in ("whole_gripper_push", "rendered_commands") and "params" in row:
                teacher = "paddle"
            elif body == "Panda":
                teacher = "official_tracker"
            else:
                continue
            available.setdefault((body, pose), []).append((root, index_sha, i, row, teacher))
    return available


def pick(config, available):
    if config["rolls"]:
        keys = [tuple(key) for key in config["rolls"]]
    else:
        # Alternate arms, in layout order within each arm.
        ordered = [sorted(k for k in available if k[0] == body) for body in BODIES]
        keys = [group[i] for i in range(max(map(len, ordered))) for group in ordered if i < len(group)]
        keys = keys[: config["count"]]
    if (
        len(keys) != config["count"]
        or not 1 <= len(keys) <= 2 * 28
        or len(set(keys)) != len(keys)
        or any(key not in available for key in keys)
    ):
        raise ValueError(
            f"Need the requested number of distinct supported source starts. Eligible body/layouts: {sorted(available)}"
        )
    return keys


def selected_source(entry):
    root, index_sha, number, row, teacher = entry
    path = (root / row["file"]).resolve(strict=True)
    if not path.is_relative_to(root) or Path(row["file"]).is_absolute() or path.suffix != ".npz":
        raise ValueError("Accepted tape is outside approved bank")
    tape_sha = sha256(path)
    if row.get("sha256", tape_sha) != tape_sha:
        raise ValueError(f"Accepted tape digest disagrees with bank: {path}")
    body, pose, seed = row["body"], row["pose"], row["seed"]
    with np.load(path, allow_pickle=False) as tape:
        if (
            any(tape[k].item() != v for k, v in (("body", body), ("pose", pose), ("seed", seed)))
            or tape["native_success"].item() is not True
            or tape["prompt"].item() != PROMPT
        ):
            raise ValueError("Native bank tape identity mismatch")
        rewards, commands, states = tape["rewards"], tape["actions"], tape["state"]
        n = row["steps"]
        if (
            type(n) is not int
            or not 1 <= n <= row["reset"]["max_steps"]
            or rewards.shape != (n,)
            or np.any(rewards[:-1] != 0)
            or rewards[-1] != row["terminal_reward"]
            or commands.shape != (n, 7)
            or not np.isfinite(commands).all()
            or np.any(abs(commands) > 1)
            or states.shape != (n, 8)
            or not np.isfinite(states).all()
            or states.dtype != np.float32
            or commands.dtype != np.float32
        ):
            raise ValueError("Invalid accepted native trajectory")
    metadata = {
        "body": body,
        "pose": pose,
        "seed": seed,
        "teacher": teacher,
        "bank": str(root),
        "index_sha256": index_sha,
        "attempt": number,
        "receipt_sha256": digest(row),
        "tape": row["file"],
        "tape_sha256": tape_sha,
    }
    return metadata, row, states, commands


def official_reference(archive, row):
    # Check donor layout before panda_reference can load actions or reset a simulator.
    import json

    pose = row["pose"]
    check_source("Panda", pose)
    with h5py.File(archive, "r") as f:
        matches = []
        for key, group in f["data"].items():
            src = json.loads(group.attrs["source_metadata"])
            if int(src["init_state_id"]) != pose or src["instruction"] != PROMPT:
                continue
            if group.attrs["success"] and group["rewards"][-1] > 0 and group["dones"][-1]:
                matches.append(int(key.removeprefix("demo_")))
    if not matches:
        return None
    number = row.get("reference_demo")
    if number not in matches:
        number = min(matches)
    # This replay owns its own Session. It never modifies the learner's episode.
    ref = panda_reference(archive, number)
    evidence = ref["evidence"]
    lineage = row["reset"]["environment_reproducibility"]
    if (
        evidence["pose"] != pose
        or not evidence["native_success"]
        or evidence["native_reward"] <= 0
        or evidence["reset"]["released_row_fingerprint"] != lineage["released_row_fingerprint"]
    ):
        raise ValueError("Official reference and target disagree about released layout or success")
    evidence.update(
        archive_sha256=ARCHIVE_SHA,
        state_sha256=array_hash(ref["states"]),
        plate_sha256=array_hash(ref["plates"]),
        action_sha256=array_hash(ref["actions"]),
    )
    return ref


def tracker_handoff(ref, initial_scene, obs, scene):
    expert = TrajectoryExpert(ref, initial_scene)
    # Match pre-action waypoints using the same reset offset as TrajectoryExpert.
    # Compare both plate progress and wrist pose so revisiting a wrist location
    # after contact does not silently select the return leg.
    states, plates = ref["states"][:-1], ref["plates"][:-1, :2]
    progress = np.clip(((plates - plates[0]) @ expert.travel) / expert.travel_sq, 0, 1)
    offsets = (1 - progress[:, None]) * expert.initial_offset
    plate_error = np.asarray(scene["plate"][:2]) - (plates + offsets)
    target = states[:, :3].copy()
    target[:, :2] += offsets + np.clip(plate_error, -0.015, 0.015)
    xyz = np.linalg.norm(target - obs["state"][:3], axis=1)
    angle = (Rotation.from_quat(states[:, 3:7]) * Rotation.from_quat(obs["state"][3:7]).inv()).magnitude()
    jaw = abs(states[:, 7] - obs["state"][7])
    plate = np.linalg.norm(plate_error, axis=1)
    cost = (xyz / 0.030) ** 2 + (angle / 0.17) ** 2 + (plate / 0.030) ** 2 + (jaw / 0.04) ** 2
    expert.i = i = int(cost.argmin())
    info = {
        "waypoint": i,
        "waypoints": len(cost),
        "xyz_error_m": float(xyz[i]),
        "angle_error_rad": float(angle[i]),
        "plate_error_m": float(plate[i]),
        "jaw_error_m": float(jaw[i]),
        "match_cost": float(cost[i]),
    }

    def command(current, live, geometry):
        a = expert.action(current, live)
        return a, expert.trace[-1]

    return command, info


def paddle_handoff(params, scene, geometry):
    route = Route(**params)
    if (
        route.mode not in ("rim", "hook", "press")
        or not np.isfinite([route.tilt, route.yaw, route.flank, route.bottom]).all()
    ):
        raise ValueError("Invalid Paddle route parameters in approved receipt")
    # Evaluate geometry for alignment and engagement without issuing a command
    # or carrying a probe's timers or integral into the controller.
    diagnostics = []
    for phase in (0, 2):
        probe = Paddle(route, scene)
        probe.phase = phase
        probe.action(geometry, scene)
        diagnostics.append(probe.last)
    approach, push = diagnostics
    aligned = np.linalg.norm(approach["error"][:2]) < 0.030 and approach["angle"] < 0.32
    low = approach["low"] - scene["table_z"]
    at_surface = low <= route.bottom + 0.007
    near_push = np.linalg.norm(push["error"][:2]) < 0.030
    phase = 0
    if aligned and low <= 0.108 + 0.023:
        phase = 1
    if at_surface and approach["angle"] < 0.32 and (aligned or near_push):
        phase = 2
    expert = Paddle(route, scene)
    expert.phase = phase

    def command(current, live, live_geometry):
        a = expert.action(live_geometry, live)
        return a, expert.last

    return command, {"phase": phase, "approach": approach, "push": push, "contact": scene["contact"]}
