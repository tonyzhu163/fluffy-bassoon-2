"""Validate accepted native Goal 5 tapes and sample the fixed source partitions."""

from collections import defaultdict
from functools import cache
import hashlib
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryFile

import numpy as np

from motor_data import BODIES, Episode, instruction
from motor_weights import sha256
from pretrain_reader import NATIVE_RESET

TRAIN = range(28)
OFFLINE = range(28, 32)
CAMERA = (256, 256, 3)


def verify_reset(reset, body, pose, seed):
    physics = reset["environment_reproducibility"]
    identity = (reset["model_identity"]["arm"], physics["suite"], physics["task_id"],
                physics["init_state_id"], physics["seed"])
    if identity != (body, "libero_goal", 5, pose, seed):
        raise ValueError(f"Wrong native reset for {body}/{pose}/{seed}: {identity}")
    if any(physics.get(k) != v for k, v in NATIVE_RESET.items()):
        raise ValueError("Release, benchmark manifest or task BDDL differs from native Goal 5")
    plan = physics["perturbation_plan"]
    if (plan["case_id"], plan["seed"]) != ("BASE", seed):
        raise ValueError("Unexpected native perturbation plan")
    return physics


@cache
def verified_archive(path, expected):
    # Older Panda replay indices identify their exact donor by demo number.
    from oracle_fallback import ARCHIVE_SHA
    if expected != ARCHIVE_SHA or sha256(path) != ARCHIVE_SHA:
        raise ValueError("Release archive differs from the collector's verified Goal 5 archive")
    return path


def archive_pose(number, index):
    import h5py
    if not os.environ.get("DUAL_ARCHIVE"):
        raise ValueError("A donor lacks reference_pose; set DUAL_ARCHIVE to the pinned Goal 5 release")
    path = verified_archive(Path(os.environ["DUAL_ARCHIVE"]).resolve(strict=True), index["archive_sha256"])
    with h5py.File(path, "r") as archive:
        row = json.loads(archive[f"data/demo_{number}"].attrs["source_metadata"])
        return int(row["init_state_id"])


def load_indices(roots):
    items, indices, paths = [], [], set()
    for bank, root in enumerate(roots):
        root = root.resolve(strict=True)
        index = root / "index.json"
        raw = index.read_bytes()
        data = json.loads(raw)
        if (data["suite"], data["task_id"], data["case"]) != ("libero_goal", 5, "BASE"):
            raise ValueError(f"{index}: wrong task")
        declared = set(data["bodies"])
        if not declared or declared - set(BODIES):
            raise ValueError(f"{index}: provide a source-only bank")
        indices.append(hashlib.sha256(raw).hexdigest())
        for row in data["attempts"]:
            if row["body"] not in declared or type(row["pose"]) is not int or row["pose"] not in range(32):
                raise ValueError(f"{index}: collector attempt outside source partitions")
            if type(row["accepted"]) is not bool:
                raise ValueError(f"{index}: expected a collector acceptance flag")
            if not row["accepted"]:
                continue
            pose, body, seed = row["pose"], row["body"], row["seed"]
            if not (row["done"] is True and row["native_success"] is True and
                    type(seed) is int and 0 <= seed < 2**32 and type(row["steps"]) is int and
                    0 < row["steps"] <= row["reset"]["max_steps"] and
                    row.get("native_return", row["terminal_reward"]) == row["terminal_reward"]):
                raise ValueError(f"{index}: invalid accepted receipt for {body}/{pose}/{seed}")
            physics = verify_reset(row["reset"], body, pose, seed)
            # A cross-pose reference can also expose a held-out layout during SFT.
            donor_pose = row.get("reference_pose")
            if "reference_demo" in row:
                ref = data.get("references", {}).get(str(row["reference_demo"]), row.get("official_demo"))
                if ref is None and data.get("reference", {}).get("demo") == row["reference_demo"]:
                    ref = data["reference"]
                if ref is not None:
                    if ((donor_pose is not None and donor_pose != ref["pose"])
                            or ref.get("demo", row["reference_demo"]) != row["reference_demo"]):
                        raise ValueError(f"{index}: conflicting donor release poses")
                    donor_pose = ref["pose"]
                elif donor_pose is None:
                    donor_pose = archive_pose(row["reference_demo"], data)
            if donor_pose is not None and donor_pose not in (TRAIN if pose in TRAIN else range(32)):
                raise ValueError(f"{index}: demonstration uses held-out donor {donor_pose}")
            relative = Path(row["file"])
            path = (root / relative).resolve(strict=True)
            if relative.is_absolute() or not path.is_relative_to(root) or path.suffix != ".npz" or path in paths:
                raise ValueError(f"{index}: invalid or repeated archive {relative}")
            paths.add(path)
            record = dict(bank=bank, file=str(relative), body=body, pose=pose, seed=seed,
                          frames=row["steps"], native_terminal_reward=row["terminal_reward"],
                          split="train" if pose in TRAIN else "validation",
                          released_row_fingerprint=physics["released_row_fingerprint"],
                          initial_state_fingerprint=physics["initial_state_fingerprint"],
                          route=row.get("route"), reference_pose=donor_pose, reference_demo=row.get("reference_demo"),
                          repeat=row.get("repeat"), variant=row.get("variant"))
            items.append((record, path, row.get("sha256")))
    return items, indices


def numeric(data, key, shape):
    a = data[key]
    if a.dtype != np.float32 or a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"Invalid {key}: {a.dtype} {a.shape}, expected finite float32 {shape}")
    return a


def load_bank(roots, scratch):
    items, indices = load_indices(roots)
    needed = sum(r["frames"] for r, _, _ in items) * 2 * int(np.prod(CAMERA))
    if shutil.disk_usage(scratch).free < needed:
        raise OSError(f"Insufficient camera staging space: need up to {needed} bytes under {scratch}")
    records, samples, seen = [], [], {}
    # Linux reclaims this unlinked file even if the scheduler kills a worker.
    with TemporaryFile(dir=scratch) as output:
        for record, path, expected_sha in items:
            digest = sha256(path)
            if expected_sha and digest != expected_sha:
                raise ValueError(f"{path}: receipt SHA-256 mismatch")
            record["sha256"] = digest
            n, start = record["frames"], output.tell()
            with np.load(path, allow_pickle=False) as tape:
                for key in ("body", "pose", "seed"):
                    if tape[key].item() != record[key]:
                        raise ValueError(f"{path}: {key} disagrees with native receipt")
                if tape["native_success"].item() is not True:
                    raise ValueError(f"{path}: missing native success")
                state = numeric(tape, "state", (n, 8))
                actions = numeric(tape, "actions", (n, 7))
                numeric(tape, "motion", (n, 7))
                terminal = numeric(tape, "terminal_state", (8,))
                reward = numeric(tape, "rewards", (n,))
                if (not np.allclose(np.linalg.norm(state[:, 3:7], axis=1), 1, atol=1e-5)
                        or not np.isclose(np.linalg.norm(terminal[3:7]), 1, atol=1e-5)
                        or np.abs(actions).max() > 1 or np.any(reward[:-1] != 0)
                        or reward[-1] <= 0 or float(reward[-1]) != record["native_terminal_reward"]):
                    raise ValueError(f"{path}: invalid quaternion, issued action, or native success reward")
                prompt = instruction(tape["prompt"])
                trajectory = hashlib.sha256(prompt.encode())
                for a in (state, actions):
                    trajectory.update(np.ascontiguousarray(a))
                for key in ("rgb", "wrist"):
                    image = tape[key]
                    if image.shape != (n, *CAMERA) or image.dtype != np.uint8:
                        raise ValueError(f"{path}: invalid {key} sequence")
                    image = np.ascontiguousarray(image)
                    trajectory.update(image)
                    image.tofile(output)
                    del image
            record.update(prompt=prompt, trajectory_sha256=trajectory.hexdigest())
            key = (record["body"], record["trajectory_sha256"])
            if key in seen:
                original = records[seen[key]]
                if original["split"] != record["split"]:
                    raise ValueError(f"{path}: duplicate trajectory across train/offline splits")
                record["duplicate_of"] = [original["bank"], original["file"]]
                output.seek(start)
                output.truncate()
            else:
                seen[key] = len(records)
                samples.append((record, start, state, actions))
            records.append(record)
        if not samples or len({r["prompt"] for r in records}) != 1:
            raise ValueError("Need successful episodes on one shared instruction")
        output.flush()
        images = np.memmap(output, dtype=np.uint8, mode="r").reshape(-1, *CAMERA)
    episodes = []
    for r, start, state, actions in samples:
        i, n = start // int(np.prod(CAMERA)), len(actions)
        episodes.append(Episode(r["body"], str(r["pose"]), images[i:i+n], images[i+n:i+2*n],
                                state, actions, r["prompt"]))
    return episodes, records, indices


def coverage(records):
    summary = {}
    layouts = {}
    for r in records:
        fp = r["released_row_fingerprint"]
        if layouts.setdefault(r["pose"], fp) != fp:
            raise ValueError("One released pose ID has different native reset rows")
    if {fp for p, fp in layouts.items() if p in TRAIN} & {fp for p, fp in layouts.items() if p in OFFLINE}:
        raise ValueError("A released reset row occurs on both sides of the offline holdout")
    for b in BODIES:
        summary[b] = {}
        for split in ("train", "validation"):
            rows = [r for r in records if r["body"] == b and r["split"] == split]
            summary[b][split] = dict(accepted=len(rows), poses=sorted({r["pose"] for r in rows}),
                                     released_rows=len({r["released_row_fingerprint"] for r in rows}),
                                     pose_seed_pairs=len({(r["pose"], r["seed"]) for r in rows}),
                                     native_initial_states=len({r["initial_state_fingerprint"] for r in rows}),
                                     distinct_trajectories=sum("duplicate_of" not in r for r in rows))
    return summary


def partition(episodes):
    groups = {"train": {b: defaultdict(list) for b in BODIES},
              "validation": {b: defaultdict(list) for b in BODIES}}
    for e in episodes:
        split = "train" if int(e.pose) in TRAIN else "validation"
        groups[split][e.body][e.pose].append(e)
    for b in BODIES:
        if len(groups["train"][b]) < 2 or not groups["validation"][b]:
            raise ValueError(f"Need distinct successful training layouts and offline validation on {b}")
    return groups["train"], groups["validation"]


def sample(groups, rng, batch_size):
    if batch_size % len(BODIES):
        raise ValueError("Global batch must divide equally across source arms")
    batch = []
    for body in BODIES:
        poses = list(groups[body].values())
        for _ in range(batch_size // len(BODIES)):
            pool = poses[rng.integers(len(poses))]
            ep = pool[rng.integers(len(pool))]
            batch.append(ep.sample(rng.integers(len(ep.actions))))
    return [batch[i] for i in rng.permutation(len(batch))]


def validation_anchors(groups, budget):
    # A fixed number per layout, spread across its episodes and temporal bins.
    result = []
    for body in BODIES:
        for pose in sorted(groups[body], key=int):
            pool = groups[body][pose]
            for i in range(budget):
                j = i * len(pool) // budget if len(pool) > budget else i % len(pool)
                e = pool[j]
                t = int((i + 0.5) / budget * len(e.actions))
                result.append((e, t))
    return result
