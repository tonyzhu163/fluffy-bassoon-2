"""Validate accepted native Goal 5 tapes and sample the fixed source partitions."""

from collections import defaultdict
from functools import cache
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from tempfile import TemporaryFile

import numpy as np

import generated_split as generated
from motor_data import BODIES, Episode, instruction
from motor_weights import sha256
from pretrain_reader import NATIVE_RESET

TRAIN = range(28)
OFFLINE = range(28, 32)
GENERATED_TRAIN_SEEDS = range(300000, 400000)
RESERVED_SEEDS = (range(740000, 750000), range(840000, 850000))
CAMERA = (256, 256, 3)


def source_split(pose):
    """Return the registered partition of an integer source group."""
    if type(pose) is int:
        if pose in TRAIN or pose - generated.GROUP_BASE in GENERATED_TRAIN_SEEDS:
            return "train"
        if pose in OFFLINE:
            return "validation"
    raise ValueError(f"Unregistered source layout {pose}")


def source_fingerprint(row):
    """Return the raw Panda row fingerprint shared by the two source arms."""
    if row.get("source_kind") == generated.SOURCE_KIND:
        return row["generated_row_fingerprint"]
    return row["released_row_fingerprint"]


def validate_source_records(records, public_row_fingerprints=None):
    """Recheck saved source groups and holdout separation without opening cameras."""
    layouts, generated_rows = {}, []
    for row in records:
        pose, seed = row["pose"], row["seed"]
        if (row["body"] not in BODIES or row["split"] != source_split(pose)
                or type(seed) is not int or not 0 <= seed < 2**32
                or any(seed in r for r in RESERVED_SEEDS)):
            raise ValueError("Unregistered source arm, split or seed")
        if pose >= generated.GROUP_BASE:
            generated_rows.append(row)
            if (type(row["frames"]) is not int or not 0 < row["frames"] <= 300
                    or row["native_terminal_reward"] != 1 or row["route"] != "rendered_commands"
                    or row.get("reference_pose") is not None or row.get("reference_demo") is not None
                    or len(row["renderer"]) != 2 or "NVIDIA" not in row["renderer"][0].upper()
                    or not row["renderer"][1]):
                raise ValueError("Generated source record lacks a positive camera replay")
            public = row.get("public_row_fingerprints", public_row_fingerprints)
            if public is None or (public_row_fingerprints is not None and tuple(public) != tuple(public_row_fingerprints)):
                raise ValueError("Missing or inconsistent public row fingerprints")
            public_row_fingerprints = public
        else:
            if any(k in row for k in ("layout_origin", "source_kind", "generated_row_fingerprint")):
                raise ValueError("Released source record contains generated provenance")
            fp = source_fingerprint(row)
            if layouts.setdefault(pose, fp) != fp:
                raise ValueError("One released pose ID has different native reset rows")
    if {fp for p, fp in layouts.items() if p in TRAIN} & {fp for p, fp in layouts.items() if p in OFFLINE}:
        raise ValueError("A released reset row occurs on both sides of the offline holdout")
    if generated_rows:
        generated.validate_generated_groups(generated_rows, public_row_fingerprints, native_reset=NATIVE_RESET)
        if any(fp != public_row_fingerprints[p] for p, fp in layouts.items()):
            raise ValueError("Released source identity disagrees with the pinned public bank order")


def bank_path(root, name, suffix):
    relative = Path(name)
    path = (root / relative).resolve(strict=True)
    if relative.is_absolute() or not path.is_relative_to(root) or path.suffix != suffix or not path.is_file():
        raise ValueError(f"Invalid bank artifact {name}")
    return path


def generated_items(root, data, bank, paths):
    generated.expect(data, dict(**generated.TASK, split="train", source_kind=generated.SOURCE_KIND,
                                source_constructor="native_generated_layout", layout_bank_row=0))
    start, budget = data["seed_base"], data["max_seeds"]
    if (type(start) is not int or type(budget) is not int or type(data["requested_layouts"]) is not int
            or not 1 <= data["requested_layouts"] <= budget or len(data["bodies"]) != len(set(data["bodies"]))
            or start not in GENERATED_TRAIN_SEEDS or start + budget - 1 not in GENERATED_TRAIN_SEEDS):
        raise ValueError("Generated bank seed budget is outside the preregistered training interval")
    public = generated.public_denylist(data["public_row_fingerprints"])
    layouts = {}
    layout_hashes = set()
    for layout in data["layouts"]:
        seed = layout["layout_seed"]
        if seed in layouts or seed not in range(start, start + budget) or layout["row_fingerprint"] in public:
            raise ValueError("Duplicate, public or unregistered generated layout")
        bank_path(root, layout["state_file"], ".npy")
        generated.validate_layout_file(root, layout)
        generated.expect(layout, NATIVE_RESET)
        for key in ("initial_state_sha256", "scene_hash", "movable_xy_hash"):
            identity = (key, layout[key])
            if identity in layout_hashes:
                raise ValueError(f"Duplicate generated layout {key} in bank")
            layout_hashes.add(identity)
        layouts[seed] = layout
    rejected = [r["layout_seed"] for r in data["rejected"]]
    seeds = sorted([*layouts, *rejected])
    if (any(type(seed) is not int for seed in rejected)
            or seeds != list(range(start, start + len(seeds))) or len(seeds) > budget
            or len(layouts) > data["requested_layouts"]
            or (len(layouts) < data["requested_layouts"] and len(seeds) != budget)
            or data["poses"] != [layout["source_index"] for layout in layouts.values()]):
        raise ValueError("Generated seed ledger and declared groups disagree")
    searches, items, positives = {}, [], {b: set() for b in BODIES}
    for row in data["attempts"]:
        origin = generated.source_record(row, NATIVE_RESET)
        body, pose, seed = row["body"], row["pose"], row["seed"]
        if (body not in data["bodies"] or type(row["accepted"]) is not bool or row["layout_seed"] != seed
                or type(row["camera"]) is not bool or seed not in layouts):
            raise ValueError("Invalid generated collector attempt")
        generated.expect(layouts[seed], origin)
        key = (body, seed, row["recipe"])
        if row["search_success"] is True:
            generated.expect(row, dict(camera=False, accepted=False, done=True, native_success=True,
                                       outcome="search_success", route="whole_gripper_push"))
            if key in searches:
                raise ValueError("Duplicate successful search receipt")
            searches[key] = row
        if not row["accepted"]:
            continue
        fields = generated.validate_generated_attempt(data, row, native_reset=NATIVE_RESET)
        generated.expect(row, dict(outcome="accepted", route="rendered_commands", search_success=False))
        if any(k in row for k in ("reference_pose", "reference_demo", "official_demo")):
            raise ValueError("Generated command replay declares a released donor")
        if (key not in searches or pose in positives[body] or type(row["steps"]) is not int
                or row["steps"] > data["max_steps"]):
            raise ValueError("Generated replay lacks a unique successful search")
        search = searches[key]
        generated.expect(search, {k: row[k] for k in ("steps", "native_return", "terminal_reward",
                                                       "search_file", "search_sha256", "params")})
        search_reset = verify_reset(search["reset"], body, pose, seed)
        generated.expect(search_reset, dict(initial_state_fingerprint=fields["initial_state_fingerprint"],
                                            generated_row_fingerprint=fields["generated_row_fingerprint"],
                                            scene_fingerprint=fields["scene_fingerprint"]))
        draft = bank_path(root, row["search_file"], ".npy")
        path = bank_path(root, row["file"], ".npz")
        if path in paths or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) or sha256(draft) != row["search_sha256"]:
            raise ValueError("Repeated generated tape or invalid search/tape digest")
        paths.add(path)
        commands = np.load(draft, allow_pickle=False)
        with np.load(path, allow_pickle=False) as tape:
            generated.validate_generated_tape(tape, row)
            if commands.dtype != np.float32 or not np.array_equal(commands, tape["actions"]):
                raise ValueError("Generated camera actions differ from saved positive search commands")
        positives[body].add(pose)
        record = dict(**fields, bank=bank, file=str(Path(row["file"])), frames=row["steps"],
                      native_terminal_reward=row["terminal_reward"], renderer=row["reset"]["renderer"],
                      route=row["route"], reference_pose=None, reference_demo=None, repeat=row["repeat"],
                      variant=None, layout_seed=seed, public_row_fingerprints=data["public_row_fingerprints"],
                      search_file=row["search_file"], search_sha256=row["search_sha256"],
                      state_file=layouts[seed]["state_file"], state_sha256=layouts[seed]["state_sha256"])
        items.append((record, path, row["sha256"]))
    counts = {b: len(positives[b]) for b in data["bodies"]}
    complete = len(layouts) == data["requested_layouts"] and all(counts.values())
    generated.expect(data, dict(status="complete" if complete else "shortfall", accepted=len(items),
                                admitted_layouts=len(layouts), accepted_by_body=counts,
                                paired_positive_groups=sorted(positives["Panda"] & positives["Sawyer"])))
    return items


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
        index = bank_path(root, "index.json", ".json")
        raw = index.read_bytes()
        data = json.loads(raw)
        if (data["suite"], data["task_id"], data["case"]) != ("libero_goal", 5, "BASE"):
            raise ValueError(f"{index}: wrong task")
        declared = set(data["bodies"])
        if not declared or declared - set(BODIES):
            raise ValueError(f"{index}: provide a source-only bank")
        indices.append(hashlib.sha256(raw).hexdigest())
        if data.get("source_kind") == generated.SOURCE_KIND:
            items.extend(generated_items(root, data, bank, paths))
            continue
        if "source_kind" in data or data.get("source_constructor") == "native_generated_layout":
            raise ValueError(f"{index}: unknown or inconsistent generated bank type")
        if "capture" in data:
            accepted = [r for r in data["attempts"] if r["accepted"]]
            if (data["status"] != "complete" or data["accepted"] != len(accepted)
                    or data["labels"] != sum(r["steps"] for r in accepted)):
                raise ValueError(f"{index}: incomplete or inconsistent recovery bank")
        for row in data["attempts"]:
            if row["body"] not in declared or type(row["pose"]) is not int or row["pose"] not in range(32):
                raise ValueError(f"{index}: collector attempt outside source partitions")
            if type(row["accepted"]) is not bool:
                raise ValueError(f"{index}: expected a collector acceptance flag")
            if (any(row.get("seed") in r for r in RESERVED_SEEDS)
                    or any(k in row for k in ("source_kind", "layout_origin", "generated_row_fingerprint"))):
                raise ValueError(f"{index}: released collector used a reserved seed or generated row")
            if not row["accepted"]:
                continue
            pose, body, seed = row["pose"], row["body"], row["seed"]
            if not (row["done"] is True and row["native_success"] is True and
                    type(seed) is int and 0 <= seed < 2**32 and type(row["steps"]) is int and
                    0 < row["steps"] <= row["reset"]["max_steps"] and
                    row.get("native_return", row["terminal_reward"]) == row["terminal_reward"]):
                raise ValueError(f"{index}: invalid accepted receipt for {body}/{pose}/{seed}")
            physics = verify_reset(row["reset"], body, pose, seed)
            if "generated_row_fingerprint" in physics or "source_kind" in physics:
                raise ValueError(f"{index}: generated reset in a released bank")
            renderer = row["reset"].get("renderer", [])
            if len(renderer) != 2 or "NVIDIA" not in renderer[0].upper() or not renderer[1]:
                raise ValueError(f"{index}: successful camera trajectory lacks allocated NVIDIA rendering evidence")
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
            if "capture" in data:
                # recover_sim saves ordinary motor NPZs containing only post-handoff controls.
                prefix = row["prefix_steps_requested"]
                if (pose not in TRAIN or row["outcome"] != "recovered" or prefix != row["learner_steps"]
                        or row["total_steps"] != prefix + row["steps"] or row["total_steps"] > row["reset"]["max_steps"]
                        or any(row["source"][key] != row[key] for key in ("body", "pose", "seed"))
                        or not row.get("sha256")):
                    raise ValueError(f"{index}: inconsistent source recovery suffix")
            record = dict(bank=bank, file=str(relative), body=body, pose=pose, seed=seed,
                          frames=row["steps"], native_terminal_reward=row["terminal_reward"], renderer=renderer,
                          split="train" if pose in TRAIN else "validation",
                          released_row_fingerprint=physics["released_row_fingerprint"],
                          initial_state_fingerprint=physics["initial_state_fingerprint"],
                          route=row.get("route"), reference_pose=donor_pose, reference_demo=row.get("reference_demo"),
                          repeat=row.get("repeat"), variant=row.get("variant"))
            items.append((record, path, row.get("sha256")))
    validate_source_records(r for r, _, _ in items)
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
                if record.get("source_kind") == generated.SOURCE_KIND and prompt != "push the plate to the front of the stove":
                    raise ValueError(f"{path}: generated tape has the wrong Goal 5 instruction")
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
                if ((record.get("source_kind") or original.get("source_kind"))
                        and source_fingerprint(record) != source_fingerprint(original)):
                    raise ValueError(f"{path}: duplicate trajectory attributed to different source layouts")
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
    validate_source_records(records)
    summary = {}
    mixed = any(r.get("source_kind") == generated.SOURCE_KIND for r in records)
    for b in BODIES:
        summary[b] = {}
        for split in ("train", "validation"):
            rows = [r for r in records if r["body"] == b and r["split"] == split]
            summary[b][split] = dict(accepted=len(rows), poses=sorted({r["pose"] for r in rows}),
                                     released_rows=len({r["released_row_fingerprint"] for r in rows
                                                        if "released_row_fingerprint" in r}),
                                     pose_seed_pairs=len({(r["pose"], r["seed"]) for r in rows}),
                                     native_initial_states=len({r["initial_state_fingerprint"] for r in rows}),
                                     distinct_trajectories=sum("duplicate_of" not in r for r in rows))
            if mixed:
                drawn = [r for r in rows if r.get("source_kind") == generated.SOURCE_KIND]
                summary[b][split].update(generated_rows=len({source_fingerprint(r) for r in drawn}),
                                         generated_groups=len({r["pose"] for r in drawn}),
                                         generated_accepted=len(drawn))
    return summary


def partition(episodes):
    groups = {"train": {b: defaultdict(list) for b in BODIES},
              "validation": {b: defaultdict(list) for b in BODIES}}
    for e in episodes:
        split = source_split(int(e.pose))
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
