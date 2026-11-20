"""8-GPU sharded collector: 1,344 7-DoF Panda/Sawyer train (24 tasks x 4 starts) + 96 held-out 6-DoF UR5e dev."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

# Reuse verified UR5e multi-arm simulator and helpers from Phase 3 baseline.
BASE_DIR = Path(__file__).resolve().parents[1] / "libero-xemb-ur5e-sysid-20261120-r1"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fluffy_bassoon.xemb.embodiment import ACTIVE_PROBE_BURSTS, EVAL_EMBODIMENTS, TRAIN_EMBODIMENTS
from collect_ur5e import (
    CHUNK_SIZE,
    DEV_ARM,
    MODE_PATH,
    SUPPORT_BUDGET,
    TRAIN_ARMS,
    chunk_counts,
    frame0_identity,
    run_active_probe_episode,
)
from ur5e_motor import SPLITS_PATH, load_frozen_pi05, run_episode, write_json
from ur5e_sim_worker import UR5eSimulator


def build_scale_schedule():
    splits = json.loads(SPLITS_PATH.read_text())
    train_tasks = splits["train"]  # All 24 fitting tasks
    dev_tasks = splits["development"]  # All 8 dev tasks
    train_starts = (0, 1, 2, 3)
    dev_starts = (11,)

    entries = []
    for split_name, arms, tasks, starts, embs, seed_base in (
        ("train", TRAIN_ARMS, train_tasks, train_starts, TRAIN_EMBODIMENTS, 21600000),
        ("dev", (DEV_ARM,), dev_tasks, dev_starts, EVAL_EMBODIMENTS, 21700000),
    ):
        official = "train" if split_name == "train" else "development"
        for arm_idx, arm in enumerate(arms):
            for suite, task_ids in sorted(tasks.items()):
                suite_idx = sorted(splits[official]).index(suite)
                for task_id in task_ids:
                    if (
                        task_id not in splits[official][suite]
                        or task_id in splits["test"][suite]
                        or task_id in splits["development" if official == "train" else "train"][suite]
                    ):
                        raise ValueError(f"Task outside registered {split_name} split: {suite}/{task_id}")
                    for init_id in starts:
                        seed = seed_base + 100_000 * arm_idx + 10_000 * suite_idx + 100 * int(task_id) + int(init_id)
                        for spec in embs:
                            entries.append((split_name, arm, suite, int(task_id), int(init_id), seed, spec))
    return entries


def partition_groups(schedule: list, num_shards: int) -> list[list]:
    """Partition by reset group so every shard checks bitwise frame-0 equality within each group."""
    groups = []
    group_map = {}
    for entry in schedule:
        gkey = entry[:5]  # (split, arm, suite, task_id, init_id)
        if gkey not in group_map:
            group_map[gkey] = []
            groups.append(gkey)
        group_map[gkey].append(entry)
    shards = [[] for _ in range(num_shards)]
    for g_idx, gkey in enumerate(groups):
        shards[g_idx % num_shards].extend(group_map[gkey])
    return shards


def validate_scale_archive(archive: dict) -> int:
    schedule = build_scale_schedule()
    expected = {
        (split, arm, suite, task, init, spec.name): (seed, spec)
        for split, arm, suite, task, init, seed, spec in schedule
    }
    modes = archive["modes"]
    seen_paths = set()
    resets = {}
    for row in archive["trials"]:
        if row["split"] == "train" and row["arm"] not in TRAIN_ARMS:
            raise ValueError(f"Forbidden training arm {row['arm']!r}")
        if row["split"] == "dev" and row["arm"] != DEV_ARM:
            raise ValueError(f"Forbidden dev arm {row['arm']!r}")
        group = tuple(row[k] for k in ("split", "arm", "suite", "task_id", "init_state_id"))
        key = (*group, row["embodiment"])
        if key not in expected:
            raise ValueError(f"Unexpected or repeated trial {key}")
        seed, spec = expected.pop(key)
        if (row["seed"], row["regime"], row["yaw_deg"], row["xy_scale"], row["gripper_sign"]) != (
            seed,
            spec.regime,
            spec.yaw_deg,
            spec.xy_scale,
            spec.gripper_sign,
        ):
            raise ValueError(f"Metadata mismatch on {key}")
        identity = tuple(row[k] for k in ("physics", "frame0_img_sha256", "frame0_state_sha256"))
        if resets.setdefault(group, identity) != identity:
            raise ValueError(f"Embodiments differ at reset on {group}")
        for mode in modes:
            p = row[MODE_PATH[mode]]
            if p in seen_paths:
                raise ValueError(f"Reused path {p}")
            seen_paths.add(p)
    if expected:
        raise ValueError(f"Incomplete scale archive: missing {len(expected)} rows")
    return len(resets)


def run_shard(output_dir: Path, shard_idx: int, num_shards: int):
    modes = ("active_probe", "passive_attempt1")
    schedule = build_scale_schedule()
    shards = partition_groups(schedule, num_shards)
    my_entries = shards[shard_idx]

    shard_dir = output_dir / f"shard-{shard_idx}"
    (shard_dir / "tapes").mkdir(parents=True, exist_ok=True)
    policy = load_frozen_pi05()

    trials = []
    prompts = {}
    reset_identities = {}
    max_passive_chunks = SUPPORT_BUDGET // CHUNK_SIZE

    with UR5eSimulator() as simulator:
        for local_idx, (split_name, arm, suite, task_id, init_id, seed, spec) in enumerate(my_entries):
            case = dict(arm=arm, suite=suite, task_id=task_id, variant="BASE", init_state_id=init_id, seed=seed)
            key = (suite, task_id)
            prompt = prompts.get(key)
            dof = 6 if arm == "UR5e" else 7

            row = dict(
                shard=shard_idx,
                index=local_idx,
                split=split_name,
                arm=arm,
                joint_count=dof,
                suite=suite,
                task_id=task_id,
                init_state_id=init_id,
                seed=seed,
                embodiment=spec.name,
                regime=spec.regime,
                yaw_deg=float(spec.yaw_deg),
                xy_scale=float(spec.xy_scale),
                gripper_sign=int(spec.gripper_sign),
            )

            act_meta, act_trace = run_active_probe_episode(simulator, case, spec, prompt=prompt)
            prompts.setdefault(key, act_meta["prompt"])
            prompt = prompts[key]
            chunk_counts(act_trace["valid"], act_meta["controls"])
            act_rel = f"shard-{shard_idx}/tapes/{local_idx:04d}-{split_name}-{arm}-active.npz"
            np.savez_compressed(output_dir / act_rel, **act_trace)
            row.update(
                model_class=act_meta["model_class"],
                **frame0_identity(act_meta["physics"], act_trace),
                active_terminated=act_meta["terminated"],
                active_truncated=act_meta["truncated"],
                active_path=act_rel,
                active_controls=act_meta["controls"],
            )

            pas_rec, pas_trace = run_episode(
                simulator,
                policy,
                case,
                spec,
                prompt=prompt,
                yaw_hat=0.0,
                scale_hat=1.0,
                grip_hat=1,
                max_controls=SUPPORT_BUDGET,
                record_chunks=True,
                max_record_chunks=max_passive_chunks,
            )
            identity = frame0_identity(pas_rec["reset"]["physics"], pas_trace)
            if any(row[k] != v for k, v in identity.items()):
                raise ValueError(f"Active/passive frame-0 mismatch on {case}")
            chunk_counts(pas_trace["valid"], pas_rec["controls"])
            pas_rel = f"shard-{shard_idx}/tapes/{local_idx:04d}-{split_name}-{arm}-passive.npz"
            np.savez_compressed(output_dir / pas_rel, **pas_trace)
            row.update(
                passive_terminated=pas_rec["terminated"],
                passive_truncated=pas_rec["truncated"],
                passive_path=pas_rel,
                passive_controls=pas_rec["controls"],
                passive_success=pas_rec["success"],
                passive_noise_sha256=pas_rec["reset"]["noise_sha256"],
                passive_first_proposal_sha256=pas_rec["opening_proposal_sha256"],
            )

            group = (split_name, arm, suite, task_id, init_id)
            gid = tuple(
                row[k]
                for k in (
                    "physics",
                    "frame0_img_sha256",
                    "frame0_state_sha256",
                    "passive_noise_sha256",
                    "passive_first_proposal_sha256",
                )
            )
            if reset_identities.setdefault(group, gid) != gid:
                raise ValueError(f"Unmatched reset on shard {shard_idx} {group}/{spec.name}")
            trials.append(row)
            if (local_idx + 1) % 25 == 0 or local_idx + 1 == len(my_entries):
                print(
                    json.dumps(dict(shard=shard_idx, done=local_idx + 1, total=len(my_entries), arm=arm, last=spec.name)),
                    flush=True,
                )

    write_json(shard_dir / "shard.json", dict(shard=shard_idx, num_shards=num_shards, modes=list(modes), trials=trials))


def merge_shards(output_dir: Path, num_shards: int):
    all_trials = []
    for s in range(num_shards):
        part = json.loads((output_dir / f"shard-{s}" / "shard.json").read_text())
        all_trials.extend(part["trials"])
    for idx, row in enumerate(all_trials):
        row["index"] = idx
    archive = dict(
        panel="scale24",
        train_arms=list(TRAIN_ARMS),
        dev_arm=DEV_ARM,
        modes=["active_probe", "passive_attempt1"],
        passive_max_controls=SUPPORT_BUDGET,
        matched_reader_controls=SUPPORT_BUDGET,
        num_shards=num_shards,
        trials=all_trials,
    )
    archive["reset_groups"] = validate_scale_archive(archive)
    archive["frame0_bitwise_identical_verified"] = True
    write_json(output_dir / "archive.json", archive)
    print(json.dumps(dict(merged_trials=len(all_trials), reset_groups=archive["reset_groups"]), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--shard-idx", type=int, default=None)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()

    if args.merge:
        merge_shards(args.output, args.num_shards)
    else:
        if args.shard_idx is None:
            parser.error("Provide --shard-idx or --merge")
        run_shard(args.output, args.shard_idx, args.num_shards)


if __name__ == "__main__":
    main()
