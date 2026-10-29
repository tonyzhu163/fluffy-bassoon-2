"""Assemble paired CND/DPT whole trials under the frozen LIBERO task split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .vla_features import FEATURE_KIND
from .vla_history import file_sha256, load_dataset, save_dataset
from .vla_noise_distillation import contract_sha256


FROZEN_SPLIT_SHA256 = "4599ca659a48e94e35a22641f0bb90477d958aed0570d0df46a021ef4492fe08"
CONTEXT_FIELDS = ("observations", "executed_noise", "rewards", "dones", "episode_starts", "teacher_steps")


def assert_paired_contexts(cnd, dpt):
    if len(cnd) != len(dpt):
        raise ValueError("Paired objectives contain different complete trials")
    for first, second in zip(cnd, dpt, strict=True):
        for field in CONTEXT_FIELDS:
            a, b = first[field], second[field]
            if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
                raise ValueError("CND/DPT executed contexts must be bitwise identical; only targets may change")


def load_frozen_split(path, *, expected_sha256=FROZEN_SPLIT_SHA256):
    split = json.loads(Path(path).read_text())
    claimed = split.get("metadata_sha256")
    content = {key: value for key, value in split.items() if key != "metadata_sha256"}
    if (split.get("format") != "libero_pi05_corpus_split_v1" or claimed != expected_sha256
            or contract_sha256(content) != claimed):
        raise ValueError("Frozen task split hash or format differs")
    tasks = split.get("tasks", [])
    if not tasks or len({row["task_key"] for row in tasks}) != len(tasks):
        raise ValueError("Duplicate or missing task identities in frozen split")
    families = {}
    for row in tasks:
        partition = "test" if row["split"].startswith("test") else row["split"]
        previous = families.setdefault(row["variant_base_family"], partition)
        if previous != partition:
            raise ValueError("A base-task family crosses frozen task partitions")
    return split, {row["task_key"]: row for row in tasks}


def merge_cnd_corpus(*, split_path, shard, output_dir, expected_split_sha256=FROZEN_SPLIT_SHA256):
    """Merge a partial or complete shard, retaining global task-level assignment.

    shard = {format: frozen_vla_cnd_corpus_shard_v1, collections: [
      {task_key, collection_manifest, teacher_contract_sha256}, ...]}.
    Both local scene splits inherit their task's GLOBAL split. Source scene
    indices must belong to the frozen source-state bank; auxiliary familiar-task
    validation is not silently mixed into primary held-out-task validation.
    """
    split, tasks = load_frozen_split(split_path, expected_sha256=expected_split_sha256)
    if shard.get("format") != "frozen_vla_cnd_corpus_shard_v1" or not shard.get("collections"):
        raise ValueError("Require an explicit nonempty corpus shard")
    output_dir = Path(output_dir)
    gathered = {objective: {partition: [] for partition in ("train", "validation")}
                for objective in ("cnd", "dpt")}
    ids = {partition: [] for partition in ("train", "validation")}
    sources, common, seen_tasks, seen_collections, seen_trials = [], None, set(), set(), set()
    for entry in shard["collections"]:
        key = entry.get("task_key")
        if key not in tasks or key in seen_tasks:
            raise ValueError("Unknown or duplicate task collection")
        task = tasks[key]
        partition = task["split"]
        if partition not in ("train", "validation") or not task["teacher_collection_allowed"]:
            raise ValueError("Held-out or excluded tasks cannot supply source teacher histories")
        seen_tasks.add(key)
        paired_path = Path(entry["collection_manifest"]).expanduser().resolve()
        paired_sha256 = file_sha256(paired_path)
        if paired_path in seen_collections:
            raise ValueError("Duplicate collection manifest")
        seen_collections.add(paired_path)
        if entry.get("collection_manifest_sha256", paired_sha256) != paired_sha256:
            raise ValueError("Paired collection manifest checksum differs")
        paired = json.loads(paired_path.read_text())
        if paired.get("format") != "frozen_vla_cnd_paired_collection_v1":
            raise ValueError("Require paired CND/DPT source collection")
        benchmark = paired.get("benchmark_contract") or {}
        expected_benchmark = {field: task[field] for field in ("suite", "task_id", "semantic_task", "bddl_sha256")}
        if (any(benchmark.get(field) != value for field, value in expected_benchmark.items())
                or not entry.get("teacher_contract_sha256")
                or benchmark.get("teacher_contract_sha256") != entry["teacher_contract_sha256"]):
            raise ValueError("Source benchmark/teacher provenance differs from frozen task or declared teacher")
        source_identity, cost, all_local_trials = None, None, set()
        source_datasets = {objective: {} for objective in ("cnd", "dpt")}
        local_splits = set(paired.get("trials", {}))
        if local_splits != {"train", "validation"} or set(paired.get("datasets", {})) != {"cnd", "dpt"}:
            raise ValueError("Require both original local scene splits and both paired objectives")
        for local_split in ("train", "validation"):
            scenes = paired["trials"][local_split]
            if (not scenes or any(scene["split"] != local_split
                    or scene["initial_state"] not in task["source_state_indices"] for scene in scenes)):
                raise ValueError("Local trial scene is outside the frozen source-state bank")
            trial_ids = [scene["trial_id"] for scene in scenes]
            if len(set(trial_ids)) != len(trial_ids) or all_local_trials & set(trial_ids):
                raise ValueError("Duplicate source trial identity")
            all_local_trials.update(trial_ids)
            loaded = {}
            for objective in ("cnd", "dpt"):
                receipt = paired["datasets"][objective][local_split]
                path = Path(receipt["path"]).expanduser()
                if not path.is_absolute():
                    path = paired_path.parent / path
                trials, manifest = load_dataset(path, expected_sha256=receipt["sha256"])
                metadata = manifest["metadata"]
                source_contract = metadata["source_contract"]
                if (manifest["objective"] != objective or metadata["trial_ids"] != trial_ids
                        or len(trials) != len(trial_ids) or metadata.get("benchmark_contract") != benchmark
                        or source_contract.get("benchmark_contract") != benchmark
                        or sorted(source_contract.get("scenes", []), key=lambda row: row["trial_id"]) != sorted(
                            [scene for split_name in ("train", "validation") for scene in paired["trials"][split_name]],
                            key=lambda row: row["trial_id"])
                        or metadata.get("source_contract_sha256") != contract_sha256(source_contract)
                        or paired["source_contract_sha256"] != contract_sha256(source_contract)):
                    raise ValueError("Dataset objective, trial identities or source contract provenance differs")
                feature = metadata["feature_contract"]
                contract = dict(motor_sha256=metadata["motor_sha256"], feature_contract=feature,
                    noise_contract=metadata["noise_contract"], chunk=source_contract["chunk"])
                if (feature.get("kind") != FEATURE_KIND or feature.get("model_config") != "pi05_libero"
                        or feature.get("checkpoint_receipt_sha256") != contract["motor_sha256"]
                        or manifest["observation_dim"] != 4104 or manifest["noise_dim"] != 32
                        or paired.get("motor_sha256") != contract["motor_sha256"]
                        or paired.get("feature_contract_sha256") != contract_sha256(feature)
                        or paired.get("noise_contract_sha256") != contract_sha256(contract["noise_contract"])
                        or paired.get("chunk") != contract["chunk"]
                        or any(source_contract[k] != contract[k] for k in ("motor_sha256", "feature_contract", "noise_contract"))):
                    raise ValueError("Require one released pi0.5 motor and common feature-v2/noise/chunk coordinates")
                if common is None:
                    common = contract
                elif common != contract:
                    raise ValueError("Source tasks changed common motor, feature, noise or chunk coordinates")
                identity = {field: source_contract[field] for field in
                            ("teacher_checkpoint_sha256", "teacher_state_sha256", "teacher_steps", "teacher_primitive_steps")}
                if source_identity is None:
                    source_identity = identity
                elif source_identity != identity:
                    raise ValueError("Paired objectives/local splits have mismatched teacher provenance")
                values = {field: metadata[field] for field in
                          ("teacher_primitive_steps", "collection_primitive_steps", "total_source_primitive_steps", "collection_settling_steps")}
                if (any(type(value) is not int or value < 0 for value in values.values())
                        or values["total_source_primitive_steps"] != values["teacher_primitive_steps"] + values["collection_primitive_steps"]
                        or values["teacher_primitive_steps"] != identity["teacher_primitive_steps"]):
                    raise ValueError("Invalid shared source interaction accounting")
                if cost is None:
                    cost = values
                elif cost != values:
                    raise ValueError("Paired exports disagree on shared source interaction costs")
                loaded[objective] = trials
                source_datasets[objective][local_split] = dict(path=str(path.resolve()), sha256=manifest["sha256"])
            assert_paired_contexts(loaded["cnd"], loaded["dpt"])
            for trial_id in trial_ids:
                identity = f"{key}/{trial_id}"
                if identity in seen_trials:
                    raise ValueError("Duplicate globally scoped complete trial identity")
                seen_trials.add(identity)
                ids[partition].append(identity)
            for objective in gathered:
                gathered[objective][partition].extend(loaded[objective])
        if file_sha256(paired_path) != paired_sha256:
            raise ValueError("Paired collection changed while reading")
        sources.append(dict(task_key=key, task_split=partition, variant_base_family=task["variant_base_family"],
            benchmark_contract=benchmark, teacher=source_identity, cost=cost,
            collection_manifest_sha256=paired_sha256, datasets=source_datasets))
    if not all(ids.values()):
        raise ValueError("A partial corpus shard still requires nonempty training and validation task trials")
    # All source validation completes before a final output is created.
    output_dir.mkdir(parents=True, exist_ok=False)
    total_cost = sum(source["cost"]["total_source_primitive_steps"] for source in sources)
    paths = {objective: {} for objective in gathered}
    for objective in gathered:
        for partition in gathered[objective]:
            path = output_dir / f"{objective}-{partition}.npz"
            save_dataset(path, gathered[objective][partition], objective=objective, metadata=dict(**common,
                trial_ids=ids[partition], task_split=partition, frozen_split_sha256=split["metadata_sha256"],
                source_collections=sources, total_source_primitive_steps=total_cost,
                source_cost_by_task_split={p: sum(s["cost"]["total_source_primitive_steps"] for s in sources
                    if s["task_split"] == p) for p in ids},
                split_rule="whole semantic task families and whole trials before any windows; local scene splits inherit task assignment"))
            paths[objective][partition] = str(path.resolve())
    summary = dict(format="frozen_vla_cnd_corpus_v1", datasets=paths,
        frozen_split_sha256=split["metadata_sha256"], partial_shard=len(seen_tasks) < len([
            task for task in tasks.values() if task["split"] in ("train", "validation")]),
        included_tasks=sorted(seen_tasks), full_split_counts=split["counts"],
        trials_by_split={key: len(value) for key, value in ids.items()},
        total_source_primitive_steps=total_cost,
        source_cost_rule="one shared teacher/collection cost per task, counted once across CND/DPT and local scene splits",
        source_collections=sources)
    (output_dir / "collections.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge_cnd_corpus(split_path=args.split, shard=json.loads(args.shard.read_text()),
                                    output_dir=args.output), indent=2))


if __name__ == "__main__":
    main()
